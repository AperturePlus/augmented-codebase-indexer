"""
LLM enricher for semantic code intelligence.

Provides LLM-powered symbol summarisation and relationship inference.
When disabled (the default), all methods return fallback results without
making any API calls.  On LLM errors the enricher falls back to the
existing template-based :class:`SummaryGeneratorInterface`.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import httpx

from aci.core.graph_models import (
    GraphEdge,
    LLMEnrichRequest,
    LLMEnrichResponse,
    SymbolIndexEntry,
)

if TYPE_CHECKING:
    from aci.core.config import LLMConfig
    from aci.core.parsers.base import SymbolReference
    from aci.core.summary_generator import SummaryGeneratorInterface

logger = logging.getLogger(__name__)


class LLMEnricher:
    """Optional LLM-powered enrichment for summaries and edge inference.

    When *disabled* (``config.enabled is False``, or ``api_key`` / ``api_url``
    are empty), every public method returns a safe fallback value and no
    network calls are made.
    """

    def __init__(
        self,
        config: LLMConfig,
        summary_generator: SummaryGeneratorInterface,
    ) -> None:
        self._enabled = config.enabled and bool(config.api_key) and bool(config.api_url)
        self._client: httpx.AsyncClient | None = None
        if self._enabled:
            self._client = httpx.AsyncClient(
                base_url=config.api_url,
                headers={"Authorization": f"Bearer {config.api_key}"},
                timeout=config.timeout,
            )
        else:
            logger.info("LLM enrichment disabled: API key or URL not configured")

        self._model = config.model
        self._batch_size = config.batch_size
        self._confidence_threshold = config.confidence_threshold
        self._fallback_generator = summary_generator

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Return ``True`` when the enricher will make real LLM calls."""
        return self._enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def enrich_symbols(
        self,
        symbols: list[SymbolIndexEntry],
    ) -> list[SymbolIndexEntry]:
        """Generate LLM summaries for *symbols*.

        Processes symbols in batches of ``config.batch_size``.  On any LLM
        error the affected batch falls back to the template-based summary
        generator (Req 7.5).

        When disabled, returns *symbols* unchanged.
        """
        if not self._enabled or not symbols:
            return symbols

        enriched: list[SymbolIndexEntry] = []
        for batch_start in range(0, len(symbols), self._batch_size):
            batch = symbols[batch_start : batch_start + self._batch_size]
            try:
                response = await self._call_llm_summarize(batch)
                summary_map: dict[str, str] = {
                    r["fqn"]: r.get("summary", "") for r in response.results
                }
                for sym in batch:
                    if sym.fqn in summary_map and summary_map[sym.fqn]:
                        sym.llm_summary = summary_map[sym.fqn]
                    enriched.append(sym)
            except Exception:
                logger.warning(
                    "LLM enrichment failed for batch starting at %d; "
                    "falling back to template summaries",
                    batch_start,
                    exc_info=True,
                )
                enriched.extend(batch)

        return enriched

    async def infer_edges(
        self,
        unresolved: list[SymbolReference],
    ) -> list[GraphEdge]:
        """Infer probable edges for *unresolved* references via LLM.

        Each returned edge is tagged with ``inferred=True`` and a confidence
        score.  Edges below ``config.confidence_threshold`` are discarded and
        logged at debug level (Req 8.4).

        When disabled, returns an empty list.
        """
        if not self._enabled or not unresolved:
            return []

        try:
            response = await self._call_llm_infer(unresolved)
        except Exception:
            logger.warning(
                "LLM edge inference failed; returning no inferred edges",
                exc_info=True,
            )
            return []

        edges: list[GraphEdge] = []
        for result in response.results:
            confidence = float(result.get("confidence", 0.0))
            if confidence < self._confidence_threshold:
                logger.debug(
                    "Discarding low-confidence inferred edge %s -> %s (%.2f < %.2f)",
                    result.get("source", "?"),
                    result.get("target", "?"),
                    confidence,
                    self._confidence_threshold,
                )
                continue
            edges.append(
                GraphEdge(
                    source_id=result.get("source", ""),
                    target_id=result.get("target", ""),
                    edge_type="inferred",
                    inferred=True,
                    confidence=confidence,
                )
            )
        return edges

    async def close(self) -> None:
        """Release the underlying HTTP client, if any."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_llm_summarize(
        self,
        symbols: list[SymbolIndexEntry],
    ) -> LLMEnrichResponse:
        """Send a summarisation request to the LLM endpoint."""
        assert self._client is not None  # guarded by self._enabled

        request = LLMEnrichRequest(
            artifacts=[
                {
                    "fqn": s.fqn,
                    "source": self._read_source(s),
                    "type": "symbol",
                }
                for s in symbols
            ],
            task="summarize",
        )
        return await self._post(request)

    async def _call_llm_infer(
        self,
        unresolved: list[SymbolReference],
    ) -> LLMEnrichResponse:
        """Send an edge-inference request to the LLM endpoint."""
        assert self._client is not None

        request = LLMEnrichRequest(
            artifacts=[
                {
                    "fqn": ref.name,
                    "source": "",
                    "type": ref.ref_type,
                    "file_path": ref.file_path,
                    "line": ref.line,
                    "parent_symbol": ref.parent_symbol or "",
                }
                for ref in unresolved
            ],
            task="infer_edges",
        )
        return await self._post(request)

    async def _post(self, request: LLMEnrichRequest) -> LLMEnrichResponse:
        """POST a request to the OpenAI-compatible chat completions endpoint."""
        assert self._client is not None

        messages = [
            {
                "role": "system",
                "content": self._system_prompt(request.task),
            },
            {
                "role": "user",
                "content": json.dumps(request.artifacts),
            },
        ]

        resp = await self._client.post(
            "/v1/chat/completions",
            json={
                "model": self._model,
                "messages": messages,
                "temperature": 0.2,
            },
        )
        resp.raise_for_status()

        body: dict[str, Any] = resp.json()
        content_str: str = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "[]")
        )

        try:
            results = json.loads(content_str)
        except json.JSONDecodeError:
            results = []

        tokens_used = body.get("usage", {}).get("total_tokens", 0)
        return LLMEnrichResponse(
            results=results if isinstance(results, list) else [],
            model=body.get("model", self._model),
            tokens_used=tokens_used,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _read_source(entry: SymbolIndexEntry) -> str:
        """Best-effort read of the source code for a symbol."""
        try:
            from pathlib import Path

            path = Path(entry.definition.file_path)
            if not path.is_file():
                return ""
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(0, entry.definition.start_line - 1)
            end = entry.definition.end_line
            return "\n".join(lines[start:end])
        except Exception:
            return ""

    @staticmethod
    def _system_prompt(task: str) -> str:
        """Return the system prompt for the given task type."""
        if task == "summarize":
            return (
                "You are a code analysis assistant. For each code artifact "
                "provided, generate a concise natural-language summary that "
                "describes its purpose, parameters, return value, and side "
                "effects. Respond with a JSON array of objects, each with "
                '"fqn" and "summary" keys.'
            )
        # infer_edges
        return (
            "You are a code analysis assistant. For each unresolved symbol "
            "reference provided, infer the most probable target definition "
            "using naming conventions and code context. Respond with a JSON "
            'array of objects, each with "source", "target", and '
            '"confidence" (0.0-1.0) keys.'
        )
