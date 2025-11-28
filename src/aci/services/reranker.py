"""
Rerankers for Project ACI.

Provides API-based and lightweight rerankers.
"""

import logging
from typing import List

import httpx

from aci.infrastructure.vector_store import SearchResult
from aci.services.search_service import RerankerInterface

logger = logging.getLogger(__name__)


class OpenAICompatibleReranker(RerankerInterface):
    """
    Reranker backed by an OpenAI-compatible HTTP API.

    Expects a `/v1/rerank` endpoint that accepts:
    {
        "model": "<model>",
        "query": "<query>",
        "documents": ["doc1", "doc2", ...],
        "top_n": <int>
    }
    and returns:
    {
        "data": [{"index": 0, "score": 0.9}, ...]
    }
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        endpoint: str = "/v1/rerank",
    ):
        self._model = model
        self._endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self._client = httpx.AsyncClient(
            base_url=api_url.rstrip("/"),
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    async def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        if not candidates:
            return []

        # API may reject top_n > doc count; keep it bounded
        top_n = min(top_k, len(candidates))

        payload = {
            "model": self._model,
            "query": query,
            "documents": [c.content for c in candidates],
            "top_n": top_n,
        }

        endpoint = self._endpoint
        try:
            response = await self._client.post(endpoint, json=payload)
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Rerank HTTP error: {exc} (base={self._client.base_url}, endpoint={endpoint})"
            ) from exc

        if response.status_code != 200:
            full_url = str(response.request.url)
            raise RuntimeError(
                f"Rerank API error: status={response.status_code}, url={full_url}, body={response.text}"
            )

        parsed = response.json()
        data = parsed.get("data", [])
        if not data:
            # Some providers return `results` instead of `data`
            data = parsed.get("results", [])
        if not data:
            logger.warning(
                "Rerank returned no data; falling back to original order "
                "(url=%s, endpoint=%s, status=%s, response=%s)",
                self._client.base_url,
                endpoint,
                response.status_code,
                parsed,
            )
            return candidates[:top_k]
        scored = []
        for item in data:
            idx = item.get("index")
            score = item.get("score") or item.get("relevance_score")
            if idx is None or idx >= len(candidates):
                continue
            base = candidates[idx]
            scored.append(
                SearchResult(
                    chunk_id=base.chunk_id,
                    file_path=base.file_path,
                    start_line=base.start_line,
                    end_line=base.end_line,
                    content=base.content,
                    score=float(score) if score is not None else base.score,
                    metadata=base.metadata,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    async def aclose(self) -> None:
        await self._client.aclose()


class SimpleReranker(RerankerInterface):
    """
    Simple reranker for testing - just returns top_k candidates.

    Useful for testing without loading heavy ML models.
    """

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Return top_k candidates without re-ranking."""
        return candidates[:top_k]
