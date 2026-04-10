"""Unit tests for LLMEnricher."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from aci.core.graph_models import (
    SymbolIndexEntry,
    SymbolLocation,
)
from aci.core.parsers.base import SymbolReference
from aci.services.llm_enricher import LLMEnricher

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


@dataclass
class _FakeLLMConfig:
    """Minimal stand-in for LLMConfig used in tests."""

    enabled: bool = False
    api_url: str = ""
    api_key: str = ""
    model: str = "test-model"
    batch_size: int = 2
    timeout: float = 5.0
    confidence_threshold: float = 0.5


def _make_config(
    *,
    enabled: bool = True,
    api_url: str = "https://api.example.com",
    api_key: str = "sk-test",
    **kwargs: Any,
) -> _FakeLLMConfig:
    return _FakeLLMConfig(enabled=enabled, api_url=api_url, api_key=api_key, **kwargs)


def _make_symbol(fqn: str = "mod.func") -> SymbolIndexEntry:
    return SymbolIndexEntry(
        fqn=fqn,
        definition=SymbolLocation(file_path="fake.py", start_line=1, end_line=5),
        graph_node_id=fqn,
    )


def _make_ref(name: str = "unknown_func") -> SymbolReference:
    return SymbolReference(
        name=name,
        ref_type="call",
        file_path="caller.py",
        line=10,
        parent_symbol="mod.caller",
    )


def _chat_response(results: list[dict], model: str = "test-model") -> dict:
    """Build a fake OpenAI-compatible chat completion response."""
    return {
        "choices": [
            {"message": {"content": json.dumps(results)}}
        ],
        "model": model,
        "usage": {"total_tokens": 42},
    }


def _mock_summary_generator() -> MagicMock:
    gen = MagicMock()
    gen.generate_function_summary = MagicMock()
    gen.generate_class_summary = MagicMock()
    gen.generate_file_summary = MagicMock()
    return gen


# ------------------------------------------------------------------
# Disabled mode (Req 7.6, 12.6)
# ------------------------------------------------------------------


class TestDisabledMode:
    """When disabled, no API calls should be made."""

    def test_disabled_when_config_disabled(self) -> None:
        cfg = _make_config(enabled=False)
        enricher = LLMEnricher(cfg, _mock_summary_generator())
        assert enricher.enabled is False

    def test_disabled_when_api_key_empty(self) -> None:
        cfg = _make_config(enabled=True, api_key="")
        enricher = LLMEnricher(cfg, _mock_summary_generator())
        assert enricher.enabled is False

    def test_disabled_when_api_url_empty(self) -> None:
        cfg = _make_config(enabled=True, api_url="")
        enricher = LLMEnricher(cfg, _mock_summary_generator())
        assert enricher.enabled is False

    @pytest.mark.asyncio
    async def test_enrich_symbols_returns_unchanged(self) -> None:
        cfg = _make_config(enabled=False)
        enricher = LLMEnricher(cfg, _mock_summary_generator())
        syms = [_make_symbol("a.b"), _make_symbol("c.d")]
        result = await enricher.enrich_symbols(syms)
        assert result is syms  # same object, untouched

    @pytest.mark.asyncio
    async def test_infer_edges_returns_empty(self) -> None:
        cfg = _make_config(enabled=False)
        enricher = LLMEnricher(cfg, _mock_summary_generator())
        result = await enricher.infer_edges([_make_ref()])
        assert result == []

    def test_no_httpx_client_created(self) -> None:
        cfg = _make_config(enabled=False)
        enricher = LLMEnricher(cfg, _mock_summary_generator())
        assert enricher._client is None  # noqa: SLF001


# ------------------------------------------------------------------
# Enabled mode – enrich_symbols (Req 7.1, 7.4)
# ------------------------------------------------------------------


class TestEnrichSymbols:
    """Test LLM-powered symbol summarisation."""

    @pytest.mark.asyncio
    async def test_single_batch_enrichment(self) -> None:
        cfg = _make_config(batch_size=10)
        enricher = LLMEnricher(cfg, _mock_summary_generator())

        sym = _make_symbol("mod.func")
        llm_results = [{"fqn": "mod.func", "summary": "Does stuff."}]

        mock_resp = httpx.Response(
            200,
            json=_chat_response(llm_results),
            request=httpx.Request("POST", "https://api.example.com/v1/chat/completions"),
        )
        enricher._client = AsyncMock()  # noqa: SLF001
        enricher._client.post = AsyncMock(return_value=mock_resp)  # noqa: SLF001

        result = await enricher.enrich_symbols([sym])
        assert len(result) == 1
        assert result[0].llm_summary == "Does stuff."

    @pytest.mark.asyncio
    async def test_batch_processing_splits_correctly(self) -> None:
        """With batch_size=2 and 3 symbols, two LLM calls should be made."""
        cfg = _make_config(batch_size=2)
        enricher = LLMEnricher(cfg, _mock_summary_generator())

        syms = [_make_symbol(f"mod.f{i}") for i in range(3)]

        call_count = 0

        async def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            # Return empty summaries — we just care about call count
            return httpx.Response(
                200,
                json=_chat_response([]),
                request=httpx.Request("POST", "https://api.example.com/v1/chat/completions"),
            )

        enricher._client = AsyncMock()  # noqa: SLF001
        enricher._client.post = fake_post  # noqa: SLF001

        result = await enricher.enrich_symbols(syms)
        assert len(result) == 3
        assert call_count == 2  # ceil(3/2)

    @pytest.mark.asyncio
    async def test_empty_symbols_returns_empty(self) -> None:
        cfg = _make_config()
        enricher = LLMEnricher(cfg, _mock_summary_generator())
        result = await enricher.enrich_symbols([])
        assert result == []


# ------------------------------------------------------------------
# Fallback on LLM error (Req 7.5)
# ------------------------------------------------------------------


class TestFallbackOnError:
    """On LLM failure, symbols should be returned without LLM summaries."""

    @pytest.mark.asyncio
    async def test_enrich_symbols_falls_back_on_http_error(self) -> None:
        cfg = _make_config(batch_size=10)
        enricher = LLMEnricher(cfg, _mock_summary_generator())

        sym = _make_symbol("mod.func")

        async def failing_post(*args: Any, **kwargs: Any) -> httpx.Response:
            raise httpx.HTTPStatusError(
                "Server error",
                request=httpx.Request("POST", "https://api.example.com/v1/chat/completions"),
                response=httpx.Response(500),
            )

        enricher._client = AsyncMock()  # noqa: SLF001
        enricher._client.post = failing_post  # noqa: SLF001

        result = await enricher.enrich_symbols([sym])
        # Symbol returned unchanged (no crash, no llm_summary set)
        assert len(result) == 1
        assert result[0].llm_summary == ""

    @pytest.mark.asyncio
    async def test_enrich_symbols_falls_back_on_json_decode_error(self) -> None:
        cfg = _make_config(batch_size=10)
        enricher = LLMEnricher(cfg, _mock_summary_generator())

        sym = _make_symbol("mod.func")

        # Return a response whose content is not valid JSON
        bad_response = {
            "choices": [{"message": {"content": "not json at all"}}],
            "model": "test-model",
            "usage": {"total_tokens": 1},
        }
        mock_resp = httpx.Response(
            200,
            json=bad_response,
            request=httpx.Request("POST", "https://api.example.com/v1/chat/completions"),
        )
        enricher._client = AsyncMock()  # noqa: SLF001
        enricher._client.post = AsyncMock(return_value=mock_resp)  # noqa: SLF001

        result = await enricher.enrich_symbols([sym])
        # Bad JSON → empty results list → symbol returned without llm_summary
        assert len(result) == 1
        assert result[0].llm_summary == ""


# ------------------------------------------------------------------
# Edge inference (Req 8.1, 8.2, 8.4)
# ------------------------------------------------------------------


class TestInferEdges:
    """Test LLM-powered edge inference."""

    @pytest.mark.asyncio
    async def test_inferred_edges_tagged_correctly(self) -> None:
        cfg = _make_config(confidence_threshold=0.5)
        enricher = LLMEnricher(cfg, _mock_summary_generator())

        llm_results = [
            {"source": "a.caller", "target": "b.callee", "confidence": 0.9},
        ]
        mock_resp = httpx.Response(
            200,
            json=_chat_response(llm_results),
            request=httpx.Request("POST", "https://api.example.com/v1/chat/completions"),
        )
        enricher._client = AsyncMock()  # noqa: SLF001
        enricher._client.post = AsyncMock(return_value=mock_resp)  # noqa: SLF001

        edges = await enricher.infer_edges([_make_ref("unknown")])
        assert len(edges) == 1
        edge = edges[0]
        assert edge.inferred is True
        assert edge.confidence == 0.9
        assert edge.edge_type == "inferred"
        assert edge.source_id == "a.caller"
        assert edge.target_id == "b.callee"

    @pytest.mark.asyncio
    async def test_low_confidence_edges_discarded(self) -> None:
        cfg = _make_config(confidence_threshold=0.5)
        enricher = LLMEnricher(cfg, _mock_summary_generator())

        llm_results = [
            {"source": "a.x", "target": "b.y", "confidence": 0.3},  # below threshold
            {"source": "a.x", "target": "c.z", "confidence": 0.8},  # above threshold
        ]
        mock_resp = httpx.Response(
            200,
            json=_chat_response(llm_results),
            request=httpx.Request("POST", "https://api.example.com/v1/chat/completions"),
        )
        enricher._client = AsyncMock()  # noqa: SLF001
        enricher._client.post = AsyncMock(return_value=mock_resp)  # noqa: SLF001

        edges = await enricher.infer_edges([_make_ref()])
        assert len(edges) == 1
        assert edges[0].target_id == "c.z"

    @pytest.mark.asyncio
    async def test_infer_edges_returns_empty_on_error(self) -> None:
        cfg = _make_config()
        enricher = LLMEnricher(cfg, _mock_summary_generator())

        async def failing_post(*args: Any, **kwargs: Any) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        enricher._client = AsyncMock()  # noqa: SLF001
        enricher._client.post = failing_post  # noqa: SLF001

        edges = await enricher.infer_edges([_make_ref()])
        assert edges == []

    @pytest.mark.asyncio
    async def test_infer_edges_empty_input(self) -> None:
        cfg = _make_config()
        enricher = LLMEnricher(cfg, _mock_summary_generator())
        edges = await enricher.infer_edges([])
        assert edges == []


# ------------------------------------------------------------------
# close() (resource cleanup)
# ------------------------------------------------------------------


class TestClose:
    """Test httpx client cleanup."""

    @pytest.mark.asyncio
    async def test_close_cleans_up_client(self) -> None:
        cfg = _make_config()
        enricher = LLMEnricher(cfg, _mock_summary_generator())
        assert enricher._client is not None  # noqa: SLF001

        await enricher.close()
        assert enricher._client is None  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_close_noop_when_disabled(self) -> None:
        cfg = _make_config(enabled=False)
        enricher = LLMEnricher(cfg, _mock_summary_generator())
        # Should not raise
        await enricher.close()
        assert enricher._client is None  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_close_idempotent(self) -> None:
        cfg = _make_config()
        enricher = LLMEnricher(cfg, _mock_summary_generator())
        await enricher.close()
        await enricher.close()  # second call should be safe
        assert enricher._client is None  # noqa: SLF001
