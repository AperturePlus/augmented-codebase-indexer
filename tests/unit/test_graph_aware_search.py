"""Unit tests for graph-aware search integration (Task 12.2).

Tests that ``SearchService.search(include_graph_context=...)`` correctly
delegates to ``ContextAssembler.enrich_search_results()`` when an
assembler is available, and silently returns unenriched results when it
is not.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aci.core.graph_models import (
    ContextMetadata,
    ContextPackage,
    QueryRequest,
    SymbolDetail,
)
from aci.infrastructure.vector_store import SearchResult
from aci.services.search_service import SearchService

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_search_result(
    chunk_id: str = "chunk_1",
    file_path: str = "test.py",
    content: str = "def foo(): pass",
    score: float = 0.9,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        file_path=file_path,
        start_line=1,
        end_line=5,
        content=content,
        score=score,
        metadata={},
    )


def _make_context_package(query: str = "test") -> ContextPackage:
    return ContextPackage(
        query=query,
        symbols=[
            SymbolDetail(
                fqn="mod.foo",
                source_code="def foo(): pass",
                summary="A test function.",
                callers=["mod.bar"],
                callees=[],
                pagerank_score=0.5,
            ),
        ],
        file_summaries=[],
        metadata=ContextMetadata(
            query_params={"query": query},
            symbol_count=1,
            total_tokens=10,
            pagerank_score_range=(0.5, 0.5),
        ),
    )


def _make_embedding_client() -> MagicMock:
    client = AsyncMock()
    client.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    return client


def _make_vector_store(results: list[SearchResult] | None = None) -> MagicMock:
    store = AsyncMock()
    store.search = AsyncMock(return_value=results or [])
    store.get_all_file_paths = AsyncMock(return_value=[])
    return store


def _make_context_assembler(
    package: ContextPackage | None = None,
) -> MagicMock:
    assembler = MagicMock()
    assembler.enrich_search_results = AsyncMock(
        return_value=package or _make_context_package(),
    )
    return assembler


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestIncludeGraphContextFalse:
    """include_graph_context=False returns normal SearchResult list."""

    @pytest.mark.asyncio
    async def test_returns_normal_results_without_enrichment(self) -> None:
        results = [_make_search_result()]
        service = SearchService(
            embedding_client=_make_embedding_client(),
            vector_store=_make_vector_store(results),
            context_assembler=_make_context_assembler(),
        )

        out = await service.search("test query", include_graph_context=False)

        assert isinstance(out, list)
        assert len(out) == 1
        assert out[0].chunk_id == "chunk_1"

    @pytest.mark.asyncio
    async def test_assembler_not_called_when_false(self) -> None:
        assembler = _make_context_assembler()
        service = SearchService(
            embedding_client=_make_embedding_client(),
            vector_store=_make_vector_store([_make_search_result()]),
            context_assembler=assembler,
        )

        await service.search("test query", include_graph_context=False)

        assembler.enrich_search_results.assert_not_called()

    @pytest.mark.asyncio
    async def test_default_is_false(self) -> None:
        """Omitting include_graph_context should behave like False."""
        assembler = _make_context_assembler()
        service = SearchService(
            embedding_client=_make_embedding_client(),
            vector_store=_make_vector_store([_make_search_result()]),
            context_assembler=assembler,
        )

        out = await service.search("test query")

        assert isinstance(out, list)
        assembler.enrich_search_results.assert_not_called()


class TestIncludeGraphContextTrueWithAssembler:
    """include_graph_context=True with assembler returns ContextPackage."""

    @pytest.mark.asyncio
    async def test_returns_context_package(self) -> None:
        pkg = _make_context_package("my query")
        assembler = _make_context_assembler(pkg)
        service = SearchService(
            embedding_client=_make_embedding_client(),
            vector_store=_make_vector_store([_make_search_result()]),
            context_assembler=assembler,
        )

        out = await service.search("my query", include_graph_context=True)

        assert isinstance(out, ContextPackage)
        assert out.query == "my query"
        assert len(out.symbols) == 1
        assert out.symbols[0].fqn == "mod.foo"

    @pytest.mark.asyncio
    async def test_assembler_called_with_results_and_request(self) -> None:
        results = [
            _make_search_result("c1"),
            _make_search_result("c2", file_path="other.py", score=0.8),
        ]
        assembler = _make_context_assembler()
        service = SearchService(
            embedding_client=_make_embedding_client(),
            vector_store=_make_vector_store(results),
            context_assembler=assembler,
        )

        await service.search("find me", include_graph_context=True)

        assembler.enrich_search_results.assert_called_once()
        call_args = assembler.enrich_search_results.call_args
        passed_results = call_args[0][0]
        passed_request = call_args[0][1]

        assert len(passed_results) == 2
        assert isinstance(passed_request, QueryRequest)
        assert passed_request.query == "find me"
        assert passed_request.include_graph_context is True


class TestIncludeGraphContextTrueWithoutAssembler:
    """include_graph_context=True without assembler returns unenriched results."""

    @pytest.mark.asyncio
    async def test_returns_plain_results(self) -> None:
        results = [_make_search_result()]
        service = SearchService(
            embedding_client=_make_embedding_client(),
            vector_store=_make_vector_store(results),
            context_assembler=None,
        )

        out = await service.search("test query", include_graph_context=True)

        assert isinstance(out, list)
        assert len(out) == 1
        assert out[0].chunk_id == "chunk_1"

    @pytest.mark.asyncio
    async def test_no_assembler_by_default(self) -> None:
        """SearchService without context_assembler kwarg has None assembler."""
        service = SearchService(
            embedding_client=_make_embedding_client(),
            vector_store=_make_vector_store([_make_search_result()]),
        )

        out = await service.search("test query", include_graph_context=True)

        assert isinstance(out, list)
