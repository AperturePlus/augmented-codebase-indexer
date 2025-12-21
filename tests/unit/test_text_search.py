from pathlib import Path

import pytest

from aci.infrastructure.fakes import InMemoryVectorStore, LocalEmbeddingClient
from aci.infrastructure.grep_searcher import GrepSearcher, TextSearchMode
from aci.services import SearchMode, SearchService, TextSearchOptions


@pytest.mark.asyncio
async def test_grep_searcher_substring_all_terms(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("foo bar\nbar foo\nfoo\n", encoding="utf-8")

    searcher = GrepSearcher(base_path=str(tmp_path))
    results = await searcher.search(
        query="foo bar",
        file_paths=[str(file_path)],
        mode=TextSearchMode.SUBSTRING,
        all_terms=True,
    )

    assert len(results) == 2
    assert results[0].metadata.get("source") == "grep"
    assert results[0].metadata.get("text_mode") == "substring"


@pytest.mark.asyncio
async def test_grep_searcher_regex(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world\nHELLO   WORLD\n", encoding="utf-8")

    searcher = GrepSearcher(base_path=str(tmp_path))
    results = await searcher.search(
        query=r"hello\s+world",
        file_paths=[str(file_path)],
        mode=TextSearchMode.REGEX,
    )

    assert len(results) == 2
    assert results[0].metadata.get("source") == "grep"
    assert results[0].metadata.get("text_mode") == "regex"


@pytest.mark.asyncio
async def test_grep_searcher_fuzzy_scores(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.js"
    file_path.write_text(
        "const x = useEffect(() => {});\nconst y = somethingElse();\n",
        encoding="utf-8",
    )

    searcher = GrepSearcher(base_path=str(tmp_path))
    results = await searcher.search(
        query="useefct",
        file_paths=[str(file_path)],
        mode=TextSearchMode.FUZZY,
    )

    assert results
    assert results[0].metadata.get("source") == "fuzzy"
    assert results[0].score >= 0.6


@pytest.mark.asyncio
async def test_search_service_fuzzy_mode_uses_indexed_file_list(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.js"
    file_path.write_text("const x = useEffect(() => {});\n", encoding="utf-8")

    vector_store = InMemoryVectorStore()
    await vector_store.upsert(
        chunk_id="chunk:1",
        vector=[0.0],
        payload={
            "file_path": str(file_path),
            "start_line": 1,
            "end_line": 1,
            "content": "dummy",
            "artifact_type": "chunk",
        },
    )

    search_service = SearchService(
        embedding_client=LocalEmbeddingClient(),
        vector_store=vector_store,
        reranker=None,
        grep_searcher=GrepSearcher(base_path=str(tmp_path)),
        default_limit=10,
    )

    results = await search_service.search(
        query="useefct",
        search_mode=SearchMode.FUZZY,
        use_rerank=False,
        text_options=TextSearchOptions(fuzzy_min_score=0.6),
    )

    assert results
    assert results[0].file_path == str(file_path)
    assert results[0].metadata.get("source") == "fuzzy"
