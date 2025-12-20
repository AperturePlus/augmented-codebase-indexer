import asyncio

import httpx

from aci.infrastructure.vector_store import SearchResult
from aci.services.reranker import OpenAICompatibleReranker


async def _make_reranker(response_json, status_code=200):
    # Mock transport to return provided JSON
    async def handler(request):
        return httpx.Response(status_code, json=response_json, request=request)

    transport = httpx.MockTransport(handler)
    reranker = OpenAICompatibleReranker(
        api_url="https://example.com",
        api_key="test",
        model="test-model",
        endpoint="/v1/rerank",
    )
    # Inject mock client
    reranker._client = httpx.AsyncClient(
        base_url="https://example.com",
        transport=transport,
        headers=reranker._client.headers,
        timeout=reranker._client.timeout,
    )
    return reranker


def _make_candidates(n=3):
    return [
        SearchResult(
            chunk_id=str(i),
            file_path=f"/f{i}.py",
            start_line=1,
            end_line=2,
            content=f"code {i}",
            score=0.0,
            metadata={},
        )
        for i in range(n)
    ]


def test_reranker_parses_results_with_relevance_score():
    candidates = _make_candidates(3)
    response = {
        "results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
        ]
    }

    async def run():
        rr = await _make_reranker(response)
        results = await rr.rerank("q", candidates, top_k=2)
        await rr.aclose()
        return results

    results = asyncio.run(run())
    assert [r.chunk_id for r in results] == ["2", "0"]
    assert results[0].score == 0.9


def test_reranker_falls_back_when_no_data():
    candidates = _make_candidates(2)
    response = {"meta": {}}

    async def run():
        rr = await _make_reranker(response)
        results = await rr.rerank("q", candidates, top_k=2)
        await rr.aclose()
        return results

    results = asyncio.run(run())
    # Should return original order when no data/results
    assert [r.chunk_id for r in results] == ["0", "1"]
