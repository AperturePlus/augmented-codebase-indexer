"""
Property-based tests for Reranker components.

Tests the correctness properties for re-ranking operations including
result subset validation and limit compliance.

Uses SimpleReranker for testing without loading heavy ML models.
"""


from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aci.infrastructure.vector_store import SearchResult
from aci.services.reranker import SimpleReranker


# Strategies for generating test data
@st.composite
def search_result(draw):
    """Generate a valid SearchResult."""
    chunk_id = draw(st.uuids().map(str))
    file_path = draw(st.from_regex(r"/[a-z]+/[a-z_]+\.py", fullmatch=True))
    start_line = draw(st.integers(min_value=1, max_value=100))
    end_line = draw(st.integers(min_value=start_line, max_value=start_line + 50))
    content = draw(
        st.text(
            min_size=10,
            max_size=200,
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        )
    )
    score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))

    return SearchResult(
        chunk_id=chunk_id,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        score=score,
        metadata={},
    )


@st.composite
def search_results_list(draw, min_size=1, max_size=20):
    """Generate a list of SearchResults with unique chunk_ids."""
    results = draw(
        st.lists(
            search_result(),
            min_size=min_size,
            max_size=max_size,
        )
    )

    # Ensure unique chunk_ids
    seen_ids = set()
    unique_results = []
    for r in results:
        if r.chunk_id not in seen_ids:
            seen_ids.add(r.chunk_id)
            unique_results.append(r)

    return unique_results


class TestRerankerResultSubset:
    """
    **Feature: codebase-semantic-search, Property 14a: Reranker Result Subset**
    **Validates: Requirements 4.2**

    *For any* search with re-ranking enabled, the Reranker's returned results
    should be a subset of the vector recall candidate set (no new results introduced).
    """

    @given(
        candidates=search_results_list(min_size=1, max_size=20),
        query=st.text(
            min_size=5,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        ),
        top_k=st.integers(min_value=1, max_value=30),
    )
    @settings(
        max_examples=50,
        deadline=10000,
    )
    def test_reranked_results_are_subset_of_candidates(self, candidates, query, top_k):
        """Reranked results should only contain items from the original candidates."""
        assume(query.strip())  # Non-empty query
        assume(len(candidates) > 0)  # At least one candidate

        reranker = SimpleReranker()

        # Get reranked results
        results = reranker.rerank(query, candidates, top_k)

        # Get chunk_ids from candidates and results
        candidate_ids: set[str] = {c.chunk_id for c in candidates}
        result_ids: set[str] = {r.chunk_id for r in results}

        # Results should be a subset of candidates
        assert result_ids.issubset(candidate_ids), (
            f"Reranked results contain items not in candidates: {result_ids - candidate_ids}"
        )

    @given(
        candidates=search_results_list(min_size=5, max_size=15),
        query=st.text(
            min_size=5,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        ),
        top_k=st.integers(min_value=1, max_value=10),
    )
    @settings(
        max_examples=30,
        deadline=10000,
    )
    def test_reranked_results_preserve_content(self, candidates, query, top_k):
        """Reranked results should preserve the original content of candidates."""
        assume(query.strip())  # Non-empty query
        assume(len(candidates) > 0)

        reranker = SimpleReranker()

        # Create a mapping of chunk_id to original candidate
        candidate_map = {c.chunk_id: c for c in candidates}

        # Get reranked results
        results = reranker.rerank(query, candidates, top_k)

        # Verify each result matches its original candidate
        for result in results:
            assert result.chunk_id in candidate_map, (
                f"Result chunk_id {result.chunk_id} not found in candidates"
            )

            original = candidate_map[result.chunk_id]

            # Content should be preserved
            assert result.file_path == original.file_path, (
                f"file_path mismatch: {result.file_path} != {original.file_path}"
            )
            assert result.start_line == original.start_line, (
                f"start_line mismatch: {result.start_line} != {original.start_line}"
            )
            assert result.end_line == original.end_line, (
                f"end_line mismatch: {result.end_line} != {original.end_line}"
            )
            assert result.content == original.content, "content mismatch"


class TestRerankerLimitCompliance:
    """
    **Feature: codebase-semantic-search, Property 14b: Reranker Limit Compliance**
    **Validates: Requirements 4.4**

    *For any* search with re-ranking enabled, the Reranker's returned results
    count should be <= the requested limit (top_k).
    """

    @given(
        candidates=search_results_list(min_size=1, max_size=30),
        query=st.text(
            min_size=5,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        ),
        top_k=st.integers(min_value=1, max_value=50),
    )
    @settings(
        max_examples=50,
        deadline=10000,
    )
    def test_reranked_results_respect_limit(self, candidates, query, top_k):
        """Reranked results should not exceed the specified top_k limit."""
        assume(query.strip())  # Non-empty query

        reranker = SimpleReranker()

        # Get reranked results
        results = reranker.rerank(query, candidates, top_k)

        # Results count should be <= top_k
        assert len(results) <= top_k, f"Got {len(results)} results, expected <= {top_k}"

    @given(
        candidates=search_results_list(min_size=10, max_size=20),
        query=st.text(
            min_size=5,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        ),
        top_k=st.integers(min_value=1, max_value=5),
    )
    @settings(
        max_examples=30,
        deadline=10000,
    )
    def test_reranked_results_return_exactly_top_k_when_enough_candidates(
        self, candidates, query, top_k
    ):
        """When candidates >= top_k, reranker should return exactly top_k results."""
        assume(query.strip())  # Non-empty query
        assume(len(candidates) >= top_k)  # Enough candidates

        reranker = SimpleReranker()

        # Get reranked results
        results = reranker.rerank(query, candidates, top_k)

        # Should return exactly top_k when we have enough candidates
        assert len(results) == top_k, f"Got {len(results)} results, expected exactly {top_k}"

    @given(
        candidates=search_results_list(min_size=1, max_size=5),
        query=st.text(
            min_size=5,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        ),
        top_k=st.integers(min_value=10, max_value=20),
    )
    @settings(
        max_examples=30,
        deadline=10000,
    )
    def test_reranked_results_return_all_when_fewer_candidates(self, candidates, query, top_k):
        """When candidates < top_k, reranker should return all candidates."""
        assume(query.strip())  # Non-empty query
        assume(len(candidates) < top_k)  # Fewer candidates than requested

        reranker = SimpleReranker()

        # Get reranked results
        results = reranker.rerank(query, candidates, top_k)

        # Should return all candidates when we have fewer than top_k
        assert len(results) == len(candidates), (
            f"Got {len(results)} results, expected {len(candidates)} (all candidates)"
        )

    @given(
        query=st.text(
            min_size=5,
            max_size=100,
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
        ),
        top_k=st.integers(min_value=1, max_value=20),
    )
    @settings(
        max_examples=20,
        deadline=10000,
    )
    def test_reranked_empty_candidates_returns_empty(self, query, top_k):
        """Reranking empty candidates should return empty results."""
        assume(query.strip())  # Non-empty query

        reranker = SimpleReranker()

        # Get reranked results with empty candidates
        results = reranker.rerank(query, [], top_k)

        # Should return empty list
        assert len(results) == 0, f"Got {len(results)} results for empty candidates, expected 0"
