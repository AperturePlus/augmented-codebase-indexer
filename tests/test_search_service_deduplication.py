"""Property-based tests for search result deduplication in hybrid search."""

from hypothesis import given, settings
from hypothesis import strategies as st

from aci.infrastructure.vector_store import SearchResult
from aci.services.search_service import _deduplicate_results, _is_near_duplicate


@st.composite
def search_result_strategy(draw, source: str = "vector"):
    """Generate a SearchResult with configurable source."""
    file_path = draw(st.from_regex(r"[a-z_]+\.py", fullmatch=True))
    start_line = draw(st.integers(min_value=1, max_value=100))
    end_line = draw(st.integers(min_value=start_line, max_value=start_line + 50))
    content = draw(st.text(min_size=10, max_size=200))
    score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))

    chunk_id = (
        f"{source}:{file_path}:{start_line}" if source == "grep" else f"chunk_{file_path}_{start_line}"
    )

    return SearchResult(
        chunk_id=chunk_id,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        content=content,
        score=score,
        metadata={"source": source},
    )


class TestDeduplication:
    """
    **Feature: hybrid-search, Property 4: Deduplication removes overlapping grep results**
    **Validates: Requirements 2.4**
    """

    @given(
        vector_results=st.lists(
            search_result_strategy(source="vector"),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_overlapping_grep_results_removed(self, vector_results):
        vector_result = vector_results[0]

        grep_start = (vector_result.start_line + vector_result.end_line) // 2
        grep_end = min(grep_start + 2, vector_result.end_line)

        overlapping_grep = SearchResult(
            chunk_id=f"grep:{vector_result.file_path}:{grep_start}",
            file_path=vector_result.file_path,
            start_line=grep_start,
            end_line=grep_end,
            content="overlapping content",
            score=0.5,
            metadata={"source": "grep"},
        )

        assert _is_near_duplicate(overlapping_grep, vector_results)

        deduplicated = _deduplicate_results([overlapping_grep], vector_results)
        assert len(deduplicated) == 0

    @given(
        vector_results=st.lists(
            search_result_strategy(source="vector"),
            min_size=1,
            max_size=3,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_non_overlapping_grep_results_retained(self, vector_results):
        non_overlapping_grep = SearchResult(
            chunk_id="grep:unique_file.py:1",
            file_path="unique_file.py",
            start_line=1,
            end_line=5,
            content="non-overlapping content",
            score=0.5,
            metadata={"source": "grep"},
        )

        assert not _is_near_duplicate(non_overlapping_grep, vector_results)

        deduplicated = _deduplicate_results([non_overlapping_grep], vector_results)
        assert len(deduplicated) == 1
        assert deduplicated[0] == non_overlapping_grep

    @given(
        vector_result=search_result_strategy(source="vector"),
    )
    @settings(max_examples=100, deadline=None)
    def test_grep_outside_line_range_retained(self, vector_result):
        vector_results = [vector_result]

        grep_start = vector_result.end_line + 10
        grep_end = grep_start + 5

        outside_grep = SearchResult(
            chunk_id=f"grep:{vector_result.file_path}:{grep_start}",
            file_path=vector_result.file_path,
            start_line=grep_start,
            end_line=grep_end,
            content="outside range content",
            score=0.5,
            metadata={"source": "grep"},
        )

        assert not _is_near_duplicate(outside_grep, vector_results)

        deduplicated = _deduplicate_results([outside_grep], vector_results)
        assert len(deduplicated) == 1

    @given(
        vector_results=st.lists(
            search_result_strategy(source="vector"),
            min_size=1,
            max_size=3,
        ),
        num_overlapping=st.integers(min_value=1, max_value=3),
        num_non_overlapping=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100, deadline=None)
    def test_mixed_grep_results_correctly_filtered(
        self, vector_results, num_overlapping, num_non_overlapping
    ):
        vector_result = vector_results[0]

        overlapping_greps = []
        for i in range(num_overlapping):
            grep_start = vector_result.start_line + i
            grep_end = min(grep_start + 1, vector_result.end_line)
            overlapping_greps.append(
                SearchResult(
                    chunk_id=f"grep:{vector_result.file_path}:{grep_start}",
                    file_path=vector_result.file_path,
                    start_line=grep_start,
                    end_line=grep_end,
                    content=f"overlapping {i}",
                    score=0.5,
                    metadata={"source": "grep"},
                )
            )

        non_overlapping_greps = []
        for i in range(num_non_overlapping):
            non_overlapping_greps.append(
                SearchResult(
                    chunk_id=f"grep:other_file_{i}.py:1",
                    file_path=f"other_file_{i}.py",
                    start_line=1,
                    end_line=5,
                    content=f"non-overlapping {i}",
                    score=0.5,
                    metadata={"source": "grep"},
                )
            )

        deduplicated = _deduplicate_results(overlapping_greps + non_overlapping_greps, vector_results)

        assert len(deduplicated) == num_non_overlapping
        for result in deduplicated:
            assert not _is_near_duplicate(result, vector_results)
