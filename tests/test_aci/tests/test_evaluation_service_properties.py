"""
Property-based tests for EvaluationService.

Tests the correctness properties for evaluation metrics including
Recall@K and MRR calculations.
"""


from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aci.services.evaluation_service import EvaluationService


# Strategies for generating test data
@st.composite
def file_path(draw):
    """Generate a valid file path."""
    return draw(st.from_regex(r"/[a-z]+/[a-z_]+\.py", fullmatch=True))


@st.composite
def file_paths_list(draw, min_size=1, max_size=20):
    """Generate a list of unique file paths."""
    paths = draw(
        st.lists(
            file_path(),
            min_size=min_size,
            max_size=max_size,
            unique=True,
        )
    )
    return paths


class TestRecallAtKCalculation:
    """
    **Feature: codebase-semantic-search, Property 21: Recall@K Calculation Correctness**
    **Validates: Requirements 9.2**

    *For any* evaluation dataset with known relevant results, the calculated
    Recall@K should equal (relevant items in top K) / (total relevant items).
    """

    @given(
        retrieved=file_paths_list(min_size=1, max_size=30),
        relevant=file_paths_list(min_size=1, max_size=10),
        k=st.integers(min_value=1, max_value=50),
    )
    @settings(
        max_examples=100,
        deadline=5000,
    )
    def test_recall_at_k_formula(self, retrieved, relevant, k):
        """Recall@K should follow the formula: (hits in top K) / (total relevant)."""
        # Calculate expected recall manually
        top_k = set(retrieved[:k])
        relevant_set = set(relevant)
        hits = len(top_k & relevant_set)
        expected_recall = hits / len(relevant_set)

        # Calculate using EvaluationService
        actual_recall = EvaluationService.calculate_recall_at_k(retrieved, relevant, k)

        # Should match
        assert abs(actual_recall - expected_recall) < 1e-10, (
            f"Recall mismatch: expected {expected_recall}, got {actual_recall}"
        )

    @given(
        relevant=file_paths_list(min_size=1, max_size=10),
        k=st.integers(min_value=1, max_value=50),
    )
    @settings(
        max_examples=50,
        deadline=5000,
    )
    def test_recall_at_k_perfect_retrieval(self, relevant, k):
        """When all relevant items are in top K, recall should be 1.0."""
        assume(k >= len(relevant))

        # Retrieved list contains all relevant items at the top
        retrieved = relevant.copy()

        recall = EvaluationService.calculate_recall_at_k(retrieved, relevant, k)

        assert recall == 1.0, f"Perfect retrieval should have recall 1.0, got {recall}"

    @given(
        retrieved=file_paths_list(min_size=1, max_size=20),
        relevant=file_paths_list(min_size=1, max_size=10),
        k=st.integers(min_value=1, max_value=50),
    )
    @settings(
        max_examples=50,
        deadline=5000,
    )
    def test_recall_at_k_no_overlap(self, retrieved, relevant, k):
        """When no relevant items are retrieved, recall should be 0.0."""
        # Ensure no overlap by modifying retrieved paths
        modified_retrieved = [f"/other{p}" for p in retrieved]

        recall = EvaluationService.calculate_recall_at_k(modified_retrieved, relevant, k)

        assert recall == 0.0, f"No overlap should have recall 0.0, got {recall}"

    @given(
        k=st.integers(min_value=1, max_value=50),
    )
    @settings(
        max_examples=20,
        deadline=5000,
    )
    def test_recall_at_k_empty_relevant(self, k):
        """When relevant list is empty, recall should be 0.0."""
        retrieved = ["/a/b.py", "/c/d.py"]
        relevant = []

        recall = EvaluationService.calculate_recall_at_k(retrieved, relevant, k)

        assert recall == 0.0, f"Empty relevant should have recall 0.0, got {recall}"

    @given(
        retrieved=file_paths_list(min_size=5, max_size=20),
        relevant=file_paths_list(min_size=1, max_size=5),
    )
    @settings(
        max_examples=50,
        deadline=5000,
    )
    def test_recall_at_k_monotonic_increasing(self, retrieved, relevant):
        """Recall@K should be monotonically non-decreasing as K increases."""
        k_values = [1, 5, 10, 15, 20]
        recalls = [
            EvaluationService.calculate_recall_at_k(retrieved, relevant, k) for k in k_values
        ]

        for i in range(len(recalls) - 1):
            assert recalls[i] <= recalls[i + 1], (
                f"Recall should be monotonic: Recall@{k_values[i]}={recalls[i]} > Recall@{k_values[i + 1]}={recalls[i + 1]}"
            )

    @given(
        retrieved=file_paths_list(min_size=1, max_size=20),
        relevant=file_paths_list(min_size=1, max_size=10),
        k=st.integers(min_value=1, max_value=50),
    )
    @settings(
        max_examples=50,
        deadline=5000,
    )
    def test_recall_at_k_bounded(self, retrieved, relevant, k):
        """Recall@K should always be between 0.0 and 1.0."""
        recall = EvaluationService.calculate_recall_at_k(retrieved, relevant, k)

        assert 0.0 <= recall <= 1.0, f"Recall should be in [0, 1], got {recall}"


class TestMRRCalculation:
    """
    **Feature: codebase-semantic-search, Property 22: MRR Calculation Correctness**
    **Validates: Requirements 9.3**

    *For any* evaluation dataset, the calculated MRR should equal the average
    of (1 / rank of first relevant result) across all queries.
    """

    @given(
        rankings=st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(
        max_examples=100,
        deadline=5000,
    )
    def test_mrr_formula(self, rankings):
        """MRR should follow the formula: (1/N) * sum(1/rank_i)."""
        # Calculate expected MRR manually
        rr_sum = sum(1.0 / r if r > 0 else 0.0 for r in rankings)
        expected_mrr = rr_sum / len(rankings)

        # Calculate using EvaluationService
        actual_mrr = EvaluationService.calculate_mrr(rankings)

        # Should match
        assert abs(actual_mrr - expected_mrr) < 1e-10, (
            f"MRR mismatch: expected {expected_mrr}, got {actual_mrr}"
        )

    @given(
        n=st.integers(min_value=1, max_value=50),
    )
    @settings(
        max_examples=30,
        deadline=5000,
    )
    def test_mrr_all_rank_one(self, n):
        """When all queries have rank 1, MRR should be 1.0."""
        rankings = [1] * n

        mrr = EvaluationService.calculate_mrr(rankings)

        assert mrr == 1.0, f"All rank-1 should have MRR 1.0, got {mrr}"

    @given(
        n=st.integers(min_value=1, max_value=50),
    )
    @settings(
        max_examples=30,
        deadline=5000,
    )
    def test_mrr_all_not_found(self, n):
        """When no relevant results are found (rank 0), MRR should be 0.0."""
        rankings = [0] * n

        mrr = EvaluationService.calculate_mrr(rankings)

        assert mrr == 0.0, f"All not-found should have MRR 0.0, got {mrr}"

    def test_mrr_empty_rankings(self):
        """MRR of empty rankings should be 0.0."""
        mrr = EvaluationService.calculate_mrr([])

        assert mrr == 0.0, f"Empty rankings should have MRR 0.0, got {mrr}"

    @given(
        rankings=st.lists(
            st.integers(min_value=1, max_value=100),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(
        max_examples=50,
        deadline=5000,
    )
    def test_mrr_bounded(self, rankings):
        """MRR should always be between 0.0 and 1.0."""
        mrr = EvaluationService.calculate_mrr(rankings)

        assert 0.0 <= mrr <= 1.0, f"MRR should be in [0, 1], got {mrr}"

    @given(
        rank=st.integers(min_value=1, max_value=100),
    )
    @settings(
        max_examples=30,
        deadline=5000,
    )
    def test_mrr_single_query(self, rank):
        """MRR for single query should be 1/rank."""
        expected = 1.0 / rank

        mrr = EvaluationService.calculate_mrr([rank])

        assert abs(mrr - expected) < 1e-10, (
            f"Single query MRR should be 1/{rank}={expected}, got {mrr}"
        )

    @given(
        rankings=st.lists(
            st.integers(min_value=0, max_value=100),
            min_size=2,
            max_size=20,
        ),
    )
    @settings(
        max_examples=30,
        deadline=5000,
    )
    def test_mrr_order_independent(self, rankings):
        """MRR should be the same regardless of query order."""
        import random

        mrr1 = EvaluationService.calculate_mrr(rankings)

        # Shuffle and recalculate
        shuffled = rankings.copy()
        random.shuffle(shuffled)
        mrr2 = EvaluationService.calculate_mrr(shuffled)

        assert abs(mrr1 - mrr2) < 1e-10, f"MRR should be order-independent: {mrr1} != {mrr2}"
