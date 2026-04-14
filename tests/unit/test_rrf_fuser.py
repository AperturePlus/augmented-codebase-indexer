"""Unit tests for RRFFuser."""

from __future__ import annotations

import pytest

from aci.services.rrf_fuser import RRFFuser


@pytest.fixture
def fuser() -> RRFFuser:
    return RRFFuser()


# ------------------------------------------------------------------
# Empty / trivial inputs
# ------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_list_returns_empty(self, fuser: RRFFuser) -> None:
        assert fuser.fuse([]) == []

    def test_all_empty_sublists_returns_empty(self, fuser: RRFFuser) -> None:
        assert fuser.fuse([[], [], []]) == []

    def test_single_empty_sublist_returns_empty(self, fuser: RRFFuser) -> None:
        assert fuser.fuse([[]]) == []


# ------------------------------------------------------------------
# Single-list passthrough  (Req 5.9)
# ------------------------------------------------------------------


class TestSingleListPassthrough:
    def test_single_list_preserves_order(self, fuser: RRFFuser) -> None:
        result = fuser.fuse([["a", "b", "c"]])
        ids = [item_id for item_id, _ in result]
        assert ids == ["a", "b", "c"]

    def test_single_list_scores_decrease(self, fuser: RRFFuser) -> None:
        result = fuser.fuse([["a", "b", "c"]])
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_single_list_with_empty_siblings(self, fuser: RRFFuser) -> None:
        """Non-empty list among empty ones is treated as single-list."""
        result = fuser.fuse([[], ["x", "y"], []])
        ids = [item_id for item_id, _ in result]
        assert ids == ["x", "y"]

    def test_single_list_score_formula(self, fuser: RRFFuser) -> None:
        """Verify the exact RRF score for single-list passthrough."""
        k = 60
        result = fuser.fuse([["a", "b"]], k=k)
        assert result[0] == ("a", pytest.approx(1.0 / (k + 1)))
        assert result[1] == ("b", pytest.approx(1.0 / (k + 2)))


# ------------------------------------------------------------------
# Multi-list fusion  (Req 5.3)
# ------------------------------------------------------------------


class TestMultiListFusion:
    def test_two_lists_correct_scores(self, fuser: RRFFuser) -> None:
        """Item appearing in both lists gets summed RRF scores."""
        k = 60
        # "a" is rank 1 in list 1, rank 2 in list 2
        # "b" is rank 2 in list 1, rank 1 in list 2
        result = fuser.fuse([["a", "b"], ["b", "a"]], k=k)
        scores = dict(result)

        expected_a = 1.0 / (k + 1) + 1.0 / (k + 2)
        expected_b = 1.0 / (k + 2) + 1.0 / (k + 1)
        assert scores["a"] == pytest.approx(expected_a)
        assert scores["b"] == pytest.approx(expected_b)
        # Both should have the same score (symmetric)
        assert scores["a"] == pytest.approx(scores["b"])

    def test_item_in_one_list_only(self, fuser: RRFFuser) -> None:
        """Item appearing in only one list gets score from that list only."""
        k = 60
        result = fuser.fuse([["a", "b"], ["c", "d"]], k=k)
        scores = dict(result)

        assert scores["a"] == pytest.approx(1.0 / (k + 1))
        assert scores["c"] == pytest.approx(1.0 / (k + 1))
        assert scores["b"] == pytest.approx(1.0 / (k + 2))

    def test_shared_item_ranks_higher(self, fuser: RRFFuser) -> None:
        """An item in multiple lists should rank above items in only one."""
        k = 60
        # "shared" appears in both lists at rank 2
        # "only1" appears in list 1 at rank 1
        result = fuser.fuse([["only1", "shared"], ["only2", "shared"]], k=k)
        scores = dict(result)

        # shared: 1/(k+2) + 1/(k+2) = 2/(k+2)
        # only1:  1/(k+1)
        # 2/(k+2) vs 1/(k+1) → 2/62 ≈ 0.0323 vs 1/61 ≈ 0.0164
        assert scores["shared"] > scores["only1"]

    def test_three_lists_fusion(self, fuser: RRFFuser) -> None:
        """Fusion across three lists."""
        k = 60
        result = fuser.fuse([["a", "b"], ["b", "c"], ["c", "a"]], k=k)
        scores = dict(result)

        # Each item appears in exactly 2 lists
        expected_a = 1.0 / (k + 1) + 1.0 / (k + 2)
        expected_b = 1.0 / (k + 2) + 1.0 / (k + 1)
        expected_c = 1.0 / (k + 2) + 1.0 / (k + 1)
        assert scores["a"] == pytest.approx(expected_a)
        assert scores["b"] == pytest.approx(expected_b)
        assert scores["c"] == pytest.approx(expected_c)

    def test_result_sorted_descending(self, fuser: RRFFuser) -> None:
        result = fuser.fuse([["a", "b", "c"], ["c", "b", "a"]])
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_custom_k_parameter(self, fuser: RRFFuser) -> None:
        """Different k values produce different scores."""
        result_k10 = dict(fuser.fuse([["a", "b"], ["b", "a"]], k=10))
        result_k100 = dict(fuser.fuse([["a", "b"], ["b", "a"]], k=100))

        # With smaller k, scores are larger
        assert result_k10["a"] > result_k100["a"]

    def test_deterministic_tiebreaking(self, fuser: RRFFuser) -> None:
        """Items with equal scores are sorted by id for determinism."""
        result = fuser.fuse([["b", "a"], ["a", "b"]])
        ids = [item_id for item_id, _ in result]
        # Both have the same score; should be sorted alphabetically
        assert ids == ["a", "b"]
