"""
Evaluation Service for Project ACI.

Provides metrics for evaluating search quality.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from aci.services.search_service import SearchService

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result for a single evaluation query."""

    query: str
    expected_files: List[str]
    retrieved_files: List[str]
    recall_at_k: Dict[int, float]
    reciprocal_rank: float


@dataclass
class EvaluationResult:
    """Overall evaluation results."""

    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    total_queries: int = 0
    per_query_results: List[QueryResult] = field(default_factory=list)


@dataclass
class EvaluationQuery:
    """A single query-answer pair for evaluation."""

    query: str
    relevant_files: List[str]  # File paths that should be returned


class EvaluationService:
    """
    Service for evaluating search quality.

    Computes Recall@K and MRR metrics against a ground truth dataset.
    """

    def __init__(self, search_service: SearchService):
        """
        Initialize the evaluation service.

        Args:
            search_service: Search service to evaluate
        """
        self._search_service = search_service

    def load_dataset(self, dataset_path: Path) -> List[EvaluationQuery]:
        """
        Load evaluation dataset from JSON file.

        Expected format:
        [
            {"query": "...", "relevant_files": ["path1", "path2"]},
            ...
        ]

        Args:
            dataset_path: Path to JSON dataset file

        Returns:
            List of EvaluationQuery objects
        """
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [
            EvaluationQuery(
                query=item["query"],
                relevant_files=item["relevant_files"],
            )
            for item in data
        ]

    async def evaluate(
        self,
        dataset_path: Path,
        k_values: Optional[List[int]] = None,
        max_results: int = 20,
    ) -> EvaluationResult:
        """
        Evaluate search quality against a dataset.

        Args:
            dataset_path: Path to evaluation dataset
            k_values: K values for Recall@K (default: [5, 10, 20])
            max_results: Maximum results to retrieve per query

        Returns:
            EvaluationResult with metrics
        """
        k_values = k_values or [5, 10, 20]
        queries = self.load_dataset(dataset_path)

        result = EvaluationResult(total_queries=len(queries))

        # Initialize recall accumulators
        recall_sums = {k: 0.0 for k in k_values}
        mrr_sum = 0.0

        for eval_query in queries:
            # Run search
            search_results = await self._search_service.search(
                query=eval_query.query,
                limit=max_results,
            )

            retrieved_files = [r.file_path for r in search_results]
            relevant_set = set(eval_query.relevant_files)

            # Calculate Recall@K for each K
            query_recall = {}
            for k in k_values:
                top_k = set(retrieved_files[:k])
                hits = len(top_k & relevant_set)
                recall = hits / len(relevant_set) if relevant_set else 0.0
                query_recall[k] = recall
                recall_sums[k] += recall

            # Calculate Reciprocal Rank
            rr = self._calculate_reciprocal_rank(retrieved_files, relevant_set)
            mrr_sum += rr

            # Store per-query result
            result.per_query_results.append(
                QueryResult(
                    query=eval_query.query,
                    expected_files=eval_query.relevant_files,
                    retrieved_files=retrieved_files,
                    recall_at_k=query_recall,
                    reciprocal_rank=rr,
                )
            )

        # Calculate averages
        n = len(queries)
        if n > 0:
            result.recall_at_k = {k: recall_sums[k] / n for k in k_values}
            result.mrr = mrr_sum / n

        # Check recall threshold and flag for investigation (Req 9.5)
        recall_10 = result.recall_at_k.get(10, 0.0)
        if recall_10 < 0.7:
            logger.warning(
                f"Recall@10 ({recall_10:.3f}) is below threshold (0.7). Investigation recommended."
            )

        return result

    def _calculate_reciprocal_rank(
        self,
        retrieved: List[str],
        relevant: set,
    ) -> float:
        """
        Calculate Reciprocal Rank.

        RR = 1 / rank of first relevant result (0 if none found)

        Args:
            retrieved: List of retrieved file paths
            relevant: Set of relevant file paths

        Returns:
            Reciprocal rank value
        """
        for i, file_path in enumerate(retrieved, 1):
            if file_path in relevant:
                return 1.0 / i
        return 0.0

    @staticmethod
    def calculate_recall_at_k(
        retrieved: List[str],
        relevant: List[str],
        k: int,
    ) -> float:
        """
        Calculate Recall@K.

        Recall@K = (relevant items in top K) / (total relevant items)

        Args:
            retrieved: List of retrieved items
            relevant: List of relevant items
            k: Number of top results to consider

        Returns:
            Recall@K value
        """
        if not relevant:
            return 0.0

        top_k = set(retrieved[:k])
        relevant_set = set(relevant)
        hits = len(top_k & relevant_set)

        return hits / len(relevant_set)

    @staticmethod
    def calculate_mrr(rankings: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank.

        MRR = (1/N) * sum(1/rank_i)

        Args:
            rankings: List of ranks of first relevant result per query
                     (0 means not found)

        Returns:
            MRR value
        """
        if not rankings:
            return 0.0

        rr_sum = sum(1.0 / r if r > 0 else 0.0 for r in rankings)
        return rr_sum / len(rankings)
