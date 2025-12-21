"""
Search Service for Project ACI.

Provides semantic search functionality over indexed codebases.
"""

import asyncio
import inspect
import logging

from aci.infrastructure.embedding import EmbeddingClientInterface
from aci.infrastructure.grep_searcher import GrepSearcherInterface, TextSearchMode
from aci.infrastructure.vector_store import SearchResult, VectorStoreInterface
from aci.services.search_types import RerankerInterface, SearchMode, TextSearchOptions
from aci.services.search_utils import (
    apply_exclusions,
    deduplicate_by_location,
    deduplicate_grep_results,
    normalize_scores,
    parse_query_modifiers,
)

logger = logging.getLogger(__name__)


class SearchService:
    """
    Service for semantic code search.

    Converts queries to embeddings, searches the vector store,
    and optionally re-ranks results. Supports hybrid search combining
    vector and grep search methods.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClientInterface,
        vector_store: VectorStoreInterface,
        reranker: RerankerInterface | None = None,
        grep_searcher: GrepSearcherInterface | None = None,
        default_limit: int = 10,
        recall_multiplier: int = 5,
        vector_candidates: int = 20,
        grep_candidates: int = 20,
    ):
        """
        Initialize the search service.

        Args:
            embedding_client: Client for generating query embeddings
            vector_store: Store for vector search
            reranker: Optional re-ranker for result refinement
            grep_searcher: Optional grep searcher for keyword search
            default_limit: Default number of results to return
            recall_multiplier: Multiplier for initial recall when re-ranking
            vector_candidates: Number of candidates to retrieve from vector search
            grep_candidates: Number of candidates to retrieve from grep search
        """
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._reranker = reranker
        self._grep_searcher = grep_searcher
        self._default_limit = default_limit
        self._recall_multiplier = recall_multiplier
        self._vector_candidates = vector_candidates
        self._grep_candidates = grep_candidates


    async def search(
        self,
        query: str,
        limit: int | None = None,
        file_filter: str | None = None,
        use_rerank: bool = True,
        search_mode: SearchMode = SearchMode.HYBRID,
        collection_name: str | None = None,
        artifact_types: list[str] | None = None,
        text_options: TextSearchOptions | None = None,
    ) -> list[SearchResult]:
        """
        Perform semantic search.

        Supports query modifiers:
        - `path:*.py` or `file:src/**` - include only matching paths
        - `-path:tests` or `exclude:tests` - exclude matching paths

        Args:
            query: Natural language search query (may include modifiers)
            limit: Maximum results to return (default: default_limit)
            file_filter: Optional glob pattern for file paths
            use_rerank: Whether to use re-ranker if available
            search_mode: Search mode (HYBRID, VECTOR, or GREP)
            collection_name: Optional collection to search
            artifact_types: Optional list of artifact types to filter by
                (e.g., ["chunk", "function_summary", "class_summary", "file_summary"]).
                If None, returns all artifact types.

        Returns:
            List of SearchResult sorted by relevance
        """
        text_options = text_options or TextSearchOptions()
        limit = limit or self._default_limit

        # Parse query for modifiers
        clean_query, query_file_filter, exclude_patterns = parse_query_modifiers(query)
        effective_filter = query_file_filter or file_filter
        search_query = clean_query if clean_query else query

        will_rerank = use_rerank and self._reranker is not None

        # Execute searches based on mode
        vector_results, grep_results = await self._dispatch_search(
            search_query,
            effective_filter,
            search_mode,
            will_rerank,
            collection_name,
            artifact_types,
            text_options,
        )

        # Merge and process results
        candidates = self._merge_results(vector_results, grep_results, will_rerank)

        if exclude_patterns:
            candidates = apply_exclusions(candidates, exclude_patterns)

        # Re-rank or sort
        return await self._finalize_results(candidates, search_query, limit, use_rerank)

    async def _dispatch_search(
        self,
        query: str,
        file_filter: str | None,
        search_mode: SearchMode,
        will_rerank: bool,
        collection_name: str | None,
        artifact_types: list[str] | None = None,
        text_options: TextSearchOptions | None = None,
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        """Dispatch search based on mode."""
        text_options = text_options or TextSearchOptions()
        # Handle SUMMARY mode: vector-only with summary artifact types
        if search_mode == SearchMode.SUMMARY:
            summary_types = ["function_summary", "class_summary", "file_summary"]
            results = await self._execute_vector_search(
                query, file_filter, will_rerank, collection_name, summary_types
            )
            return results, []

        if search_mode == SearchMode.FUZZY:
            results = await self._execute_text_search(
                query=query,
                file_filter=file_filter,
                collection_name=collection_name,
                text_mode=TextSearchMode.FUZZY,
                text_options=text_options,
            )
            return [], results

        if search_mode == SearchMode.HYBRID:
            # Skip grep if artifact_types is specified and doesn't contain "chunk"
            # Grep operates on raw file content and cannot filter by artifact type
            if artifact_types is not None and "chunk" not in artifact_types:
                logger.debug(
                    "Skipping grep search: artifact_types filter specified without 'chunk'"
                )
                results = await self._execute_vector_search(
                    query, file_filter, will_rerank, collection_name, artifact_types
                )
                return results, []
            return await self._execute_hybrid_search(
                query, file_filter, will_rerank, collection_name, artifact_types, text_options
            )
        elif search_mode == SearchMode.VECTOR:
            results = await self._execute_vector_search(
                query, file_filter, will_rerank, collection_name, artifact_types
            )
            return results, []
        else:  # GREP
            results = await self._execute_grep_search(
                query=query,
                file_filter=file_filter,
                collection_name=collection_name,
                text_options=text_options,
            )
            return [], results

    def _merge_results(
        self,
        vector_results: list[SearchResult],
        grep_results: list[SearchResult],
        will_rerank: bool,
    ) -> list[SearchResult]:
        """Merge and deduplicate vector and grep results."""
        # Normalize scores for hybrid without reranking
        if vector_results and grep_results and not will_rerank:
            grep_results, vector_results = normalize_scores(grep_results, vector_results)

        if vector_results and grep_results:
            deduplicated_grep = deduplicate_grep_results(grep_results, vector_results)
            candidates = vector_results + deduplicated_grep
        elif vector_results:
            candidates = vector_results
        else:
            candidates = grep_results

        return deduplicate_by_location(candidates)


    async def _finalize_results(
        self,
        candidates: list[SearchResult],
        query: str,
        limit: int,
        use_rerank: bool,
    ) -> list[SearchResult]:
        """Apply reranking or sorting and return final results."""
        if use_rerank and self._reranker and candidates:
            reranked = self._reranker.rerank(query, candidates, limit)
            if inspect.iscoroutine(reranked):
                return await reranked
            return reranked

        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[:limit]

    async def _execute_vector_search(
        self,
        query: str,
        file_filter: str | None,
        use_rerank: bool = False,
        collection_name: str | None = None,
        artifact_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """Execute vector search and return results."""
        try:
            embeddings = await self._embedding_client.embed_batch([query])
            query_vector = embeddings[0]

            if use_rerank and self._reranker:
                fetch_limit = self._vector_candidates * self._recall_multiplier
            else:
                fetch_limit = self._vector_candidates

            return await self._vector_store.search(
                query_vector=query_vector,
                limit=fetch_limit,
                file_filter=file_filter,
                collection_name=collection_name,
                artifact_types=artifact_types,
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def _execute_grep_search(
        self,
        query: str,
        file_filter: str | None,
        collection_name: str | None = None,
        text_options: TextSearchOptions | None = None,
    ) -> list[SearchResult]:
        """Execute grep search and return results."""
        text_options = text_options or TextSearchOptions()
        text_mode = TextSearchMode.REGEX if text_options.regex else TextSearchMode.SUBSTRING
        return await self._execute_text_search(
            query=query,
            file_filter=file_filter,
            collection_name=collection_name,
            text_mode=text_mode,
            text_options=text_options,
        )

    async def _execute_text_search(
        self,
        query: str,
        file_filter: str | None,
        collection_name: str | None,
        text_mode: TextSearchMode,
        text_options: TextSearchOptions,
    ) -> list[SearchResult]:
        if not self._grep_searcher:
            return []

        try:
            file_paths = await self._vector_store.get_all_file_paths(collection_name)

            search_fn = self._grep_searcher.search
            kwargs = {
                "query": query,
                "file_paths": file_paths,
                "limit": self._grep_candidates,
                "context_lines": text_options.context_lines,
                "case_sensitive": text_options.case_sensitive,
                "file_filter": file_filter,
                "mode": text_mode,
                "all_terms": text_options.all_terms,
                "fuzzy_min_score": text_options.fuzzy_min_score,
            }

            try:
                sig = inspect.signature(search_fn)
                has_var_kwargs = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                if not has_var_kwargs:
                    accepted = set(sig.parameters.keys())
                    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
            except (TypeError, ValueError):
                pass

            return await search_fn(**kwargs)
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    async def _execute_hybrid_search(
        self,
        query: str,
        file_filter: str | None,
        use_rerank: bool = False,
        collection_name: str | None = None,
        artifact_types: list[str] | None = None,
        text_options: TextSearchOptions | None = None,
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        """Execute both vector and grep search in parallel."""
        try:
            vector_task = self._execute_vector_search(
                query, file_filter, use_rerank, collection_name, artifact_types
            )
            grep_task = self._execute_grep_search(
                query=query,
                file_filter=file_filter,
                collection_name=collection_name,
                text_options=text_options,
            )

            vector_results, grep_results = await asyncio.gather(
                vector_task, grep_task, return_exceptions=True
            )

            if isinstance(vector_results, Exception):
                logger.error(f"Vector search failed in hybrid mode: {vector_results}")
                vector_results = []
            if isinstance(grep_results, Exception):
                logger.error(f"Grep search failed in hybrid mode: {grep_results}")
                grep_results = []

            return vector_results, grep_results
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return [], []

    async def search_by_file(
        self,
        query: str,
        file_path: str,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search within a specific file."""
        return await self.search(
            query=query,
            limit=limit,
            file_filter=file_path,
            use_rerank=False,
        )

    async def get_similar_chunks(
        self,
        chunk_id: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Find chunks similar to a given chunk."""
        reference = await self._vector_store.get_by_id(chunk_id)
        if not reference:
            return []

        results = await self.search(
            query=reference.content,
            limit=limit + 1,
            use_rerank=False,
        )

        return [r for r in results if r.chunk_id != chunk_id][:limit]
