"""
Context assembler for structured code intelligence.

Composes rich :class:`ContextPackage` responses from fused query results,
graph neighborhoods, summaries, and (optionally) LLM annotations.

The assembler is the final stage in the query pipeline: the
:class:`QueryRouter` fans out to backends, fuses results via RRF, and
hands the ranked ID list to this assembler for packaging.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aci.core.graph_models import (
    ContextMetadata,
    ContextPackage,
    FileSummary,
    GraphNeighborhood,
    QueryRequest,
    SymbolDetail,
)

if TYPE_CHECKING:
    from aci.core.graph_store import GraphStoreInterface
    from aci.core.tokenizer import TokenizerInterface
    from aci.infrastructure.vector_store import SearchResult
    from aci.services.topology_analyzer import TopologyAnalyzer

logger = logging.getLogger(__name__)

# Maximum time (seconds) allowed for graph enrichment of a single
# search result (Req 9.4).
_ENRICHMENT_TIMEOUT: float = 0.2


class ContextAssembler:
    """Assemble :class:`ContextPackage` from fused query results.

    The assembler resolves result IDs to symbol index entries or chunks,
    fetches source code and summaries, optionally enriches with graph
    neighborhood data, and applies a token budget with PageRank-based
    priority truncation.
    """

    def __init__(
        self,
        graph_store: GraphStoreInterface | None,
        topology_analyzer: TopologyAnalyzer | None,
        tokenizer: TokenizerInterface,
        llm_enricher: Any | None = None,
    ) -> None:
        self._graph_store = graph_store
        self._topology = topology_analyzer
        self._tokenizer = tokenizer
        self._llm_enricher = llm_enricher

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def assemble(
        self,
        fused_results: list[str],
        request: QueryRequest,
    ) -> ContextPackage:
        """Build a :class:`ContextPackage` from fused result IDs.

        Steps:
        1. Resolve each result ID to a SymbolIndexEntry (via graph store).
        2. Fetch source code and summaries.
        3. If ``request.include_graph_context``, fetch graph neighborhood
           up to ``request.depth``.
        4. Apply token budget (``request.max_tokens``) using PageRank-based
           priority for truncation (Req 6.5, 6.6).
        5. Build and return ContextPackage with metadata.
        """
        store = self._graph_store
        depth = min(request.depth, 3)

        symbols: list[SymbolDetail] = []
        file_paths_seen: set[str] = set()
        file_summaries: list[FileSummary] = []
        graph_neighborhood: GraphNeighborhood | None = None

        # Resolve each fused result to a SymbolDetail.
        for result_id in fused_results:
            entry = await asyncio.to_thread(store.lookup_symbol, result_id) if store else None
            if entry is None:
                # Try short-name lookup as fallback.
                if store is not None:
                    entries = await asyncio.to_thread(
                        store.lookup_symbols_by_name, result_id
                    )
                    if entries:
                        entry = entries[0]

            if entry is None:
                # Result doesn't map to a known symbol — include as
                # a minimal detail with the ID as the FQN.
                symbols.append(SymbolDetail(fqn=result_id, source_code="", summary=""))
                continue

            # Fetch source code from disk.
            source_code = self._read_source(
                entry.definition.file_path,
                entry.definition.start_line,
                entry.definition.end_line,
            )

            summary = entry.llm_summary or entry.summary

            # Fetch callers / callees (depth-1 neighbors).
            callers: list[str] = []
            callees: list[str] = []
            pagerank: float = 0.0
            if store is not None:
                caller_nodes = await asyncio.to_thread(
                    store.get_neighbors, entry.fqn, "callers", depth=1
                )
                callers = [n.symbol_id for n in caller_nodes]

                callee_nodes = await asyncio.to_thread(
                    store.get_neighbors, entry.fqn, "callees", depth=1
                )
                callees = [n.symbol_id for n in callee_nodes]

                pagerank = await asyncio.to_thread(
                    store.get_pagerank, entry.fqn
                )

            symbols.append(
                SymbolDetail(
                    fqn=entry.fqn,
                    source_code=source_code,
                    summary=summary,
                    callers=callers,
                    callees=callees,
                    pagerank_score=pagerank,
                )
            )

            # Collect file path for file-level summary.
            fp = entry.definition.file_path
            if fp and fp not in file_paths_seen:
                file_paths_seen.add(fp)
                fs = await self._build_file_summary(fp)
                if fs is not None:
                    file_summaries.append(fs)

        # Graph neighborhood (Req 6.3).
        if request.include_graph_context and store is not None and fused_results:
            graph_neighborhood = await self._build_graph_neighborhood(
                fused_results, depth
            )

        # Token-budget truncation (Req 6.4, 6.5, 6.6).
        symbols = self._apply_token_budget(symbols, request.max_tokens)

        # Build metadata (Req 6.7).
        pr_scores = [s.pagerank_score for s in symbols]
        pr_range = (min(pr_scores), max(pr_scores)) if pr_scores else (0.0, 0.0)
        total_tokens = self._count_tokens_for_symbols(symbols)

        metadata = ContextMetadata(
            query_params={
                "query": request.query,
                "query_type": request.query_type,
                "depth": request.depth,
                "max_tokens": request.max_tokens,
            },
            symbol_count=len(symbols),
            total_tokens=total_tokens,
            pagerank_score_range=pr_range,
        )

        return ContextPackage(
            query=request.query,
            symbols=symbols,
            graph_neighborhood=graph_neighborhood,
            file_summaries=file_summaries,
            metadata=metadata,
        )

    async def enrich_search_results(
        self,
        results: list[SearchResult],
        request: QueryRequest,
    ) -> ContextPackage:
        """Graph-enrich existing search results (Req 9).

        For each result that maps to a known symbol, attaches direct
        callers, callees, and module dependencies.  Graph enrichment is
        bounded to 200 ms per result (Req 9.4).

        When the graph is disabled (``graph_store is None``), returns
        results as-is wrapped in a :class:`ContextPackage`.
        """
        store = self._graph_store
        symbols: list[SymbolDetail] = []
        file_paths_seen: set[str] = set()
        file_summaries: list[FileSummary] = []

        for result in results:
            detail = SymbolDetail(
                fqn=result.chunk_id,
                source_code=result.content,
                summary="",
            )

            if store is not None:
                try:
                    detail = await asyncio.wait_for(
                        self._enrich_single_result(result, store),
                        timeout=_ENRICHMENT_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.debug(
                        "Graph enrichment timed out for %s", result.chunk_id
                    )
                except Exception:
                    logger.debug(
                        "Graph enrichment failed for %s",
                        result.chunk_id,
                        exc_info=True,
                    )

            symbols.append(detail)

            # Collect file-level summaries.
            fp = result.file_path
            if fp and fp not in file_paths_seen and store is not None:
                file_paths_seen.add(fp)
                try:
                    fs = await asyncio.wait_for(
                        self._build_file_summary(fp),
                        timeout=_ENRICHMENT_TIMEOUT,
                    )
                    if fs is not None:
                        file_summaries.append(fs)
                except (asyncio.TimeoutError, Exception):
                    logger.debug(
                        "File summary enrichment failed for %s",
                        fp,
                        exc_info=True,
                    )

        pr_scores = [s.pagerank_score for s in symbols]
        pr_range = (min(pr_scores), max(pr_scores)) if pr_scores else (0.0, 0.0)
        total_tokens = self._count_tokens_for_symbols(symbols)

        metadata = ContextMetadata(
            query_params={
                "query": request.query,
                "query_type": request.query_type,
                "depth": request.depth,
                "max_tokens": request.max_tokens,
                "include_graph_context": request.include_graph_context,
            },
            symbol_count=len(symbols),
            total_tokens=total_tokens,
            pagerank_score_range=pr_range,
        )

        return ContextPackage(
            query=request.query,
            symbols=symbols,
            file_summaries=file_summaries,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _enrich_single_result(
        self,
        result: SearchResult,
        store: GraphStoreInterface,
    ) -> SymbolDetail:
        """Enrich a single search result with graph context."""
        # Try to resolve the chunk to a symbol.
        entry = await asyncio.to_thread(store.lookup_symbol, result.chunk_id)

        fqn = entry.fqn if entry else result.chunk_id
        summary = ""
        if entry:
            summary = entry.llm_summary or entry.summary

        callers: list[str] = []
        callees: list[str] = []
        pagerank: float = 0.0

        if entry:
            caller_nodes = await asyncio.to_thread(
                store.get_neighbors, entry.fqn, "callers", depth=1
            )
            callers = [n.symbol_id for n in caller_nodes]

            callee_nodes = await asyncio.to_thread(
                store.get_neighbors, entry.fqn, "callees", depth=1
            )
            callees = [n.symbol_id for n in callee_nodes]

            pagerank = await asyncio.to_thread(store.get_pagerank, entry.fqn)

        return SymbolDetail(
            fqn=fqn,
            source_code=result.content,
            summary=summary,
            callers=callers,
            callees=callees,
            pagerank_score=pagerank,
        )

    async def _build_file_summary(self, file_path: str) -> FileSummary | None:
        """Build a :class:`FileSummary` for *file_path* from graph data."""
        store = self._graph_store
        if store is None:
            return None

        module_data = await asyncio.to_thread(store.query_module, file_path)
        nodes = module_data.get("nodes", [])
        edges = module_data.get("edges", [])

        symbol_fqns = [n.symbol_id for n in nodes]

        # Imports: edges of type "import" originating from this file.
        imports = [e.target_id for e in edges if e.edge_type == "import"]

        # Dependents: modules that import symbols from this file.
        # We look for edges targeting any symbol defined in this file.
        dependents: list[str] = []
        for node in nodes:
            dep_edges = await asyncio.to_thread(
                store.get_edges, node.symbol_id, "callers", depth=1
            )
            for e in dep_edges:
                if e.edge_type == "import" and e.file_path not in dependents:
                    dependents.append(e.file_path)

        # Build a simple summary from the first node's summary if available.
        summary = ""
        if symbol_fqns:
            first_entry = await asyncio.to_thread(
                store.lookup_symbol, symbol_fqns[0]
            )
            if first_entry:
                summary = first_entry.llm_summary or first_entry.summary

        return FileSummary(
            file_path=file_path,
            summary=summary,
            symbols=symbol_fqns,
            imports=imports,
            dependents=dependents,
        )

    async def _build_graph_neighborhood(
        self,
        result_ids: list[str],
        depth: int,
    ) -> GraphNeighborhood:
        """Build a :class:`GraphNeighborhood` for the given result IDs."""
        store = self._graph_store
        assert store is not None  # caller checks

        all_nodes = []
        all_edges = []
        seen_node_ids: set[str] = set()
        seen_edge_keys: set[tuple[str, str, str]] = set()

        for result_id in result_ids:
            nodes = await asyncio.to_thread(
                store.get_neighbors, result_id, "callees", depth=depth
            )
            for n in nodes:
                if n.symbol_id not in seen_node_ids:
                    seen_node_ids.add(n.symbol_id)
                    all_nodes.append(n)

            caller_nodes = await asyncio.to_thread(
                store.get_neighbors, result_id, "callers", depth=depth
            )
            for n in caller_nodes:
                if n.symbol_id not in seen_node_ids:
                    seen_node_ids.add(n.symbol_id)
                    all_nodes.append(n)

            edges = await asyncio.to_thread(
                store.get_edges, result_id, "callees", depth=depth
            )
            for e in edges:
                key = (e.source_id, e.target_id, e.edge_type)
                if key not in seen_edge_keys:
                    seen_edge_keys.add(key)
                    all_edges.append(e)

            caller_edges = await asyncio.to_thread(
                store.get_edges, result_id, "callers", depth=depth
            )
            for e in caller_edges:
                key = (e.source_id, e.target_id, e.edge_type)
                if key not in seen_edge_keys:
                    seen_edge_keys.add(key)
                    all_edges.append(e)

        return GraphNeighborhood(
            nodes=all_nodes,
            edges=all_edges,
            depth=depth,
        )

    def _apply_token_budget(
        self,
        symbols: list[SymbolDetail],
        max_tokens: int,
    ) -> list[SymbolDetail]:
        """Truncate symbols to fit within *max_tokens*.

        Uses PageRank-based priority: symbols with higher PageRank
        scores are retained first (Req 6.5, 6.6).  Within each symbol,
        graph neighborhood data is truncated before source code, and
        source code before summaries.
        """
        if not symbols:
            return symbols

        # Sort by descending PageRank so high-importance symbols are kept.
        sorted_symbols = sorted(
            symbols, key=lambda s: s.pagerank_score, reverse=True
        )

        budget = max_tokens
        result: list[SymbolDetail] = []

        for sym in sorted_symbols:
            tokens = self._count_symbol_tokens(sym)
            if tokens <= budget:
                result.append(sym)
                budget -= tokens
            else:
                # Try to fit a truncated version.
                truncated = self._truncate_symbol(sym, budget)
                if truncated is not None:
                    result.append(truncated)
                    budget -= self._count_symbol_tokens(truncated)
                break  # No room for more symbols.

        return result

    def _truncate_symbol(
        self,
        sym: SymbolDetail,
        budget: int,
    ) -> SymbolDetail | None:
        """Truncate a single symbol to fit within *budget* tokens.

        Truncation order (Req 6.5):
        1. Drop callers/callees (graph neighborhood).
        2. Truncate source code.
        3. Truncate summary.
        """
        # Start by dropping graph neighborhood.
        candidate = SymbolDetail(
            fqn=sym.fqn,
            source_code=sym.source_code,
            summary=sym.summary,
            callers=[],
            callees=[],
            pagerank_score=sym.pagerank_score,
        )
        tokens = self._count_symbol_tokens(candidate)
        if tokens <= budget:
            return candidate

        # Truncate source code.
        candidate = SymbolDetail(
            fqn=sym.fqn,
            source_code=self._tokenizer.truncate_to_tokens(
                sym.source_code, max(budget - self._tokenizer.count_tokens(sym.summary) - 10, 0)
            ),
            summary=sym.summary,
            callers=[],
            callees=[],
            pagerank_score=sym.pagerank_score,
        )
        tokens = self._count_symbol_tokens(candidate)
        if tokens <= budget:
            return candidate

        # Truncate summary too.
        remaining = max(budget - 10, 0)
        candidate = SymbolDetail(
            fqn=sym.fqn,
            source_code="",
            summary=self._tokenizer.truncate_to_tokens(sym.summary, remaining),
            callers=[],
            callees=[],
            pagerank_score=sym.pagerank_score,
        )
        tokens = self._count_symbol_tokens(candidate)
        if tokens <= budget:
            return candidate

        return None

    def _count_symbol_tokens(self, sym: SymbolDetail) -> int:
        """Count the total tokens for a single :class:`SymbolDetail`."""
        total = 0
        if sym.source_code:
            total += self._tokenizer.count_tokens(sym.source_code)
        if sym.summary:
            total += self._tokenizer.count_tokens(sym.summary)
        # Callers/callees are FQN strings — count their token cost.
        for fqn in sym.callers:
            total += self._tokenizer.count_tokens(fqn)
        for fqn in sym.callees:
            total += self._tokenizer.count_tokens(fqn)
        return total

    def _count_tokens_for_symbols(self, symbols: list[SymbolDetail]) -> int:
        """Count total tokens across all symbols."""
        return sum(self._count_symbol_tokens(s) for s in symbols)

    @staticmethod
    def _read_source(file_path: str, start_line: int, end_line: int) -> str:
        """Read source code lines from disk.

        Returns an empty string if the file cannot be read.
        """
        try:
            path = Path(file_path)
            if not path.is_file():
                return ""
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines(
                keepends=True
            )
            # Lines are 1-based in the symbol index.
            start = max(start_line - 1, 0)
            end = min(end_line, len(lines))
            return "".join(lines[start:end])
        except Exception:
            logger.debug("Failed to read source from %s", file_path, exc_info=True)
            return ""
