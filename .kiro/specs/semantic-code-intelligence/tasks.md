# Implementation Plan: Semantic Code Intelligence

## Overview

This plan implements the semantic code intelligence feature for ACI, adding graph-based code analysis, structured context assembly, unified query routing, LLM enrichment, and a library-mode API. The implementation proceeds bottom-up: data models and configuration first, then infrastructure (graph store), then core services (graph builder, topology, PageRank, RRF, context assembler, query router, LLM enricher), then entrypoints (MCP tools, library API), and finally Docker/deployment updates.

Python is the implementation language. Tests use pytest + hypothesis. Linting with ruff. Type checking with mypy.

## Tasks

- [x] 1. Configuration extensions and data models
  - [x] 1.1 Add GraphConfig, LLMConfig, HttpConfig dataclasses to `src/aci/core/config.py`
    - Add `GraphConfig(enabled=True, storage_path=".aci/graph.db", max_depth=3)`
    - Add `LLMConfig(enabled=False, api_url="", api_key="", model="", batch_size=10, timeout=60.0, confidence_threshold=0.5)`
    - Add `HttpConfig(enabled=False)`
    - Add `graph`, `llm`, `http` fields to `ACIConfig`
    - Update `apply_env_overrides()` with new env var mappings (`ACI_GRAPH_*`, `ACI_LLM_*`, `ACI_HTTP_*`)
    - Update `from_file()` to handle new sections
    - Update `to_dict_safe()` to redact `llm.api_key`
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [ ] 1.2 Write unit tests for configuration extensions
    - Test GraphConfig, LLMConfig, HttpConfig defaults
    - Test env var overrides for all new fields
    - Test `from_file()` with new sections
    - Test `to_dict_safe()` redacts `llm.api_key`
    - Test `load_config()` does not raise when LLM keys are absent
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8_

  - [x] 1.3 Create graph data models in `src/aci/core/graph_models.py`
    - Define `GraphNode`, `GraphEdge`, `SymbolIndexEntry`, `SymbolLocation`
    - Define `ContextPackage`, `ContextMetadata`, `SymbolDetail`, `FileSummary`, `GraphNeighborhood`
    - Define `QueryRequest`, `GraphQueryResult`
    - Define `LLMEnrichRequest`, `LLMEnrichResponse`
    - All dataclasses with full type annotations
    - _Requirements: 1.1, 1.3, 2.1, 4.5, 6.1, 6.7_

  - [x] 1.4 Add `SymbolReference` dataclass to `src/aci/core/parsers/base.py`
    - Add `SymbolReference(name, ref_type, file_path, line, parent_symbol)` dataclass
    - _Requirements: 1.2_

- [x] 2. Graph store interface and SQLite implementation
  - [x] 2.1 Create `GraphStoreInterface` in `src/aci/core/graph_store.py`
    - Define abstract methods: `upsert_node`, `upsert_nodes_batch`, `upsert_edge`, `upsert_edges_batch`, `delete_by_file`, `get_neighbors`, `get_edges`, `get_pagerank`, `store_pagerank_scores`, `query_symbol`, `query_module`, `export_json`, `import_json`, `get_all_edges`, `get_all_nodes`, `close`
    - Define symbol index methods: `upsert_symbol`, `upsert_symbols_batch`, `lookup_symbol`, `lookup_symbols_by_name`, `get_symbols_in_file`, `delete_symbols_by_file`
    - _Requirements: 2.9, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 13.1, 13.2_

  - [x] 2.2 Implement `SQLiteGraphStore` in `src/aci/infrastructure/graph_store/sqlite.py`
    - Create `src/aci/infrastructure/graph_store/__init__.py` with re-exports
    - Implement SQLite schema creation (WAL mode, foreign keys, all tables and indexes per design)
    - Implement all `GraphStoreInterface` methods
    - Implement recursive CTE for depth-limited traversal with `include_inferred` filtering
    - Implement `export_json` with schema_version and ISO-8601 timestamp
    - Implement `import_json` with "merge" and "replace" modes in transactions
    - _Requirements: 2.2, 2.9, 2.10, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 8.3, 13.1, 13.2, 13.3, 13.4_

  - [x] 2.3 Write unit tests for SQLiteGraphStore
    - Test schema creation is idempotent
    - Test node/edge CRUD operations
    - Test `delete_by_file` removes all related data
    - Test `get_neighbors` with depth 1, 2, 3 and direction (callers/callees)
    - Test `include_inferred` filtering on queries
    - Test `get_pagerank` returns 0.0 for unknown symbols
    - Test `query_symbol` returns None for missing symbols
    - Test `query_module` returns empty for missing modules
    - _Requirements: 2.2, 2.8, 2.9, 2.10, 4.1, 4.2, 4.3, 4.4, 4.6_

  - [x] 2.4 Write property test for graph export/import round-trip
    - Generate arbitrary graph states (nodes + edges + pagerank scores)
    - Verify export then import in "replace" mode produces equivalent graph
    - Verify schema_version field is present in exported JSON
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

- [x] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise. Run git add & git commit batchly according to semantic diffs. 

- [x] 4. Reference extractors and AST parser extensions
  - [x] 4.1 Add `parse_tree()` method to `TreeSitterParser` in `src/aci/core/ast_parser.py`
    - Add thin method returning raw `tree_sitter.Tree` for reuse by Graph_Builder and reference extractors
    - _Requirements: 1.1, 1.2_

  - [x] 4.2 Create `ReferenceExtractorInterface` in `src/aci/core/parsers/reference_extractor.py`
    - Define abstract `extract_references(root_node, content, file_path) -> list[SymbolReference]`
    - Define abstract `extract_imports(root_node, content, file_path) -> list[SymbolReference]`
    - _Requirements: 1.2_

  - [x] 4.3 Implement `PythonReferenceExtractor` in `src/aci/core/parsers/python_reference_extractor.py`
    - Extract function/method calls, imports, type annotations, inheritance references from Python AST
    - Set `parent_symbol` to enclosing function/class FQN where applicable
    - _Requirements: 1.1, 1.2_

  - [x] 4.4 Implement `JavaScriptReferenceExtractor` in `src/aci/core/parsers/javascript_reference_extractor.py`
    - Extract calls, imports (ES6 + CommonJS), type annotations (JSDoc/TS), inheritance
    - _Requirements: 1.1, 1.2_

  - [x] 4.5 Implement `GoReferenceExtractor` in `src/aci/core/parsers/go_reference_extractor.py`
    - Extract calls, imports, type embedding/interface references
    - _Requirements: 1.1, 1.2_

  - [x] 4.6 Implement `JavaReferenceExtractor` in `src/aci/core/parsers/java_reference_extractor.py`
    - Extract calls, imports, type annotations, inheritance/implements
    - _Requirements: 1.1, 1.2_

  - [x] 4.7 Implement `CppReferenceExtractor` in `src/aci/core/parsers/cpp_reference_extractor.py`
    - Extract calls, includes, type references, inheritance
    - _Requirements: 1.1, 1.2_

  - [x] 4.8 Write unit tests for reference extractors
    - Test PythonReferenceExtractor with calls, imports, type annotations, inheritance
    - Test JavaScriptReferenceExtractor with ES6 imports, CommonJS requires, calls
    - Test GoReferenceExtractor with imports, calls, type embedding
    - Test JavaReferenceExtractor with imports, calls, inheritance
    - Test CppReferenceExtractor with includes, calls, inheritance
    - Test `parent_symbol` is correctly set for nested references
    - _Requirements: 1.1, 1.2_

- [~] 5. Graph builder and indexing integration
  - [ ] 5.1 Implement `GraphBuilder` in `src/aci/services/graph_builder.py`
    - Implement `process_file()`: extract definitions from AST nodes, extract references via ReferenceExtractor, build FQNs, upsert nodes/edges/symbols to GraphStore
    - Implement `remove_file()`: delete all graph data for a file
    - Implement `build_full_graph()`: process multiple files
    - Implement `_build_fqn()`: construct fully-qualified names from ASTNode + file path
    - Resolve references to FQNs using symbol_index lookups; mark unresolved references
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 2.1, 2.3, 3.1, 3.4_

  - [ ] 5.2 Integrate GraphBuilder into IndexingService
    - Add optional `graph_builder: GraphBuilder | None = None` parameter to `IndexingService.__init__()`
    - Call `graph_builder.process_file()` in `_process_file()` after chunking
    - For parallel processing, run graph building as post-processing in main process (same pattern as summary generation)
    - Call `graph_builder.remove_file()` in `update_incremental()` for deleted/modified files
    - When `config.graph.enabled` is False, skip all graph operations
    - _Requirements: 1.1, 1.4, 2.1, 2.3, 3.1, 3.4, 12.5_

  - [ ] 5.3 Write unit tests for GraphBuilder
    - Test `process_file()` creates correct nodes and edges for a Python file
    - Test `remove_file()` cleans up all related graph data
    - Test `_build_fqn()` produces correct FQNs for functions, methods, classes
    - Test unresolved references are recorded with `unresolved=True`
    - Test incremental update: modify file, verify only affected edges change
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.3_

- [~] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [~] 7. Topology analyzer and PageRank scorer
  - [ ] 7.1 Implement `TopologyAnalyzer` in `src/aci/services/topology_analyzer.py`
    - Implement `transitive_callers(symbol_id, max_depth=3)` using GraphStore CTE queries
    - Implement `transitive_callees(symbol_id, max_depth=3)` using GraphStore CTE queries
    - Implement `detect_cycles()` for circular dependency detection
    - Implement `topological_sort()` for acyclic dependency subgraph
    - _Requirements: 2.4, 2.5, 3.2, 3.3_

  - [ ] 7.2 Implement `PageRankScorer` in `src/aci/services/pagerank_scorer.py`
    - Implement power iteration over adjacency data from GraphStore
    - Configurable damping (0.85), max_iterations (50), tolerance (1e-6)
    - Read all edges of given type, build in-memory adjacency, iterate, store scores back
    - _Requirements: 2.6, 2.7, 2.8, 3.6_

  - [ ] 7.3 Write unit tests for TopologyAnalyzer
    - Test transitive callers/callees with known graph structures
    - Test cycle detection with a graph containing cycles
    - Test topological sort on an acyclic graph
    - Test empty graph returns empty results
    - _Requirements: 2.4, 2.5, 3.2, 3.3_

  - [ ] 7.4 Write unit tests for PageRankScorer
    - Test PageRank on a simple known graph (verify convergence)
    - Test scores are stored in GraphStore after compute
    - Test `get_pagerank()` returns 0.0 for unknown symbols
    - Test computation completes within time budget for moderate graphs
    - _Requirements: 2.6, 2.7, 2.8_

- [~] 8. RRF fuser and query router
  - [ ] 8.1 Implement `RRFFuser` in `src/aci/services/rrf_fuser.py`
    - Implement `fuse(ranked_lists, k=60)` using Reciprocal Rank Fusion formula
    - Single-list passthrough when only one backend returns results
    - _Requirements: 5.3, 5.9_

  - [ ] 8.2 Implement `QueryRouter` in `src/aci/services/query_router.py`
    - Implement `query(request)` with parallel fan-out via `asyncio.gather`
    - Dispatch to SearchService, GraphStore, AST parser based on enabled backends
    - Collect results, fuse via RRFFuser, forward to ContextAssembler
    - Handle individual backend failures with `partial_results` flag
    - 2-second timeout budget with cancellation of slow backends
    - Support `backends` parameter to restrict which backends are invoked
    - Skip graph dispatch when `graph_enabled=False` or `graph_store is None`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

  - [ ] 8.3 Write unit tests for RRFFuser
    - Test fusion of multiple ranked lists produces correct RRF scores
    - Test single-list passthrough
    - Test empty input returns empty output
    - _Requirements: 5.3, 5.9_

  - [ ] 8.4 Write unit tests for QueryRouter
    - Test fan-out dispatches to all enabled backends
    - Test `partial_results` flag when a backend fails
    - Test `backends` parameter restricts dispatch
    - Test graph-disabled mode skips graph backend
    - Test timeout handling cancels slow backends
    - _Requirements: 5.1, 5.2, 5.5, 5.6, 5.7, 5.8_

- [ ] 9. Context assembler
  - [ ] 9.1 Implement `ContextAssembler` in `src/aci/services/context_assembler.py`
    - Implement `assemble(fused_results, request)` to build ContextPackage
    - Resolve result IDs to SymbolIndexEntry or chunks
    - Fetch source code, summaries, graph neighborhood based on depth
    - Apply token budget with PageRank-based priority truncation
    - Build metadata section (query params, symbol count, total tokens, PageRank range)
    - Implement `enrich_search_results(results, request)` for graph-aware search
    - Attach direct callers/callees and module dependencies per result
    - Bound graph enrichment to 200ms per result via `asyncio.wait_for`
    - When graph is disabled, return results as-is wrapped in ContextPackage
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 9.1, 9.2, 9.3, 9.4_

  - [ ] 9.2 Write unit tests for ContextAssembler
    - Test symbol query returns source code, summary, callers, callees, file summary
    - Test file query returns file summary, symbols, imports, dependents
    - Test depth parameter controls graph neighborhood levels
    - Test max_tokens truncation uses PageRank priority
    - Test `enrich_search_results` attaches graph context
    - Test graph-disabled mode returns results without enrichment
    - Test 200ms timeout per result for graph enrichment
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 9.1, 9.2, 9.4_

- [ ] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. LLM enricher
  - [ ] 11.1 Implement `LLMEnricher` in `src/aci/services/llm_enricher.py`
    - Implement constructor with disabled-mode detection (no API calls when disabled)
    - Implement `enrich_symbols()` for LLM-generated summaries with batch processing
    - Implement `infer_edges()` for unresolved reference inference with confidence scoring
    - Implement fallback to template-based SummaryGenerator on error
    - Discard inferred edges below confidence threshold, log at debug level
    - Tag inferred edges with `inferred=True` and confidence score
    - Implement `close()` for httpx client cleanup
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.1, 8.2, 8.4_

  - [ ] 11.2 Write unit tests for LLMEnricher
    - Test disabled mode makes no API calls
    - Test fallback to template summaries on LLM error
    - Test batch processing of symbol enrichment
    - Test inferred edges are tagged with `inferred=True` and confidence
    - Test low-confidence edges are discarded
    - Test `close()` cleans up httpx client
    - _Requirements: 7.1, 7.3, 7.5, 8.1, 8.2, 8.4_

- [ ] 12. Graph-aware search integration
  - [ ] 12.1 Update `SearchService` to support `include_graph_context` parameter
    - Add optional `context_assembler: ContextAssembler | None = None` to `SearchService.__init__()`
    - Add `include_graph_context: bool = False` parameter to `search()` method
    - When True and assembler is available, pass results through `ContextAssembler.enrich_search_results()`
    - When True and assembler is None, silently ignore and return unenriched results
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [ ] 12.2 Write unit tests for graph-aware search
    - Test `include_graph_context=False` returns normal results (no change)
    - Test `include_graph_context=True` with assembler enriches results
    - Test `include_graph_context=True` without assembler returns unenriched results
    - _Requirements: 9.1, 9.3_

- [ ] 13. Service container wiring
  - [ ] 13.1 Update `ServicesContainer` and `create_services()` in `src/aci/services/container.py`
    - Add new fields: `graph_store`, `graph_builder`, `topology_analyzer`, `pagerank_scorer`, `context_assembler`, `query_router`, `llm_enricher`, `rrf_fuser`
    - Conditionally create graph components when `config.graph.enabled`
    - Conditionally create LLM enricher when `config.llm.enabled`
    - Create RRFFuser, ContextAssembler, QueryRouter
    - Wire GraphBuilder into IndexingService
    - Wire ContextAssembler into SearchService
    - Create reference extractors registry helper `_create_reference_extractors()`
    - _Requirements: 12.5, 12.6_

  - [ ] 13.2 Update `MCPContext` and `create_mcp_context()` in `src/aci/mcp/context.py`
    - Add `graph_store`, `query_router`, `context_assembler` fields to MCPContext
    - Wire from ServicesContainer in `create_mcp_context()`
    - Update `cleanup_context()` to close graph_store and llm_enricher
    - _Requirements: 14.1, 14.2, 14.3_

- [ ] 14. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 15. MCP tool exposure
  - [ ] 15.1 Add `get_symbol_context` and `query_graph` tool definitions to `src/aci/mcp/tools.py`
    - Add tool schemas per design (symbol, path, depth, max_tokens, include_graph_context for get_symbol_context; symbol_or_path, path, query_type, depth, include_inferred for query_graph)
    - _Requirements: 14.1, 14.2_

  - [ ] 15.2 Implement MCP handlers for new tools in `src/aci/mcp/handlers.py`
    - Implement `_handle_get_symbol_context`: construct QueryRequest, call QueryRouter.query(), serialize ContextPackage to JSON
    - Implement `_handle_query_graph`: call GraphStore.get_neighbors() + TopologyAnalyzer for depth > 1, serialize GraphQueryResult
    - Return structured error `{"error": "graph feature is disabled", "hint": "set ACI_GRAPH_ENABLED=true"}` when graph is disabled
    - _Requirements: 14.1, 14.2, 14.3_

  - [ ]* 15.3 Write unit tests for MCP graph handlers
    - Test `get_symbol_context` returns valid ContextPackage JSON
    - Test `query_graph` returns valid GraphQueryResult JSON
    - Test graph-disabled returns descriptive error
    - Test missing symbol returns empty result
    - _Requirements: 14.1, 14.2, 14.3_

- [ ] 16. ACI library API
  - [ ] 16.1 Implement `ACI` class in `src/aci/__init__.py`
    - Implement `__init__()` with config loading and background event loop on daemon thread
    - Implement `index(path, **options)` → IndexResult
    - Implement `search(query, **options)` → list[SearchResult]
    - Implement `get_context(symbol_or_path, **options)` → ContextPackage
    - Implement `get_graph(symbol_or_path, **options)` → GraphQueryResult
    - Implement `close()` to shut down event loop and release resources
    - Implement context manager (`__enter__`, `__exit__`)
    - Bridge sync callers to async services via `asyncio.run_coroutine_threadsafe`
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [ ] 16.2 Update `pyproject.toml` for library installability
    - Verify package is installable via `pip install aci` with all runtime dependencies
    - Ensure `__all__` exports include `ACI` class
    - _Requirements: 10.5_

  - [ ]* 16.3 Write unit tests for ACI library API
    - Test `ACI()` initializes without starting a server
    - Test `index()`, `search()`, `get_context()`, `get_graph()` return correct types
    - Test context manager properly closes resources
    - Test sync methods correctly bridge to async event loop
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 17. Docker deployment enhancements
  - [ ] 17.1 Update Dockerfile and Docker configuration
    - Verify all dependencies for graph storage and LLM enrichment are included (sqlite3 is stdlib, httpx already present)
    - Add LLM environment variables to `.env.example` (`ACI_LLM_API_KEY`, `ACI_LLM_API_URL`, `ACI_LLM_MODEL`)
    - Add graph environment variables (`ACI_GRAPH_ENABLED`, `ACI_HTTP_ENABLED`)
    - Ensure `/data` volume persists both `index.db` and `graph.db`
    - Verify container starts with LLM disabled when env vars are absent
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [ ] 18. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
  - Run `uv run ruff check src tests`
  - Run `uv run pytest tests/ -v --tb=short -q --durations=10`
  - Run `uv run mypy src --ignore-missing-imports --no-error-summary`

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- The implementation follows the layering rules from AGENTS.md: core → infrastructure → services → entrypoints
- Graph building runs as post-processing in the main process during parallel indexing (same pattern as existing summary generation) since SQLite connections cannot cross process boundaries
- LLM enricher operates in disabled mode by default — no API calls unless explicitly configured
