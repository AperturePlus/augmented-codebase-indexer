# Requirements Document — Semantic Code Intelligence

## Introduction

ACI (Augmented Codebase Indexer) currently provides AST-based code parsing, chunking, embedding, and semantic search. This feature evolves ACI into a next-generation semantic code intelligence platform by adding four capabilities:

1. **Structured context for callers** — well-organized, rich context that goes beyond raw code chunks, giving callers (LLMs, IDEs, humans) a coherent picture of symbols, relationships, and intent.
2. **Graph & topology analysis** — call graphs, dependency graphs, and data-flow analysis that go beyond what flat AST parsing and RAG can offer.
3. **Multiple deployment options** — Docker image (already exists), `pip install` from PyPI, and a library-mode API so ACI can be embedded in other tools.
4. **LLM integration for context building** — using LLMs to enrich, summarize, and reason about code context during indexing and query time.

## Glossary

- **Graph_Store**: The lightweight, embedded subsystem that persists code-relationship graphs (call graphs, dependency graphs, import graphs) and exposes traversal/query APIs. The Graph_Store uses an in-process storage backend (e.g., SQLite-backed adjacency lists or an in-memory graph with file-based persistence) and does not depend on any external database process.
- **Graph_Builder**: The component that constructs code-relationship graphs from AST nodes and cross-file resolution during indexing.
- **Context_Assembler**: The service that composes structured context packages from chunks, summaries, graph data, and LLM enrichments for caller consumption.
- **LLM_Enricher**: The component that calls an LLM to generate semantic annotations, explanations, or relationship inferences for code artifacts.
- **Topology_Analyzer**: The component that computes graph-level metrics and traversals (e.g., transitive callers, dependency depth, strongly connected components, PageRank centrality scores).
- **ACI_Library**: The public Python API surface that allows programmatic use of ACI without CLI, HTTP, or MCP.
- **Caller**: Any consumer of ACI context — an LLM agent, an IDE plugin, a human via CLI, or another service via MCP.
- **Symbol_Index**: An in-memory or persisted mapping from fully-qualified symbol names to their definitions, usages, and graph node IDs.
- **Context_Package**: A structured response object containing code, summaries, graph neighborhood, and LLM annotations for a queried symbol or region.
- **PageRank_Scorer**: The component within the Topology_Analyzer that computes PageRank scores over the call graph and dependency graph, assigning each symbol and module an importance score based on graph centrality. Higher scores indicate more connected, structurally central code elements.
- **RRF_Fuser**: The component within the Query_Router that merges ranked result lists from multiple analysis backends into a single unified ranking using Reciprocal Rank Fusion (RRF). RRF combines ranks without requiring score normalization across heterogeneous backends.
- **Query_Router**: The unified entry-point component that accepts a caller query, dispatches it to the appropriate analysis backends (Graph_Store, Topology_Analyzer, AST parser, grep searcher, vector search), collects their results, fuses the ranked lists via the RRF_Fuser, and forwards the unified ranking to the Context_Assembler for final packaging.

## Requirements

### Requirement 1: Symbol Index Construction

**User Story:** As a developer, I want ACI to build a cross-file symbol index during indexing, so that symbol definitions and references can be resolved across the entire codebase.

#### Acceptance Criteria

1. WHEN a codebase is indexed, THE Graph_Builder SHALL extract symbol definitions (functions, classes, methods, module-level variables) and their fully-qualified names from AST nodes for each supported language.
2. WHEN a codebase is indexed, THE Graph_Builder SHALL extract symbol references (calls, imports, type annotations) from AST nodes for each supported language.
3. WHEN symbol extraction completes for all files, THE Symbol_Index SHALL map each fully-qualified symbol name to its definition location (file path, start line, end line) and list of reference locations.
4. WHEN a file is updated during incremental indexing, THE Symbol_Index SHALL update only the entries affected by the changed file within 2 seconds for files under 5000 lines.
5. IF a symbol reference cannot be resolved to a definition in the indexed codebase, THEN THE Symbol_Index SHALL record the reference as "unresolved" with the raw reference text preserved.

### Requirement 2: Call Graph Construction

**User Story:** As a developer, I want ACI to build call graphs from the symbol index, so that I can understand caller/callee relationships across the codebase.

#### Acceptance Criteria

1. WHEN symbol indexing completes, THE Graph_Builder SHALL construct a directed call graph where each node represents a function or method and each edge represents a call relationship.
2. THE Graph_Store SHALL persist the call graph so that it survives process restarts without requiring a full re-index.
3. WHEN a file is updated during incremental indexing, THE Graph_Builder SHALL update only the call graph edges originating from symbols defined in the changed file.
4. THE Topology_Analyzer SHALL compute transitive callers (all functions that directly or transitively call a given function) for a queried symbol within 500ms for codebases up to 100,000 symbols.
5. THE Topology_Analyzer SHALL compute transitive callees (all functions that a given function directly or transitively calls) for a queried symbol within 500ms for codebases up to 100,000 symbols.
6. WHEN graph construction completes or a graph update occurs, THE PageRank_Scorer SHALL compute PageRank scores for all nodes in the call graph and store the scores in the Graph_Store alongside the graph data.
7. THE PageRank_Scorer SHALL complete the PageRank computation within 5 seconds for codebases up to 100,000 symbols.
8. THE Graph_Store SHALL expose a method to retrieve the PageRank score for a given symbol, returning 0.0 for symbols not present in the graph.
9. THE Graph_Store SHALL use a lightweight, embedded storage backend that runs in-process and does not require any external database server (e.g., Neo4j, JanusGraph, or any separate database process).
10. THE Graph_Store SHALL store graph data in a single file or a small set of files within the project's `.aci` directory, keeping the storage footprint under 50 MB for codebases up to 100,000 symbols.

### Requirement 3: Dependency Graph Construction

**User Story:** As a developer, I want ACI to build module-level and package-level dependency graphs, so that I can understand how modules depend on each other.

#### Acceptance Criteria

1. WHEN a codebase is indexed, THE Graph_Builder SHALL construct a directed dependency graph where each node represents a module (file) and each edge represents an import relationship.
2. THE Topology_Analyzer SHALL detect circular dependency cycles in the dependency graph and report each cycle as an ordered list of module paths.
3. THE Topology_Analyzer SHALL compute the topological sort order of the dependency graph for acyclic subgraphs.
4. WHEN a file is updated during incremental indexing, THE Graph_Builder SHALL update only the dependency graph edges originating from the changed file.
5. THE Graph_Store SHALL persist the dependency graph alongside the call graph using the same embedded storage backend, with no additional external process dependencies.
6. WHEN dependency graph construction completes or a graph update occurs, THE PageRank_Scorer SHALL compute PageRank scores for all nodes in the dependency graph and store the scores in the Graph_Store.

### Requirement 4: Graph Query API

**User Story:** As a caller, I want to query the code graph through a well-defined API, so that I can retrieve relationship data for any symbol or module.

#### Acceptance Criteria

1. THE Graph_Store SHALL expose a query method that accepts a symbol name and returns its direct callers and callees.
2. THE Graph_Store SHALL expose a query method that accepts a module path and returns its direct imports and reverse-imports (modules that import the queried module).
3. THE Graph_Store SHALL expose a traversal method that accepts a symbol name, a direction (callers or callees), and a maximum depth, and returns the subgraph within that depth.
4. WHEN a query references a symbol that does not exist in the graph, THE Graph_Store SHALL return an empty result set with no error.
5. THE Graph_Store SHALL support serialization of query results to JSON format.
6. THE Graph_Store SHALL execute all graph queries in-process without network calls to external database services.


### Requirement 5: Unified Query Router

**User Story:** As a developer, I want a single query entry point that routes my query to all relevant analysis backends (graph analysis, AST parsing, grep search, vector search) and returns a unified context package built by the Context_Assembler, so that I do not need to invoke each backend individually.

#### Acceptance Criteria

1. THE Query_Router SHALL expose a single `query(request)` method that accepts a query string, an optional query type hint (symbol, file, text), and optional parameters (depth, max_tokens, include_graph_context).
2. WHEN a query is received, THE Query_Router SHALL dispatch the query in parallel to all enabled analysis backends: the SearchService (vector and grep search), the Graph_Store (graph traversal), and the AST parser (structural lookup).
3. WHEN all backend responses are collected, THE RRF_Fuser SHALL merge the ranked result lists from each backend into a single unified ranking using Reciprocal Rank Fusion with a configurable k parameter (default 60).
4. WHEN the RRF_Fuser produces the unified ranking, THE Query_Router SHALL forward the fused result set to the Context_Assembler, which SHALL return a single Context_Package to the caller.
5. IF an individual analysis backend fails or times out, THEN THE Query_Router SHALL proceed with the results from the remaining backends and include a `partial_results` flag in the Context_Package metadata.
6. THE Query_Router SHALL complete the full fan-out, RRF fusion, and context assembly cycle within 2 seconds for codebases up to 100,000 symbols.
7. WHEN the graph feature is disabled, THE Query_Router SHALL skip the graph analysis dispatch and route only to the SearchService and AST parser backends.
8. THE Query_Router SHALL accept a `backends` parameter that allows the caller to restrict which analysis backends are invoked (e.g., only graph, only grep, or any combination).
9. WHEN only a single backend is invoked, THE RRF_Fuser SHALL pass through the backend's ranking unchanged.

### Requirement 6: Structured Context Assembly

**User Story:** As a caller (LLM agent, IDE, human), I want ACI to return well-organized context packages instead of raw chunks, so that I get a coherent understanding of the queried code.

#### Acceptance Criteria

1. WHEN a caller queries a symbol, THE Context_Assembler SHALL return a Context_Package containing: the symbol's source code, its docstring/summary, its direct callers and callees, and the file-level summary of its containing module.
2. WHEN a caller queries a file, THE Context_Assembler SHALL return a Context_Package containing: the file summary, the list of symbols defined in the file, the file's import dependencies, and the modules that depend on the file.
3. THE Context_Assembler SHALL accept a `depth` parameter (default 1, max 3) that controls how many levels of graph neighborhood to include in the Context_Package.
4. THE Context_Assembler SHALL accept a `max_tokens` parameter that limits the total token count of the returned Context_Package.
5. WHEN the `max_tokens` limit requires truncation, THE Context_Assembler SHALL use PageRank scores from the Graph_Store to prioritize retention of higher-PageRank symbols, truncating lower-PageRank content first within each priority tier (graph neighborhood before source code, source code before summaries).
6. WHEN two symbols fall in the same priority tier during truncation, THE Context_Assembler SHALL retain the symbol with the higher PageRank score.
7. THE Context_Package SHALL include a `metadata` section with the query parameters, the number of symbols included, the total token count of the response, and the PageRank score range of included symbols.

### Requirement 7: LLM-Enriched Summaries

**User Story:** As a developer, I want ACI to use an LLM to generate richer semantic summaries during indexing, so that search and context quality improve beyond template-based summaries.

#### Acceptance Criteria

1. WHERE LLM enrichment is enabled, THE LLM_Enricher SHALL generate a natural-language summary for each function and class that describes its purpose, parameters, return value, and side effects.
2. WHERE LLM enrichment is enabled, THE LLM_Enricher SHALL generate a file-level summary that describes the module's responsibility, its key exports, and its role in the broader codebase.
3. THE LLM_Enricher SHALL use a configurable LLM endpoint (API URL, API key, model name) following the same OpenAI-compatible pattern as the existing embedding client.
4. THE LLM_Enricher SHALL batch requests to the LLM to minimize API calls, processing up to a configurable number of artifacts per request.
5. IF the LLM endpoint is unavailable or returns an error, THEN THE LLM_Enricher SHALL fall back to the existing template-based SummaryGenerator and log a warning with the error details.
6. WHERE LLM enrichment is disabled (default), THE Context_Assembler SHALL use the existing template-based summaries with no behavioral change.

### Requirement 8: LLM-Powered Relationship Inference

**User Story:** As a developer, I want ACI to use an LLM to infer semantic relationships that static analysis cannot detect (e.g., duck typing, dynamic dispatch, convention-based coupling), so that the graph is more complete.

#### Acceptance Criteria

1. WHERE LLM enrichment is enabled, THE LLM_Enricher SHALL analyze unresolved symbol references and infer probable target definitions using code context and naming conventions.
2. THE LLM_Enricher SHALL tag LLM-inferred edges in the graph with a confidence score (0.0 to 1.0) and an `inferred: true` flag to distinguish them from statically-resolved edges.
3. THE Graph_Store SHALL support filtering query results to include or exclude LLM-inferred edges.
4. IF the LLM inference confidence is below a configurable threshold (default 0.5), THEN THE LLM_Enricher SHALL discard the inferred edge and log the low-confidence result at debug level.

### Requirement 9: Graph-Aware Search

**User Story:** As a developer, I want search results to be enriched with graph context, so that I understand not just the matching code but its relationships.

#### Acceptance Criteria

1. WHEN a search query returns results, THE Context_Assembler SHALL attach to each result the direct callers and callees of the matched symbol (if the result maps to a known symbol).
2. WHEN a search query returns results, THE Context_Assembler SHALL attach to each result the module-level dependencies of the file containing the match.
3. THE SearchService SHALL accept an optional `include_graph_context` parameter (default false) that controls whether graph enrichment is applied to search results.
4. WHEN `include_graph_context` is true, THE SearchService SHALL complete the graph enrichment step within 200ms per result for codebases up to 100,000 symbols.

### Requirement 10: Library-Mode API (pip install)

**User Story:** As a developer, I want to `pip install aci` and use ACI as a Python library in my own scripts, so that I can programmatically index and query codebases without running a server.

#### Acceptance Criteria

1. THE ACI_Library SHALL expose a public Python API with at minimum: `index(path)`, `search(query, **options)`, `get_context(symbol_or_path, **options)`, and `get_graph(symbol_or_path, **options)`.
2. THE ACI_Library SHALL be importable as `from aci import ACI` and usable without starting any server process.
3. THE ACI_Library SHALL accept configuration via constructor parameters, environment variables, or a config file path, consistent with the existing `ACIConfig` system.
4. THE ACI_Library SHALL manage its own async event loop internally so that callers can use synchronous method calls.
5. THE pyproject.toml SHALL declare the package as installable via `pip install aci` with all required runtime dependencies.

### Requirement 11: Docker Deployment Enhancements

**User Story:** As a DevOps engineer, I want the Docker image to support the new graph and LLM features, so that containerized deployments get the full semantic intelligence capabilities.

#### Acceptance Criteria

1. THE Dockerfile SHALL include all dependencies required for graph storage and LLM enrichment, with no external graph database service required at runtime.
2. THE Docker image SHALL accept LLM configuration via environment variables (`ACI_LLM_API_KEY`, `ACI_LLM_API_URL`, `ACI_LLM_MODEL`).
3. THE Docker image SHALL persist graph data in the `/data` volume alongside the existing metadata database.
4. WHEN LLM environment variables are not set, THE Docker container SHALL start and operate with LLM enrichment disabled, using template-based summaries only.

### Requirement 12: Configuration for New Features

**User Story:** As a developer, I want to configure graph analysis, LLM enrichment, and the HTTP server through the existing configuration system, so that I can enable/disable features and tune parameters.

#### Acceptance Criteria

1. THE ACIConfig SHALL include a `graph` section with fields: `enabled` (bool, default true), `storage_path` (str, default `.aci/graph.db`, pointing to the embedded database file), and `max_depth` (int, default 3).
2. THE ACIConfig SHALL include an `llm` section with fields: `enabled` (bool, default false), `api_url` (str), `api_key` (str), `model` (str), `batch_size` (int, default 10), `timeout` (float, default 60.0), and `confidence_threshold` (float, default 0.5).
3. THE ACIConfig SHALL include an `http` section with field: `enabled` (bool, default false).
4. THE ACIConfig SHALL support environment variable overrides for all new fields following the existing `ACI_<SECTION>_<KEY>` pattern (e.g., `ACI_LLM_API_KEY`, `ACI_GRAPH_ENABLED`, `ACI_HTTP_ENABLED`).
5. WHEN `graph.enabled` is false, THE IndexingService SHALL skip graph construction and the Graph_Store SHALL return empty results for all queries.
6. WHEN `llm.enabled` is false, THE LLM_Enricher SHALL not make any LLM API calls and the system SHALL use template-based summaries.
7. WHEN `http.enabled` is false (the default), THE ACI system SHALL not start the HTTP server process.
8. WHEN `http.enabled` is true, THE ACI system SHALL start the HTTP server with its existing functionality preserved.

### Requirement 13: Graph Data Serialization

**User Story:** As a developer, I want to export and import graph data, so that I can share code intelligence across environments or debug graph contents.

#### Acceptance Criteria

1. THE Graph_Store SHALL support exporting the full graph (nodes and edges) to a JSON file.
2. THE Graph_Store SHALL support importing a graph from a JSON file, merging or replacing the existing graph based on a caller-specified mode ("merge" or "replace").
3. FOR ALL valid graph states, exporting then importing in "replace" mode SHALL produce an equivalent graph (round-trip property).
4. THE JSON export format SHALL include a schema version field to support future format evolution.

### Requirement 14: MCP Exposure of New Capabilities

**User Story:** As a caller using MCP, I want to access graph queries and structured context through the MCP server interface, so that LLM agents can leverage the new features.

> **Note:** The HTTP server is soft-disabled by default (see Requirement 12, `http.enabled`). The existing HTTP server code is retained but does not start unless explicitly enabled. New capabilities (graph queries, structured context) are exposed only through MCP. The HTTP server, when enabled, continues to serve its pre-existing endpoints only.

#### Acceptance Criteria

1. THE MCP server SHALL expose a `get_symbol_context` tool that accepts a symbol name and returns a Context_Package.
2. THE MCP server SHALL expose a `query_graph` tool that accepts a symbol or module path, a query type (callers, callees, dependencies, dependents), and an optional depth, and returns the graph query result.
3. WHEN the graph feature is disabled, THE MCP server SHALL return a descriptive error indicating the feature is not enabled.
