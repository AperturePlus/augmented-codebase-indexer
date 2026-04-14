"""
Data models for the code graph and structured context assembly.

All graph-related domain primitives live here in the core layer.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------


@dataclass
class GraphNode:
    """A node in the code graph representing a symbol or module."""

    symbol_id: str  # fully-qualified name
    symbol_name: str  # short name
    symbol_type: str  # "function" | "class" | "method" | "module" | "variable"
    file_path: str
    start_line: int
    end_line: int
    language: str = ""
    pagerank_score: float = 0.0


@dataclass
class GraphEdge:
    """A directed edge in the code graph."""

    source_id: str  # fully-qualified source symbol
    target_id: str  # fully-qualified target symbol
    edge_type: str  # "call" | "import" | "inherits" | "inferred"
    inferred: bool = False
    confidence: float = 1.0  # 0.0–1.0; < 1.0 for inferred edges
    file_path: str = ""  # file where the edge originates
    line: int = 0


# ---------------------------------------------------------------------------
# Symbol index
# ---------------------------------------------------------------------------


@dataclass
class SymbolLocation:
    """A source location for a symbol definition or reference."""

    file_path: str
    start_line: int
    end_line: int


@dataclass
class SymbolIndexEntry:
    """An entry in the cross-file symbol index."""

    fqn: str  # fully-qualified name
    definition: SymbolLocation
    references: list[SymbolLocation] = field(default_factory=list)
    graph_node_id: str = ""
    summary: str = ""
    llm_summary: str = ""
    unresolved: bool = False  # True if definition not found in codebase


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------


@dataclass
class SymbolDetail:
    """Detailed information about a symbol for context packages."""

    fqn: str
    source_code: str
    summary: str
    callers: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)
    pagerank_score: float = 0.0


@dataclass
class FileSummary:
    """Summary of a file for context packages."""

    file_path: str
    summary: str
    symbols: list[str] = field(default_factory=list)  # FQNs defined in the file
    imports: list[str] = field(default_factory=list)  # module paths imported
    dependents: list[str] = field(default_factory=list)  # modules that import this file


@dataclass
class GraphNeighborhood:
    """A subgraph around a queried symbol."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    depth: int = 1


@dataclass
class ContextMetadata:
    """Metadata about a context package response."""

    query_params: dict = field(default_factory=dict)
    symbol_count: int = 0
    total_tokens: int = 0
    pagerank_score_range: tuple[float, float] = (0.0, 0.0)
    partial_results: bool = False
    backends_used: list[str] = field(default_factory=list)


@dataclass
class ContextPackage:
    """A structured context response for a query."""

    query: str
    symbols: list[SymbolDetail] = field(default_factory=list)
    graph_neighborhood: GraphNeighborhood | None = None
    file_summaries: list[FileSummary] = field(default_factory=list)
    metadata: ContextMetadata = field(default_factory=ContextMetadata)


# ---------------------------------------------------------------------------
# Query / response
# ---------------------------------------------------------------------------


@dataclass
class QueryRequest:
    """A unified query request for the query router."""

    query: str
    query_type: str = "text"  # "symbol" | "file" | "text"
    depth: int = 1  # graph neighborhood depth, max 3
    max_tokens: int = 8192
    include_graph_context: bool = False
    backends: list[str] | None = None  # None = all enabled backends
    rrf_k: int = 60


@dataclass
class GraphQueryResult:
    """Result of a direct graph query."""

    symbol: str
    query_type: str  # "callers" | "callees" | "dependencies" | "dependents"
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    depth: int = 1


# ---------------------------------------------------------------------------
# LLM enrichment
# ---------------------------------------------------------------------------


@dataclass
class LLMEnrichRequest:
    """Request payload for LLM enrichment."""

    artifacts: list[dict] = field(default_factory=list)  # {"fqn", "source", "type"}
    task: str = "summarize"  # "summarize" | "infer_edges"


@dataclass
class LLMEnrichResponse:
    """Response from LLM enrichment."""

    results: list[dict] = field(default_factory=list)
    model: str = ""
    tokens_used: int = 0
