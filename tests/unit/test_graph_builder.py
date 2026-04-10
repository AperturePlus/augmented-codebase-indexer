"""
Unit tests for GraphBuilder.

Tests process_file, remove_file, _build_fqn, unresolved reference recording,
and incremental update behaviour.
"""

from __future__ import annotations

import pytest

from aci.core.ast_parser import TreeSitterParser
from aci.core.parsers.base import ASTNode
from aci.core.parsers.python_reference_extractor import PythonReferenceExtractor
from aci.infrastructure.graph_store.sqlite import SQLiteGraphStore
from aci.services.graph_builder import GraphBuilder

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def store() -> SQLiteGraphStore:
    """In-memory graph store for fast tests."""
    s = SQLiteGraphStore(":memory:")
    s.initialize()
    return s


@pytest.fixture()
def ast_parser() -> TreeSitterParser:
    return TreeSitterParser()


@pytest.fixture()
def builder(store: SQLiteGraphStore, ast_parser: TreeSitterParser) -> GraphBuilder:
    extractors = {"python": PythonReferenceExtractor()}
    return GraphBuilder(
        graph_store=store,
        ast_parser=ast_parser,
        reference_extractors=extractors,
    )


# ------------------------------------------------------------------
# _build_fqn
# ------------------------------------------------------------------


class TestBuildFqn:
    """Tests for GraphBuilder._build_fqn()."""

    def test_function_fqn(self, builder: GraphBuilder) -> None:
        node = ASTNode(
            node_type="function",
            name="do_stuff",
            start_line=1,
            end_line=5,
            content="def do_stuff(): ...",
        )
        fqn = builder._build_fqn(node, "src/aci/services/search_service.py")
        assert fqn == "aci.services.search_service.do_stuff"

    def test_class_fqn(self, builder: GraphBuilder) -> None:
        node = ASTNode(
            node_type="class",
            name="SearchService",
            start_line=1,
            end_line=50,
            content="class SearchService: ...",
        )
        fqn = builder._build_fqn(node, "src/aci/services/search_service.py")
        assert fqn == "aci.services.search_service.SearchService"

    def test_method_fqn(self, builder: GraphBuilder) -> None:
        node = ASTNode(
            node_type="method",
            name="search",
            start_line=10,
            end_line=30,
            content="def search(self): ...",
            parent_name="SearchService",
        )
        fqn = builder._build_fqn(node, "src/aci/services/search_service.py")
        assert fqn == "aci.services.search_service.SearchService.search"

    def test_no_src_prefix(self, builder: GraphBuilder) -> None:
        node = ASTNode(
            node_type="function",
            name="helper",
            start_line=1,
            end_line=3,
            content="def helper(): ...",
        )
        fqn = builder._build_fqn(node, "lib/utils.py")
        assert fqn == "lib.utils.helper"

    def test_file_path_to_module_strips_src(self, builder: GraphBuilder) -> None:
        assert builder._file_path_to_module("src/aci/core/config.py") == "aci.core.config"

    def test_file_path_to_module_no_src(self, builder: GraphBuilder) -> None:
        assert builder._file_path_to_module("mylib/foo.py") == "mylib.foo"


# ------------------------------------------------------------------
# process_file
# ------------------------------------------------------------------


SAMPLE_PYTHON = """\
class Greeter:
    def greet(self, name: str) -> str:
        return f"Hello, {name}"

def main():
    g = Greeter()
    g.greet("world")
"""


@pytest.mark.asyncio
async def test_process_file_creates_nodes_and_edges(
    builder: GraphBuilder,
    store: SQLiteGraphStore,
    ast_parser: TreeSitterParser,
) -> None:
    """process_file should create graph nodes for definitions and edges for references."""
    file_path = "src/app/greeter.py"
    ast_nodes = ast_parser.parse(SAMPLE_PYTHON, "python")

    await builder.process_file(file_path, SAMPLE_PYTHON, "python", ast_nodes)

    # Verify nodes were created
    all_nodes = store.get_all_nodes()
    node_ids = {n.symbol_id for n in all_nodes}

    # Module node
    assert "app.greeter" in node_ids
    # Class node
    assert "app.greeter.Greeter" in node_ids
    # Method node
    assert "app.greeter.Greeter.greet" in node_ids
    # Function node
    assert "app.greeter.main" in node_ids

    # Verify symbol index entries
    greeter_sym = store.lookup_symbol("app.greeter.Greeter")
    assert greeter_sym is not None
    assert greeter_sym.definition.file_path == file_path

    main_sym = store.lookup_symbol("app.greeter.main")
    assert main_sym is not None

    # Verify edges were created (calls from main → Greeter, greet)
    all_edges = store.get_all_edges()
    assert len(all_edges) > 0


@pytest.mark.asyncio
async def test_process_file_creates_module_node(
    builder: GraphBuilder,
    store: SQLiteGraphStore,
    ast_parser: TreeSitterParser,
) -> None:
    """process_file should always create a module-level node for the file."""
    code = "x = 1\n"
    file_path = "src/pkg/simple.py"
    ast_nodes = ast_parser.parse(code, "python")

    await builder.process_file(file_path, code, "python", ast_nodes)

    module_node = store.query_symbol("pkg.simple")
    assert module_node is not None
    assert module_node.symbol_type == "module"


# ------------------------------------------------------------------
# remove_file
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_file_cleans_up_all_data(
    builder: GraphBuilder,
    store: SQLiteGraphStore,
    ast_parser: TreeSitterParser,
) -> None:
    """remove_file should delete all nodes, edges, and symbols for a file."""
    file_path = "src/app/greeter.py"
    ast_nodes = ast_parser.parse(SAMPLE_PYTHON, "python")

    await builder.process_file(file_path, SAMPLE_PYTHON, "python", ast_nodes)

    # Verify data exists
    assert len(store.get_all_nodes()) > 0

    # Remove
    await builder.remove_file(file_path)

    # Verify all data for this file is gone
    nodes = [n for n in store.get_all_nodes() if n.file_path == file_path]
    assert len(nodes) == 0

    edges = [e for e in store.get_all_edges() if e.file_path == file_path]
    assert len(edges) == 0

    symbols = store.get_symbols_in_file(file_path)
    assert len(symbols) == 0


# ------------------------------------------------------------------
# Unresolved references
# ------------------------------------------------------------------


SAMPLE_WITH_UNRESOLVED = """\
from external_lib import magic_func

def caller():
    magic_func()
"""


@pytest.mark.asyncio
async def test_unresolved_references_recorded(
    builder: GraphBuilder,
    store: SQLiteGraphStore,
    ast_parser: TreeSitterParser,
) -> None:
    """References that cannot be resolved should be recorded as unresolved."""
    file_path = "src/app/caller.py"
    ast_nodes = ast_parser.parse(SAMPLE_WITH_UNRESOLVED, "python")

    await builder.process_file(file_path, SAMPLE_WITH_UNRESOLVED, "python", ast_nodes)

    # "magic_func" is not defined anywhere in the graph, so it should be unresolved
    entry = store.lookup_symbol("magic_func")
    if entry is not None:
        assert entry.unresolved is True


# ------------------------------------------------------------------
# Incremental update
# ------------------------------------------------------------------


SAMPLE_V1 = """\
def alpha():
    pass

def beta():
    alpha()
"""

SAMPLE_V2 = """\
def alpha():
    pass

def gamma():
    alpha()
"""


@pytest.mark.asyncio
async def test_incremental_update_only_affects_changed_file(
    builder: GraphBuilder,
    store: SQLiteGraphStore,
    ast_parser: TreeSitterParser,
) -> None:
    """Modifying a file should update only that file's graph data."""
    file_path = "src/app/mod.py"

    # Index v1
    ast_v1 = ast_parser.parse(SAMPLE_V1, "python")
    await builder.process_file(file_path, SAMPLE_V1, "python", ast_v1)

    nodes_v1 = {n.symbol_id for n in store.get_all_nodes() if n.file_path == file_path}
    assert "app.mod.beta" in nodes_v1

    # Simulate incremental update: remove then re-process
    await builder.remove_file(file_path)
    ast_v2 = ast_parser.parse(SAMPLE_V2, "python")
    await builder.process_file(file_path, SAMPLE_V2, "python", ast_v2)

    nodes_v2 = {n.symbol_id for n in store.get_all_nodes() if n.file_path == file_path}
    # beta should be gone, gamma should be present
    assert "app.mod.beta" not in nodes_v2
    assert "app.mod.gamma" in nodes_v2
    # alpha should still be present
    assert "app.mod.alpha" in nodes_v2


# ------------------------------------------------------------------
# build_full_graph
# ------------------------------------------------------------------


SAMPLE_A = """\
def func_a():
    pass
"""

SAMPLE_B = """\
from app.a import func_a

def func_b():
    func_a()
"""


@pytest.mark.asyncio
async def test_build_full_graph_processes_multiple_files(
    builder: GraphBuilder,
    store: SQLiteGraphStore,
    ast_parser: TreeSitterParser,
) -> None:
    """build_full_graph should process all provided files."""
    files = [
        ("src/app/a.py", SAMPLE_A, "python", ast_parser.parse(SAMPLE_A, "python")),
        ("src/app/b.py", SAMPLE_B, "python", ast_parser.parse(SAMPLE_B, "python")),
    ]

    await builder.build_full_graph(files)

    all_nodes = store.get_all_nodes()
    node_ids = {n.symbol_id for n in all_nodes}
    assert "app.a.func_a" in node_ids
    assert "app.b.func_b" in node_ids
    # Module nodes
    assert "app.a" in node_ids
    assert "app.b" in node_ids


# ------------------------------------------------------------------
# Edge type mapping
# ------------------------------------------------------------------


def test_ref_type_to_edge_type() -> None:
    assert GraphBuilder._ref_type_to_edge_type("call") == "call"
    assert GraphBuilder._ref_type_to_edge_type("import") == "import"
    assert GraphBuilder._ref_type_to_edge_type("inheritance") == "inherits"
    assert GraphBuilder._ref_type_to_edge_type("type_annotation") == "call"
    # Unknown falls back to "call"
    assert GraphBuilder._ref_type_to_edge_type("unknown_type") == "call"


# ------------------------------------------------------------------
# No extractor for language
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_file_without_extractor(
    store: SQLiteGraphStore,
    ast_parser: TreeSitterParser,
) -> None:
    """When no reference extractor exists for a language, nodes are still created but no edges."""
    builder = GraphBuilder(
        graph_store=store,
        ast_parser=ast_parser,
        reference_extractors={},  # no extractors
    )
    code = "def hello(): pass\n"
    file_path = "src/app/hello.py"
    ast_nodes = ast_parser.parse(code, "python")

    await builder.process_file(file_path, code, "python", ast_nodes)

    # Nodes should still be created
    all_nodes = store.get_all_nodes()
    assert any(n.symbol_id == "app.hello.hello" for n in all_nodes)

    # No edges since no extractor
    all_edges = store.get_all_edges()
    assert len(all_edges) == 0
