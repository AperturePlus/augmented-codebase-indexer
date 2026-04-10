"""
Graph Builder — constructs code-relationship graphs from AST data.

Hooks into IndexingService as a post-processing step.  For each file
processed, it extracts symbol definitions (from ASTNode) and references
(from ReferenceExtractor), builds fully-qualified names, and writes
nodes / edges / symbol-index entries to the GraphStore.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from aci.core.graph_models import GraphEdge, GraphNode, SymbolIndexEntry, SymbolLocation
from aci.core.parsers.base import ASTNode, SymbolReference

if TYPE_CHECKING:
    from aci.core.ast_parser import TreeSitterParser
    from aci.core.graph_store import GraphStoreInterface
    from aci.core.parsers.reference_extractor import ReferenceExtractorInterface

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds code-relationship graphs from AST nodes and reference extractors.

    Injected into ``IndexingService`` as an optional dependency.  When present,
    ``IndexingService._process_file()`` passes parsed AST nodes to
    ``process_file()`` after chunking completes.
    """

    def __init__(
        self,
        graph_store: GraphStoreInterface,
        ast_parser: TreeSitterParser,
        reference_extractors: dict[str, ReferenceExtractorInterface],
    ) -> None:
        self._graph_store = graph_store
        self._ast_parser = ast_parser
        self._reference_extractors = reference_extractors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_file(
        self,
        file_path: str,
        content: str,
        language: str,
        ast_nodes: list[ASTNode],
    ) -> None:
        """Extract symbols and references from a single file and write to graph store.

        Called by IndexingService after AST parsing.  Uses the existing
        ``ast_nodes`` for definitions and calls the appropriate
        ``ReferenceExtractor`` for references.  Resolves references to FQNs
        where possible using the symbol_index table.
        """
        # 1. Build graph nodes + symbol index entries from definitions
        nodes: list[GraphNode] = []
        symbols: list[SymbolIndexEntry] = []

        for ast_node in ast_nodes:
            fqn = self._build_fqn(ast_node, file_path)
            node = GraphNode(
                symbol_id=fqn,
                symbol_name=ast_node.name,
                symbol_type=ast_node.node_type,
                file_path=file_path,
                start_line=ast_node.start_line,
                end_line=ast_node.end_line,
                language=language,
            )
            nodes.append(node)

            symbol_entry = SymbolIndexEntry(
                fqn=fqn,
                definition=SymbolLocation(
                    file_path=file_path,
                    start_line=ast_node.start_line,
                    end_line=ast_node.end_line,
                ),
                graph_node_id=fqn,
            )
            symbols.append(symbol_entry)

        # Also create a module-level node for the file itself
        module_fqn = self._file_path_to_module(file_path)
        line_count = content.count("\n") + 1
        module_node = GraphNode(
            symbol_id=module_fqn,
            symbol_name=os.path.basename(file_path),
            symbol_type="module",
            file_path=file_path,
            start_line=1,
            end_line=line_count,
            language=language,
        )
        nodes.append(module_node)

        # Batch-upsert nodes and symbols
        self._graph_store.upsert_nodes_batch(nodes)
        self._graph_store.upsert_symbols_batch(symbols)

        # 2. Extract references and build edges
        edges: list[GraphEdge] = []

        extractor = self._reference_extractors.get(language)
        if extractor is not None:
            tree = self._ast_parser.parse_tree(content, language)
            if tree is not None:
                root_node = tree.root_node

                # Call references → call edges
                refs = extractor.extract_references(root_node, content, file_path)
                for ref in refs:
                    edge = self._resolve_reference(ref, file_path, ast_nodes)
                    if edge is not None:
                        edges.append(edge)

                # Import references → import edges (module-level)
                import_refs = extractor.extract_imports(root_node, content, file_path)
                for imp in import_refs:
                    edge = self._resolve_import(imp, file_path, module_fqn)
                    if edge is not None:
                        edges.append(edge)

        if edges:
            self._graph_store.upsert_edges_batch(edges)

    async def remove_file(self, file_path: str) -> None:
        """Remove all graph nodes, edges, and symbols originating from a file."""
        self._graph_store.delete_by_file(file_path)
        self._graph_store.delete_symbols_by_file(file_path)

    async def build_full_graph(self, files: list[tuple[str, str, str, list[ASTNode]]]) -> None:
        """Rebuild the full graph for a list of files.

        Each element is ``(file_path, content, language, ast_nodes)``.
        """
        for file_path, content, language, ast_nodes in files:
            await self.process_file(file_path, content, language, ast_nodes)

    # ------------------------------------------------------------------
    # FQN construction
    # ------------------------------------------------------------------

    def _build_fqn(self, node: ASTNode, file_path: str) -> str:
        """Construct a fully-qualified name from an ASTNode + file path.

        Convention: dot-separated path derived from the file path (converted
        from filesystem separators to dots, with the extension stripped) plus
        the symbol name.  For methods, includes the parent class.

        Example::

            src/aci/services/search_service.py + class SearchService + method search
            → aci.services.search_service.SearchService.search
        """
        module = self._file_path_to_module(file_path)

        if node.node_type == "method" and node.parent_name:
            return f"{module}.{node.parent_name}.{node.name}"
        return f"{module}.{node.name}"

    @staticmethod
    def _file_path_to_module(file_path: str) -> str:
        """Convert a file path to a dot-separated module path.

        Strips a leading ``src/`` prefix if present, replaces path separators
        with dots, and removes the file extension.

        Examples::

            src/aci/services/search_service.py → aci.services.search_service
            lib/utils.py → lib.utils
        """
        # Normalise separators
        path = file_path.replace(os.sep, "/")

        # Strip leading src/ prefix (common convention)
        if path.startswith("src/"):
            path = path[4:]

        # Remove extension
        dot_idx = path.rfind(".")
        if dot_idx != -1:
            path = path[:dot_idx]

        # Replace slashes with dots
        return path.replace("/", ".")

    # ------------------------------------------------------------------
    # Reference resolution helpers
    # ------------------------------------------------------------------

    def _resolve_reference(
        self,
        ref: SymbolReference,
        file_path: str,
        ast_nodes: list[ASTNode],
    ) -> GraphEdge | None:
        """Resolve a symbol reference to a graph edge.

        Attempts to find the target symbol in the graph store's symbol index.
        If not found, records the reference as unresolved.
        """
        # Determine the source FQN (the enclosing symbol that contains this ref)
        source_fqn = ref.parent_symbol
        if source_fqn is None:
            # Fall back to the module-level node
            source_fqn = self._file_path_to_module(file_path)

        # Map ref_type to edge_type
        edge_type = self._ref_type_to_edge_type(ref.ref_type)

        # Try to resolve the target by looking up the symbol index
        target_fqn = self._resolve_target_fqn(ref.name, file_path)

        if target_fqn is not None:
            return GraphEdge(
                source_id=source_fqn,
                target_id=target_fqn,
                edge_type=edge_type,
                file_path=file_path,
                line=ref.line,
            )

        # Record as unresolved symbol in the index
        self._record_unresolved(ref, file_path)
        return None

    def _resolve_import(
        self,
        ref: SymbolReference,
        file_path: str,
        module_fqn: str,
    ) -> GraphEdge | None:
        """Resolve an import reference to a module-level import edge."""
        target_fqn = self._resolve_target_fqn(ref.name, file_path)

        if target_fqn is not None:
            return GraphEdge(
                source_id=module_fqn,
                target_id=target_fqn,
                edge_type="import",
                file_path=file_path,
                line=ref.line,
            )

        # Try treating the import name as a module path directly
        # (e.g. "os.path" → look for a module node)
        module_target = ref.name
        existing = self._graph_store.query_symbol(module_target)
        if existing is not None:
            return GraphEdge(
                source_id=module_fqn,
                target_id=module_target,
                edge_type="import",
                file_path=file_path,
                line=ref.line,
            )

        # Unresolved import — record it
        self._record_unresolved(ref, file_path)
        return None

    def _resolve_target_fqn(self, ref_name: str, file_path: str) -> str | None:
        """Try to resolve a reference name to a fully-qualified symbol.

        Resolution strategy:
        1. Exact FQN match in symbol index.
        2. Short-name match (suffix match) in symbol index.
        3. Module-qualified match: prepend the current file's module path.
        """
        # 1. Exact match
        entry = self._graph_store.lookup_symbol(ref_name)
        if entry is not None:
            return entry.fqn

        # 2. Short-name / suffix match
        candidates = self._graph_store.lookup_symbols_by_name(ref_name)
        if len(candidates) == 1:
            return candidates[0].fqn

        # 3. Module-qualified: try current module prefix + ref_name
        module = self._file_path_to_module(file_path)
        qualified = f"{module}.{ref_name}"
        entry = self._graph_store.lookup_symbol(qualified)
        if entry is not None:
            return entry.fqn

        return None

    def _record_unresolved(self, ref: SymbolReference, file_path: str) -> None:
        """Record an unresolved reference in the symbol index."""
        unresolved_entry = SymbolIndexEntry(
            fqn=ref.name,
            definition=SymbolLocation(
                file_path=file_path,
                start_line=ref.line,
                end_line=ref.line,
            ),
            graph_node_id="",
            unresolved=True,
        )
        # Only upsert if not already present as a resolved symbol
        existing = self._graph_store.lookup_symbol(ref.name)
        if existing is None:
            self._graph_store.upsert_symbol(unresolved_entry)

    @staticmethod
    def _ref_type_to_edge_type(ref_type: str) -> str:
        """Map a SymbolReference.ref_type to a GraphEdge.edge_type."""
        mapping = {
            "call": "call",
            "import": "import",
            "type_annotation": "call",
            "inheritance": "inherits",
        }
        return mapping.get(ref_type, "call")
