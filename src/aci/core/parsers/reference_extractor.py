"""
Abstract interface for language-specific reference extractors.

Reference extractors complement the existing LanguageParser hierarchy by
extracting symbol *references* (calls, imports, type annotations, inheritance)
from a parsed tree-sitter tree, whereas LanguageParser extracts symbol
*definitions*.  The Graph_Builder uses both to construct call graphs and
dependency graphs.
"""

from abc import ABC, abstractmethod
from typing import Any

from aci.core.parsers.base import SymbolReference


class ReferenceExtractorInterface(ABC):
    """Abstract base class for language-specific reference extractors.

    Each supported language provides a concrete subclass that knows how to
    walk a tree-sitter parse tree and emit :class:`SymbolReference` objects
    for every call, import, type annotation, or inheritance relationship
    found in the source.
    """

    @abstractmethod
    def extract_references(
        self, root_node: Any, content: str, file_path: str
    ) -> list[SymbolReference]:
        """Extract symbol references (calls, type annotations, inheritance) from the AST.

        Args:
            root_node: Tree-sitter root node of the parsed file.
            content: Original source code of the file.
            file_path: Path to the source file (used in ``SymbolReference.file_path``).

        Returns:
            List of ``SymbolReference`` objects found in the source.
        """
        ...

    @abstractmethod
    def extract_imports(
        self, root_node: Any, content: str, file_path: str
    ) -> list[SymbolReference]:
        """Extract import references from the AST.

        Args:
            root_node: Tree-sitter root node of the parsed file.
            content: Original source code of the file.
            file_path: Path to the source file (used in ``SymbolReference.file_path``).

        Returns:
            List of ``SymbolReference`` objects with ``ref_type="import"``.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete utility methods (shared across all extractors)
    # ------------------------------------------------------------------

    def get_node_content(self, node: Any, content: str) -> str:
        """Extract the source code text covered by *node*."""
        return content[node.start_byte : node.end_byte]

    def get_line_numbers(self, node: Any) -> tuple[int, int]:
        """Return 1-based ``(start_line, end_line)`` for *node*."""
        start_line: int = node.start_point[0] + 1
        end_line: int = node.end_point[0] + 1
        return start_line, end_line
