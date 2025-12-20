"""
Base classes for language-specific AST parsers.

Provides the abstract interface and common utilities for all language parsers.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ASTNode:
    """AST node information representing a code structure."""

    node_type: str  # 'function', 'class', 'method'
    name: str  # Function/class name
    start_line: int  # Start line number (1-based)
    end_line: int  # End line number (1-based, inclusive)
    content: str  # Source code content of the node
    parent_name: str | None = None  # Parent class name (only for 'method' type)
    docstring: str | None = None  # Documentation string if present


class LanguageParser(ABC):
    """
    Abstract base class for language-specific parsers.

    Each language parser implements the strategy pattern to handle
    language-specific AST traversal and node extraction.
    """

    @property
    @abstractmethod
    def language_id(self) -> str:
        """Return the language identifier (e.g., 'python', 'javascript')."""
        pass

    @property
    @abstractmethod
    def tree_sitter_module(self) -> str:
        """Return the tree-sitter module name for this language."""
        pass

    @abstractmethod
    def extract_nodes(self, root_node: Any, content: str) -> list[ASTNode]:
        """
        Extract AST nodes from the parsed tree.

        Args:
            root_node: Tree-sitter root node
            content: Original source code

        Returns:
            List of ASTNode objects.
        """
        pass

    # Common utility methods

    def get_node_content(self, node: Any, content: str) -> str:
        """Extract the source code content for a node."""
        return content[node.start_byte:node.end_byte]

    def get_line_numbers(self, node: Any) -> tuple:
        """Get 1-based start and end line numbers for a node."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        return start_line, end_line
