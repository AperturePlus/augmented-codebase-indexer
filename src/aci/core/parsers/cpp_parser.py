"""
C and C++ language parsers.

Extracts functions and class/struct definitions using Tree-sitter.
Includes Doxygen comment extraction.
"""

from typing import Any

from aci.core.comment_extractor import extract_doxygen
from aci.core.parsers.base import ASTNode, LanguageParser


class _CBaseParser(LanguageParser):
    """Shared helpers for C-family parsers."""

    def _find_identifier(self, node: Any, content: str) -> str | None:
        """Depth-first search for the first identifier-like node."""
        stack = [node]
        while stack:
            current = stack.pop()
            if current.type in ("identifier", "field_identifier", "type_identifier"):
                return self.get_node_content(current, content)
            stack.extend(reversed(current.children))
        return None

    def _extract_function(
        self, node: Any, content: str, parent_class: str | None, root: Any
    ) -> ASTNode | None:
        """Extract a function or method definition/declaration."""
        name = self._find_identifier(node, content)
        if not name:
            return None

        start_line, end_line = self.get_line_numbers(node)
        node_content = self.get_node_content(node, content)
        docstring = extract_doxygen(node, content, root, node_name=name)

        return ASTNode(
            node_type="method" if parent_class else "function",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=parent_class,
            docstring=docstring,
        )


class CppParser(_CBaseParser):
    """Parser for C++ source code (also used for headers)."""

    @property
    def language_id(self) -> str:
        return "cpp"

    @property
    def tree_sitter_module(self) -> str:
        return "tree_sitter_cpp"

    def extract_nodes(self, root_node: Any, content: str) -> list[ASTNode]:
        nodes: list[ASTNode] = []
        self._traverse(root_node, content, nodes, parent_class=None, root=root_node)
        return nodes

    def _traverse(
        self,
        node: Any,
        content: str,
        nodes: list[ASTNode],
        parent_class: str | None,
        root: Any,
    ) -> None:
        if node.type in ("class_specifier", "struct_specifier"):
            class_node = self._extract_class(node, content, root)
            class_name = class_node.name if class_node else parent_class
            if class_node:
                nodes.append(class_node)
            for child in node.children:
                self._traverse(child, content, nodes, class_name, root)
            return

        if node.type in ("function_definition", "function_declaration"):
            func_node = self._extract_function(node, content, parent_class, root)
            if func_node:
                nodes.append(func_node)
            return

        if node.type == "field_declaration":
            method_node = self._extract_method_from_field(node, content, parent_class, root)
            if method_node:
                nodes.append(method_node)
            return

        for child in node.children:
            self._traverse(child, content, nodes, parent_class, root)

    def _extract_class(self, node: Any, content: str, root: Any) -> ASTNode | None:
        """Extract a C++ class/struct definition."""
        name = self._find_identifier(node, content)
        if not name:
            return None

        start_line, end_line = self.get_line_numbers(node)
        node_content = self.get_node_content(node, content)
        docstring = extract_doxygen(node, content, root, node_name=name)

        return ASTNode(
            node_type="class",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=None,
            docstring=docstring,
        )

    def _extract_method_from_field(
        self, node: Any, content: str, parent_class: str | None, root: Any
    ) -> ASTNode | None:
        """
        Extract a method declaration that appears as a field_declaration inside a class.
        This captures declarations like `int compute();` in class bodies.
        """
        if parent_class is None:
            return None

        # Check if this field contains a function declarator
        contains_func = any(child.type == "function_declarator" for child in node.children)
        if not contains_func:
            return None

        name = self._find_identifier(node, content)
        if not name:
            return None

        start_line, end_line = self.get_line_numbers(node)
        node_content = self.get_node_content(node, content)
        docstring = extract_doxygen(node, content, root, node_name=name)

        return ASTNode(
            node_type="method",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=parent_class,
            docstring=docstring,
        )


class CParser(_CBaseParser):
    """Parser for C source code."""

    @property
    def language_id(self) -> str:
        return "c"

    @property
    def tree_sitter_module(self) -> str:
        return "tree_sitter_c"

    def extract_nodes(self, root_node: Any, content: str) -> list[ASTNode]:
        nodes: list[ASTNode] = []
        self._traverse(root_node, content, nodes, root=root_node)
        return nodes

    def _traverse(self, node: Any, content: str, nodes: list[ASTNode], root: Any) -> None:
        if node.type in ("function_definition", "function_declaration"):
            func_node = self._extract_function(node, content, parent_class=None, root=root)
            if func_node:
                nodes.append(func_node)
            return

        for child in node.children:
            self._traverse(child, content, nodes, root)
