"""
Python language parser.

Extracts functions, classes, and methods from Python code using Tree-sitter.
"""

from typing import Any

from aci.core.parsers.base import ASTNode, LanguageParser


class PythonParser(LanguageParser):
    """Parser for Python source code."""

    @property
    def language_id(self) -> str:
        return "python"

    @property
    def tree_sitter_module(self) -> str:
        return "tree_sitter_python"

    def extract_nodes(self, root_node: Any, content: str) -> list[ASTNode]:
        """Extract functions, classes, and methods from Python code."""
        nodes = []
        self._traverse(root_node, content, nodes, parent_class=None)
        return nodes

    def _traverse(
        self, node: Any, content: str, nodes: list[ASTNode], parent_class: str | None
    ) -> None:
        """Recursively traverse Python AST and extract nodes."""
        if node.type == "function_definition":
            ast_node = self._extract_function(node, content, parent_class)
            if ast_node:
                nodes.append(ast_node)
            return

        elif node.type == "class_definition":
            class_node = self._extract_class(node, content)
            if class_node:
                nodes.append(class_node)
                class_name = class_node.name
                for child in node.children:
                    if child.type == "block":
                        for block_child in child.children:
                            self._traverse(block_child, content, nodes, class_name)
            return

        for child in node.children:
            self._traverse(child, content, nodes, parent_class)

    def _extract_function(
        self, node: Any, content: str, parent_class: str | None
    ) -> ASTNode | None:
        """Extract a Python function or method definition."""
        name = None
        docstring = None

        for child in node.children:
            if child.type == "identifier":
                name = self.get_node_content(child, content)
            elif child.type == "block":
                docstring = self._extract_docstring(child, content)

        if not name:
            return None

        start_line, end_line = self.get_line_numbers(node)
        node_content = self.get_node_content(node, content)
        node_type = "method" if parent_class else "function"

        return ASTNode(
            node_type=node_type,
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=parent_class,
            docstring=docstring,
        )

    def _extract_class(self, node: Any, content: str) -> ASTNode | None:
        """Extract a Python class definition."""
        name = None
        docstring = None

        for child in node.children:
            if child.type == "identifier":
                name = self.get_node_content(child, content)
            elif child.type == "block":
                docstring = self._extract_docstring(child, content)

        if not name:
            return None

        start_line, end_line = self.get_line_numbers(node)
        node_content = self.get_node_content(node, content)

        return ASTNode(
            node_type="class",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=None,
            docstring=docstring,
        )

    def _extract_docstring(self, block_node: Any, content: str) -> str | None:
        """Extract docstring from a Python block."""
        for child in block_node.children:
            if child.type == "expression_statement":
                for expr_child in child.children:
                    if expr_child.type == "string":
                        docstring = self.get_node_content(expr_child, content)
                        if docstring.startswith('"""') or docstring.startswith("'''"):
                            return docstring[3:-3].strip()
                        elif docstring.startswith('"') or docstring.startswith("'"):
                            return docstring[1:-1].strip()
                        return docstring
                break
        return None
