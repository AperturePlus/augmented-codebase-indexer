"""
JavaScript/TypeScript language parser.

Extracts functions, classes, and methods from JS/TS code using Tree-sitter.
Includes JSDoc comment extraction.
"""

from typing import Any

from aci.core.comment_extractor import extract_jsdoc
from aci.core.parsers.base import ASTNode, LanguageParser


class JavaScriptParser(LanguageParser):
    """Parser for JavaScript and TypeScript source code."""

    @property
    def language_id(self) -> str:
        return "javascript"

    @property
    def tree_sitter_module(self) -> str:
        return "tree_sitter_javascript"

    def extract_nodes(self, root_node: Any, content: str) -> list[ASTNode]:
        """Extract functions, classes, and methods from JavaScript code."""
        nodes = []
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
        """Recursively traverse JavaScript AST and extract nodes."""
        if node.type == "function_declaration":
            ast_node = self._extract_function(node, content, parent_class, root)
            if ast_node:
                nodes.append(ast_node)
            return

        elif node.type in ("lexical_declaration", "variable_declaration"):
            ast_node = self._extract_arrow_function(node, content, parent_class, root)
            if ast_node:
                nodes.append(ast_node)
            return

        elif node.type == "class_declaration":
            class_node = self._extract_class(node, content, root)
            if class_node:
                nodes.append(class_node)
                class_name = class_node.name
                for child in node.children:
                    if child.type == "class_body":
                        for body_child in child.children:
                            self._traverse(body_child, content, nodes, class_name, root)
            return

        elif node.type == "method_definition":
            ast_node = self._extract_method(node, content, parent_class, root)
            if ast_node:
                nodes.append(ast_node)
            return

        for child in node.children:
            self._traverse(child, content, nodes, parent_class, root)

    def _extract_function(
        self, node: Any, content: str, parent_class: str | None, root: Any
    ) -> ASTNode | None:
        """Extract a JavaScript function declaration."""
        name = None
        for child in node.children:
            if child.type == "identifier":
                name = self.get_node_content(child, content)
                break

        if not name:
            return None

        start_line, end_line = self.get_line_numbers(node)
        node_content = self.get_node_content(node, content)
        docstring = extract_jsdoc(node, content, root, node_name=name)

        return ASTNode(
            node_type="method" if parent_class else "function",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=parent_class,
            docstring=docstring,
        )

    def _extract_arrow_function(
        self, node: Any, content: str, parent_class: str | None, root: Any
    ) -> ASTNode | None:
        """Extract an arrow function from variable declaration."""
        name = None
        has_arrow = False

        for child in node.children:
            if child.type == "variable_declarator":
                for decl_child in child.children:
                    if decl_child.type == "identifier":
                        name = self.get_node_content(decl_child, content)
                    elif decl_child.type == "arrow_function":
                        has_arrow = True

        if not name or not has_arrow:
            return None

        start_line, end_line = self.get_line_numbers(node)
        node_content = self.get_node_content(node, content)
        docstring = extract_jsdoc(node, content, root, node_name=name)

        return ASTNode(
            node_type="method" if parent_class else "function",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=parent_class,
            docstring=docstring,
        )

    def _extract_class(self, node: Any, content: str, root: Any) -> ASTNode | None:
        """Extract a JavaScript class declaration."""
        name = None
        for child in node.children:
            if child.type == "identifier":
                name = self.get_node_content(child, content)
                break

        if not name:
            return None

        start_line, end_line = self.get_line_numbers(node)
        node_content = self.get_node_content(node, content)
        docstring = extract_jsdoc(node, content, root, node_name=name)

        return ASTNode(
            node_type="class",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=None,
            docstring=docstring,
        )

    def _extract_method(
        self, node: Any, content: str, parent_class: str | None, root: Any
    ) -> ASTNode | None:
        """Extract a method from a JavaScript class."""
        name = None
        for child in node.children:
            if child.type == "property_identifier":
                name = self.get_node_content(child, content)
                break

        if not name:
            return None

        start_line, end_line = self.get_line_numbers(node)
        node_content = self.get_node_content(node, content)
        docstring = extract_jsdoc(node, content, root, node_name=name)

        return ASTNode(
            node_type="method",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=parent_class,
            docstring=docstring,
        )
