"""
Java language parser.

Extracts classes and methods from Java code using Tree-sitter.
Includes Javadoc extraction.
"""

from typing import Any, List, Optional

from aci.core.parsers.base import ASTNode, LanguageParser
from aci.core.comment_extractor import extract_javadoc


class JavaParser(LanguageParser):
    """Parser for Java source code."""

    @property
    def language_id(self) -> str:
        return "java"

    @property
    def tree_sitter_module(self) -> str:
        return "tree_sitter_java"

    def extract_nodes(self, root_node: Any, content: str) -> List[ASTNode]:
        """Extract classes and methods from Java code."""
        nodes: List[ASTNode] = []
        self._traverse(root_node, content, nodes, parent_class=None, root=root_node)
        return nodes

    def _traverse(
        self,
        node: Any,
        content: str,
        nodes: List[ASTNode],
        parent_class: Optional[str],
        root: Any,
    ) -> None:
        """Recursively traverse Java AST and extract nodes."""
        class_types = {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        }

        if node.type in class_types:
            class_node = self._extract_class(node, content, root)
            class_name = class_node.name if class_node else parent_class
            if class_node:
                nodes.append(class_node)
            for child in node.children:
                self._traverse(child, content, nodes, class_name, root)
            return

        if node.type in ("method_declaration", "constructor_declaration"):
            method_node = self._extract_method(node, content, parent_class, root)
            if method_node:
                nodes.append(method_node)
            return

        for child in node.children:
            self._traverse(child, content, nodes, parent_class, root)

    def _extract_class(self, node: Any, content: str, root: Any) -> Optional[ASTNode]:
        """Extract a Java class/interface/enum/record declaration."""
        name = None
        for child in node.children:
            if child.type == "identifier":
                name = self.get_node_content(child, content)
                break

        if not name:
            return None

        start_line, end_line = self.get_line_numbers(node)
        node_content = self.get_node_content(node, content)
        docstring = extract_javadoc(node, content, root, node_name=name)

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
        self, node: Any, content: str, parent_class: Optional[str], root: Any
    ) -> Optional[ASTNode]:
        """Extract a Java method or constructor declaration."""
        name = None
        for child in node.children:
            if child.type == "identifier":
                name = self.get_node_content(child, content)
                break

        if not name:
            return None

        start_line, end_line = self.get_line_numbers(node)
        node_content = self.get_node_content(node, content)
        docstring = extract_javadoc(node, content, root, node_name=name)

        return ASTNode(
            node_type="method",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=parent_class,
            docstring=docstring,
        )
