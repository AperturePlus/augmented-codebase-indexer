"""
Go language parser.

Extracts functions, structs, and methods from Go code using Tree-sitter.
Includes Go doc comment extraction.
"""

from typing import Any, List, Optional

from aci.core.parsers.base import LanguageParser, ASTNode
from aci.core.comment_extractor import extract_go_doc


class GoParser(LanguageParser):
    """Parser for Go source code."""

    @property
    def language_id(self) -> str:
        return "go"

    @property
    def tree_sitter_module(self) -> str:
        return "tree_sitter_go"

    def extract_nodes(self, root_node: Any, content: str) -> List[ASTNode]:
        """Extract functions, structs, and methods from Go code."""
        nodes = []
        self._traverse(root_node, content, nodes, root=root_node)
        return nodes

    def _traverse(self, node: Any, content: str, nodes: List[ASTNode], root: Any) -> None:
        """Recursively traverse Go AST and extract nodes."""
        if node.type == "function_declaration":
            ast_node = self._extract_function(node, content, root)
            if ast_node:
                nodes.append(ast_node)
            return

        elif node.type == "method_declaration":
            ast_node = self._extract_method(node, content, root)
            if ast_node:
                nodes.append(ast_node)
            return

        elif node.type == "type_declaration":
            ast_node = self._extract_struct(node, content, root)
            if ast_node:
                nodes.append(ast_node)
            return

        for child in node.children:
            self._traverse(child, content, nodes, root)

    def _extract_function(self, node: Any, content: str, root: Any) -> Optional[ASTNode]:
        """Extract a Go function declaration."""
        name = None
        for child in node.children:
            if child.type == "identifier":
                name = self.get_node_content(child, content)
                break

        if not name:
            return None

        start_line, end_line = self.get_line_numbers(node)
        node_content = self.get_node_content(node, content)
        docstring = extract_go_doc(node, content, root, node_name=name)

        return ASTNode(
            node_type="function",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=None,
            docstring=docstring,
        )

    def _extract_method(self, node: Any, content: str, root: Any) -> Optional[ASTNode]:
        """Extract a Go method declaration (function with receiver)."""
        name = None
        receiver_type = None
        found_receiver = False

        for child in node.children:
            if child.type == "parameter_list" and not found_receiver:
                receiver_type = self._extract_receiver_type(child, content)
                found_receiver = True
            elif child.type == "field_identifier":
                name = self.get_node_content(child, content)

        if not name:
            return None

        start_line, end_line = self.get_line_numbers(node)
        node_content = self.get_node_content(node, content)
        docstring = extract_go_doc(node, content, root, node_name=name)

        return ASTNode(
            node_type="method",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=receiver_type,
            docstring=docstring,
        )

    def _extract_receiver_type(self, param_list: Any, content: str) -> Optional[str]:
        """Extract the receiver type from a Go method's parameter list."""
        for child in param_list.children:
            if child.type == "parameter_declaration":
                for param_child in child.children:
                    if param_child.type == "type_identifier":
                        return self.get_node_content(param_child, content)
                    elif param_child.type == "pointer_type":
                        for ptr_child in param_child.children:
                            if ptr_child.type == "type_identifier":
                                return self.get_node_content(ptr_child, content)
        return None

    def _extract_struct(self, node: Any, content: str, root: Any) -> Optional[ASTNode]:
        """Extract a Go struct type declaration."""
        name = None
        is_struct = False

        for child in node.children:
            if child.type == "type_spec":
                for spec_child in child.children:
                    if spec_child.type == "type_identifier":
                        name = self.get_node_content(spec_child, content)
                    elif spec_child.type == "struct_type":
                        is_struct = True

        if not name or not is_struct:
            return None

        start_line, end_line = self.get_line_numbers(node)
        node_content = self.get_node_content(node, content)
        docstring = extract_go_doc(node, content, root, node_name=name)

        return ASTNode(
            node_type="class",  # Use 'class' for consistency
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=None,
            docstring=docstring,
        )
