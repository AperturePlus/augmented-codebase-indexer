"""
AST Parser - Tree-sitter based code structure parsing.

This module provides AST parsing capabilities for extracting semantic code structures
(functions, classes, methods) from source code using Tree-sitter.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ASTNode:
    """AST node information representing a code structure."""

    node_type: str  # 'function', 'class', 'method'
    name: str  # Function/class name
    start_line: int  # Start line number (1-based)
    end_line: int  # End line number (1-based, inclusive)
    content: str  # Source code content of the node
    parent_name: Optional[str] = None  # Parent class name (only for 'method' type)
    docstring: Optional[str] = None  # Documentation string if present


class ASTParserInterface(ABC):
    """Abstract interface for AST parsing."""

    @abstractmethod
    def parse(self, content: str, language: str) -> List[ASTNode]:
        """
        Parse code content and return AST nodes.

        Args:
            content: Source code content to parse
            language: Language identifier ('python', 'javascript', 'typescript', 'go')

        Returns:
            List of ASTNode objects representing functions, classes, and methods.
            Line numbers are 1-based.
        """
        pass

    @abstractmethod
    def supports_language(self, language: str) -> bool:
        """
        Check if the parser supports the specified language.

        Args:
            language: Language identifier

        Returns:
            True if the language is supported, False otherwise.
        """
        pass


# Supported languages and their Tree-sitter language modules
SUPPORTED_LANGUAGES = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_javascript",  # Uses same parser
    "go": "tree_sitter_go",
    "java": "tree_sitter_java",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
}


class TreeSitterParser(ASTParserInterface):
    """Tree-sitter based AST parser implementation."""

    def __init__(self):
        """Initialize the parser with Tree-sitter languages."""
        self._parsers: Dict[str, Any] = {}
        self._languages: Dict[str, Any] = {}
        self._initialized_languages: set = set()

    def _ensure_language_loaded(self, language: str) -> bool:
        """
        Lazily load a language parser when first needed.

        Args:
            language: Language identifier

        Returns:
            True if language was loaded successfully, False otherwise.
        """
        if language in self._initialized_languages:
            return language in self._parsers

        self._initialized_languages.add(language)

        if language not in SUPPORTED_LANGUAGES:
            logger.warning(f"Language '{language}' is not supported")
            return False

        module_name = SUPPORTED_LANGUAGES[language]

        try:
            import tree_sitter

            # Import the language module
            if module_name == "tree_sitter_python":
                import tree_sitter_python

                lang = tree_sitter.Language(tree_sitter_python.language())
            elif module_name == "tree_sitter_javascript":
                import tree_sitter_javascript

                lang = tree_sitter.Language(tree_sitter_javascript.language())
            elif module_name == "tree_sitter_go":
                import tree_sitter_go

                lang = tree_sitter.Language(tree_sitter_go.language())
            elif module_name == "tree_sitter_java":
                import tree_sitter_java

                lang = tree_sitter.Language(tree_sitter_java.language())
            elif module_name == "tree_sitter_c":
                import tree_sitter_c

                lang = tree_sitter.Language(tree_sitter_c.language())
            elif module_name == "tree_sitter_cpp":
                import tree_sitter_cpp

                lang = tree_sitter.Language(tree_sitter_cpp.language())
            else:
                logger.error(f"Unknown language module: {module_name}")
                return False

            parser = tree_sitter.Parser(lang)
            self._parsers[language] = parser
            self._languages[language] = lang
            logger.debug(f"Loaded Tree-sitter parser for '{language}'")
            return True

        except ImportError as e:
            logger.error(f"Failed to import Tree-sitter language module for '{language}': {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Tree-sitter parser for '{language}': {e}")
            return False

    def supports_language(self, language: str) -> bool:
        """Check if the parser supports the specified language."""
        return language in SUPPORTED_LANGUAGES

    def parse(self, content: str, language: str) -> List[ASTNode]:
        """
        Parse code content and return AST nodes.

        Args:
            content: Source code content to parse
            language: Language identifier

        Returns:
            List of ASTNode objects. Returns empty list if language not supported.
        """
        if not self._ensure_language_loaded(language):
            return []

        parser = self._parsers.get(language)
        if not parser:
            return []

        try:
            tree = parser.parse(content.encode("utf-8"))
            return self._extract_nodes(tree.root_node, content, language)
        except Exception as e:
            logger.error(f"Failed to parse content for language '{language}': {e}")
            return []

    def _extract_nodes(self, root_node: Any, content: str, language: str) -> List[ASTNode]:
        """
        Extract AST nodes from the parsed tree.

        Args:
            root_node: Tree-sitter root node
            content: Original source code
            language: Language identifier

        Returns:
            List of ASTNode objects.
        """
        if language == "python":
            return self._extract_python_nodes(root_node, content)
        elif language in ("javascript", "typescript"):
            return self._extract_javascript_nodes(root_node, content)
        elif language == "go":
            return self._extract_go_nodes(root_node, content)
        return []

    def _get_node_content(self, node: Any, content: str) -> str:
        """Extract the source code content for a node."""
        return content[node.start_byte : node.end_byte]

    def _get_line_numbers(self, node: Any) -> tuple:
        """Get 1-based start and end line numbers for a node."""
        # Tree-sitter uses 0-based line numbers
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        return start_line, end_line

    # ========== Python Language Support ==========

    def _extract_python_nodes(self, root_node: Any, content: str) -> List[ASTNode]:
        """Extract functions, classes, and methods from Python code."""
        nodes = []
        self._traverse_python(root_node, content, nodes, parent_class=None)
        return nodes

    def _traverse_python(
        self, node: Any, content: str, nodes: List[ASTNode], parent_class: Optional[str]
    ) -> None:
        """Recursively traverse Python AST and extract nodes."""
        if node.type == "function_definition":
            ast_node = self._extract_python_function(node, content, parent_class)
            if ast_node:
                nodes.append(ast_node)
            # Don't traverse into nested functions for now
            return

        elif node.type == "class_definition":
            class_node = self._extract_python_class(node, content)
            if class_node:
                nodes.append(class_node)
                # Traverse class body to find methods
                class_name = class_node.name
                for child in node.children:
                    if child.type == "block":
                        for block_child in child.children:
                            self._traverse_python(block_child, content, nodes, class_name)
            return

        # Continue traversing other nodes
        for child in node.children:
            self._traverse_python(child, content, nodes, parent_class)

    def _extract_python_function(
        self, node: Any, content: str, parent_class: Optional[str]
    ) -> Optional[ASTNode]:
        """Extract a Python function or method definition."""
        name = None
        docstring = None

        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_content(child, content)
            elif child.type == "block":
                # Look for docstring as first statement
                docstring = self._extract_python_docstring(child, content)

        if not name:
            return None

        start_line, end_line = self._get_line_numbers(node)
        node_content = self._get_node_content(node, content)

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

    def _extract_python_class(self, node: Any, content: str) -> Optional[ASTNode]:
        """Extract a Python class definition."""
        name = None
        docstring = None

        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_content(child, content)
            elif child.type == "block":
                docstring = self._extract_python_docstring(child, content)

        if not name:
            return None

        start_line, end_line = self._get_line_numbers(node)
        node_content = self._get_node_content(node, content)

        return ASTNode(
            node_type="class",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=None,
            docstring=docstring,
        )

    def _extract_python_docstring(self, block_node: Any, content: str) -> Optional[str]:
        """Extract docstring from a Python block (first expression statement with string)."""
        for child in block_node.children:
            if child.type == "expression_statement":
                for expr_child in child.children:
                    if expr_child.type == "string":
                        docstring = self._get_node_content(expr_child, content)
                        # Remove quotes
                        if docstring.startswith('"""') or docstring.startswith("'''"):
                            return docstring[3:-3].strip()
                        elif docstring.startswith('"') or docstring.startswith("'"):
                            return docstring[1:-1].strip()
                        return docstring
                break  # Only check first statement
        return None

    # ========== JavaScript/TypeScript Language Support ==========

    def _extract_javascript_nodes(self, root_node: Any, content: str) -> List[ASTNode]:
        """Extract functions, classes, and methods from JavaScript/TypeScript code."""
        nodes = []
        self._traverse_javascript(root_node, content, nodes, parent_class=None)
        return nodes

    def _traverse_javascript(
        self, node: Any, content: str, nodes: List[ASTNode], parent_class: Optional[str]
    ) -> None:
        """Recursively traverse JavaScript AST and extract nodes."""
        # Function declaration: function foo() {}
        if node.type == "function_declaration":
            ast_node = self._extract_js_function_declaration(node, content, parent_class)
            if ast_node:
                nodes.append(ast_node)
            return

        # Arrow function in variable declaration: const foo = () => {}
        elif node.type == "lexical_declaration" or node.type == "variable_declaration":
            ast_node = self._extract_js_arrow_function(node, content, parent_class)
            if ast_node:
                nodes.append(ast_node)
            return

        # Class declaration
        elif node.type == "class_declaration":
            class_node = self._extract_js_class(node, content)
            if class_node:
                nodes.append(class_node)
                # Traverse class body to find methods
                class_name = class_node.name
                for child in node.children:
                    if child.type == "class_body":
                        for body_child in child.children:
                            self._traverse_javascript(body_child, content, nodes, class_name)
            return

        # Method definition inside class
        elif node.type == "method_definition":
            ast_node = self._extract_js_method(node, content, parent_class)
            if ast_node:
                nodes.append(ast_node)
            return

        # Continue traversing
        for child in node.children:
            self._traverse_javascript(child, content, nodes, parent_class)

    def _extract_js_function_declaration(
        self, node: Any, content: str, parent_class: Optional[str]
    ) -> Optional[ASTNode]:
        """Extract a JavaScript function declaration."""
        name = None

        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_content(child, content)
                break

        if not name:
            return None

        start_line, end_line = self._get_line_numbers(node)
        node_content = self._get_node_content(node, content)

        return ASTNode(
            node_type="method" if parent_class else "function",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=parent_class,
            docstring=None,
        )

    def _extract_js_arrow_function(
        self, node: Any, content: str, parent_class: Optional[str]
    ) -> Optional[ASTNode]:
        """Extract an arrow function from variable declaration."""
        name = None
        has_arrow_function = False

        for child in node.children:
            if child.type == "variable_declarator":
                for decl_child in child.children:
                    if decl_child.type == "identifier":
                        name = self._get_node_content(decl_child, content)
                    elif decl_child.type == "arrow_function":
                        has_arrow_function = True

        if not name or not has_arrow_function:
            return None

        start_line, end_line = self._get_line_numbers(node)
        node_content = self._get_node_content(node, content)

        return ASTNode(
            node_type="method" if parent_class else "function",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=parent_class,
            docstring=None,
        )

    def _extract_js_class(self, node: Any, content: str) -> Optional[ASTNode]:
        """Extract a JavaScript class declaration."""
        name = None

        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_content(child, content)
                break

        if not name:
            return None

        start_line, end_line = self._get_line_numbers(node)
        node_content = self._get_node_content(node, content)

        return ASTNode(
            node_type="class",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=None,
            docstring=None,
        )

    def _extract_js_method(
        self, node: Any, content: str, parent_class: Optional[str]
    ) -> Optional[ASTNode]:
        """Extract a method from a JavaScript class."""
        name = None

        for child in node.children:
            if child.type == "property_identifier":
                name = self._get_node_content(child, content)
                break

        if not name:
            return None

        start_line, end_line = self._get_line_numbers(node)
        node_content = self._get_node_content(node, content)

        return ASTNode(
            node_type="method",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=parent_class,
            docstring=None,
        )

    # ========== Go Language Support ==========

    def _extract_go_nodes(self, root_node: Any, content: str) -> List[ASTNode]:
        """Extract functions, structs, and methods from Go code."""
        nodes = []
        self._traverse_go(root_node, content, nodes)
        return nodes

    def _traverse_go(self, node: Any, content: str, nodes: List[ASTNode]) -> None:
        """Recursively traverse Go AST and extract nodes."""
        # Function declaration: func foo() {}
        if node.type == "function_declaration":
            ast_node = self._extract_go_function(node, content)
            if ast_node:
                nodes.append(ast_node)
            return

        # Method declaration: func (r *Receiver) foo() {}
        elif node.type == "method_declaration":
            ast_node = self._extract_go_method(node, content)
            if ast_node:
                nodes.append(ast_node)
            return

        # Type declaration (for structs)
        elif node.type == "type_declaration":
            ast_node = self._extract_go_struct(node, content)
            if ast_node:
                nodes.append(ast_node)
            return

        # Continue traversing
        for child in node.children:
            self._traverse_go(child, content, nodes)

    def _extract_go_function(self, node: Any, content: str) -> Optional[ASTNode]:
        """Extract a Go function declaration."""
        name = None

        for child in node.children:
            if child.type == "identifier":
                name = self._get_node_content(child, content)
                break

        if not name:
            return None

        start_line, end_line = self._get_line_numbers(node)
        node_content = self._get_node_content(node, content)

        return ASTNode(
            node_type="function",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=None,
            docstring=None,
        )

    def _extract_go_method(self, node: Any, content: str) -> Optional[ASTNode]:
        """Extract a Go method declaration (function with receiver)."""
        name = None
        receiver_type = None
        found_receiver = False

        for child in node.children:
            if child.type == "parameter_list" and not found_receiver:
                # First parameter_list is the receiver
                receiver_type = self._extract_go_receiver_type(child, content)
                found_receiver = True
            elif child.type == "field_identifier":
                name = self._get_node_content(child, content)

        if not name:
            return None

        start_line, end_line = self._get_line_numbers(node)
        node_content = self._get_node_content(node, content)

        return ASTNode(
            node_type="method",
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=receiver_type,
            docstring=None,
        )

    def _extract_go_receiver_type(self, param_list: Any, content: str) -> Optional[str]:
        """Extract the receiver type from a Go method's parameter list."""
        for child in param_list.children:
            if child.type == "parameter_declaration":
                for param_child in child.children:
                    if param_child.type == "type_identifier":
                        return self._get_node_content(param_child, content)
                    elif param_child.type == "pointer_type":
                        # Handle *Type
                        for ptr_child in param_child.children:
                            if ptr_child.type == "type_identifier":
                                return self._get_node_content(ptr_child, content)
        return None

    def _extract_go_struct(self, node: Any, content: str) -> Optional[ASTNode]:
        """Extract a Go struct type declaration."""
        name = None
        is_struct = False

        for child in node.children:
            if child.type == "type_spec":
                for spec_child in child.children:
                    if spec_child.type == "type_identifier":
                        name = self._get_node_content(spec_child, content)
                    elif spec_child.type == "struct_type":
                        is_struct = True

        if not name or not is_struct:
            return None

        start_line, end_line = self._get_line_numbers(node)
        node_content = self._get_node_content(node, content)

        return ASTNode(
            node_type="class",  # Use 'class' for consistency with other languages
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=node_content,
            parent_name=None,
            docstring=None,
        )


def check_tree_sitter_setup() -> Dict[str, bool]:
    """
    Check Tree-sitter environment and language pack loading status.

    Returns:
        Dictionary mapping language names to their availability status.
    """
    results = {}
    parser = TreeSitterParser()

    for language in SUPPORTED_LANGUAGES:
        try:
            available = parser._ensure_language_loaded(language)
            results[language] = available
        except Exception as e:
            logger.error(f"Error checking language '{language}': {e}")
            results[language] = False

    return results
