"""
AST Parser - Tree-sitter based code structure parsing.

This module provides AST parsing capabilities for extracting semantic code structures
(functions, classes, methods) from source code using Tree-sitter.

Uses the Strategy pattern to delegate language-specific parsing to dedicated parsers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from aci.core.parsers.base import ASTNode, LanguageParser
from aci.core.parsers.cpp_parser import CParser, CppParser
from aci.core.parsers.go_parser import GoParser
from aci.core.parsers.java_parser import JavaParser
from aci.core.parsers.javascript_parser import JavaScriptParser
from aci.core.parsers.python_parser import PythonParser

logger = logging.getLogger(__name__)

# Re-export ASTNode for backward compatibility
__all__ = ["ASTNode", "ASTParserInterface", "TreeSitterParser", "SUPPORTED_LANGUAGES"]


class ASTParserInterface(ABC):
    """Abstract interface for AST parsing."""

    @abstractmethod
    def parse(self, content: str, language: str) -> list[ASTNode]:
        """Parse code content and return AST nodes."""
        pass

    @abstractmethod
    def supports_language(self, language: str) -> bool:
        """Check if the parser supports the specified language."""
        pass


# Supported languages and their Tree-sitter language modules
SUPPORTED_LANGUAGES = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_javascript",
    "go": "tree_sitter_go",
    "java": "tree_sitter_java",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
}


class TreeSitterParser(ASTParserInterface):
    """
    Tree-sitter based AST parser implementation.

    Uses the Strategy pattern to delegate language-specific parsing
    to dedicated LanguageParser implementations.
    """

    def __init__(self):
        """Initialize the parser with language strategies."""
        self._parsers: dict[str, Any] = {}
        self._languages: dict[str, Any] = {}
        self._initialized_languages: set = set()

        # Register language-specific parsers (Strategy pattern)
        self._language_parsers: dict[str, LanguageParser] = {
            "python": PythonParser(),
            "javascript": JavaScriptParser(),
            "typescript": JavaScriptParser(),  # Uses same parser
            "go": GoParser(),
            "java": JavaParser(),
            "c": CParser(),
            "cpp": CppParser(),
        }

    def _ensure_language_loaded(self, language: str) -> bool:
        """Lazily load a language parser when first needed."""
        if language in self._initialized_languages:
            return language in self._parsers

        self._initialized_languages.add(language)

        if language not in SUPPORTED_LANGUAGES:
            logger.warning(f"Language '{language}' is not supported")
            return False

        module_name = SUPPORTED_LANGUAGES[language]

        try:
            import tree_sitter
            lang = self._load_tree_sitter_language(module_name)
            if lang is None:
                return False

            parser = tree_sitter.Parser(lang)
            self._parsers[language] = parser
            self._languages[language] = lang
            logger.debug(f"Loaded Tree-sitter parser for '{language}'")
            return True

        except ImportError as e:
            logger.error(f"Failed to import Tree-sitter for '{language}': {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize parser for '{language}': {e}")
            return False

    def _load_tree_sitter_language(self, module_name: str) -> Any:
        """Load a Tree-sitter language module."""
        import tree_sitter

        loaders = {
            "tree_sitter_python": lambda: __import__("tree_sitter_python").language(),
            "tree_sitter_javascript": lambda: __import__("tree_sitter_javascript").language(),
            "tree_sitter_go": lambda: __import__("tree_sitter_go").language(),
            "tree_sitter_java": lambda: __import__("tree_sitter_java").language(),
            "tree_sitter_c": lambda: __import__("tree_sitter_c").language(),
            "tree_sitter_cpp": lambda: __import__("tree_sitter_cpp").language(),
        }

        loader = loaders.get(module_name)
        if loader:
            lang_obj = loader()
            if isinstance(lang_obj, tree_sitter.Language):
                return lang_obj
            return tree_sitter.Language(lang_obj)

        logger.error(f"Unknown language module: {module_name}")
        return None

    def supports_language(self, language: str) -> bool:
        """
        Check if the parser fully supports the specified language.

        Returns True only if both:
        1. Tree-sitter grammar is available for the language
        2. A language-specific parser implementation exists
        """
        return language in SUPPORTED_LANGUAGES and language in self._language_parsers

    def parse(self, content: str, language: str) -> list[ASTNode]:
        """
        Parse code content and return AST nodes.

        Delegates to language-specific parsers using the Strategy pattern.
        """
        if not self._ensure_language_loaded(language):
            return []

        parser = self._parsers.get(language)
        if not parser:
            return []

        try:
            tree = parser.parse(content.encode("utf-8"))

            # Use language-specific parser if available
            lang_parser = self._language_parsers.get(language)
            if lang_parser:
                return lang_parser.extract_nodes(tree.root_node, content)

            # Language has tree-sitter support but no dedicated parser
            logger.info(
                f"Language '{language}' has tree-sitter grammar available but no "
                f"dedicated parser implementation yet. Consider adding a parser "
                f"in src/aci/core/parsers/{language}_parser.py"
            )
            return []

        except Exception as e:
            logger.error(f"Failed to parse content for '{language}': {e}")
            # Re-raise in debug mode for better debugging
            if logger.isEnabledFor(logging.DEBUG):
                raise
            return []


def check_tree_sitter_setup() -> dict[str, bool]:
    """Check Tree-sitter environment and language pack loading status."""
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
