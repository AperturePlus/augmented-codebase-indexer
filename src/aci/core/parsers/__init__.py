"""
Language-specific AST parsers.

This package contains strategy implementations for parsing different
programming languages using Tree-sitter.
"""

from aci.core.parsers.base import ASTNode, LanguageParser
from aci.core.parsers.cpp_parser import CParser, CppParser
from aci.core.parsers.go_parser import GoParser
from aci.core.parsers.java_parser import JavaParser
from aci.core.parsers.javascript_parser import JavaScriptParser
from aci.core.parsers.python_parser import PythonParser

__all__ = [
    "LanguageParser",
    "ASTNode",
    "PythonParser",
    "JavaScriptParser",
    "GoParser",
    "JavaParser",
    "CParser",
    "CppParser",
]
