"""
REPL module for ACI interactive shell.

This package provides the interactive shell components including
context management, command completion, indexing operations, search operations,
event loop management, syntax highlighting, dynamic prompts, and the main controller.
"""

from aci.cli.repl.completer import CommandCompleter
from aci.cli.repl.context import REPLContext
from aci.cli.repl.controller import REPLController
from aci.cli.repl.event_loop import EventLoopManager
from aci.cli.repl.indexing_ops import IndexingOperations
from aci.cli.repl.lexer import CommandLexer
from aci.cli.repl.prompt import PromptBuilder
from aci.cli.repl.search_ops import SearchOperations

__all__ = [
    "CommandCompleter",
    "CommandLexer",
    "EventLoopManager",
    "IndexingOperations",
    "PromptBuilder",
    "REPLContext",
    "REPLController",
    "SearchOperations",
]
