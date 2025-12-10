"""
MCP (Model Context Protocol) server package for ACI.

This package provides the MCP interface for semantic code search and indexing.
"""

from aci.mcp.services import (
    get_initialized_services,
    cleanup_services,
)
from aci.mcp.tools import list_tools
from aci.mcp.handlers import call_tool

__all__ = [
    "get_initialized_services",
    "cleanup_services",
    "list_tools",
    "call_tool",
]
