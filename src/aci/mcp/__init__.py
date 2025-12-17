"""
MCP (Model Context Protocol) server package for ACI.

This package provides the MCP interface for semantic code search and indexing.
"""

from aci.mcp.context import MCPContext, create_mcp_context, cleanup_context
from aci.mcp.tools import list_tools
from aci.mcp.handlers import call_tool

# Backward compatibility: cleanup_services is an alias for cleanup_context
cleanup_services = cleanup_context

__all__ = [
    # DI-based exports
    "MCPContext",
    "create_mcp_context",
    "cleanup_context",
    # Core exports
    "list_tools",
    "call_tool",
    # Backward compatibility alias
    "cleanup_services",
]
