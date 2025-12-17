"""
Service constants for MCP server.

This module previously contained the Service Locator pattern implementation.
That has been replaced by dependency injection via MCPContext.

For service initialization, use:
    from aci.mcp.context import create_mcp_context, cleanup_context
"""

# Maximum allowed workers (matches HTTP API limit)
MAX_WORKERS = 32
