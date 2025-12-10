"""
MCP Server for Project ACI.

Provides Model Context Protocol interface for semantic code search and indexing.

This module serves as the entry point for the MCP server. The implementation
is split across submodules:
- mcp/services.py: Service initialization and caching
- mcp/tools.py: Tool definitions and schemas
- mcp/handlers.py: Tool handler implementations
"""

import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server

from aci.mcp.services import cleanup_services
from aci.mcp.tools import list_tools
from aci.mcp.handlers import call_tool

# Re-export for backward compatibility
__all__ = ["app", "list_tools", "call_tool", "main"]

# Initialize MCP server
app = Server("aci-mcp-server")


@app.list_tools()
async def _list_tools():
    """List available MCP tools."""
    return list_tools()


@app.call_tool()
async def _call_tool(name: str, arguments):
    """Handle tool calls from MCP clients."""
    return await call_tool(name, arguments)


async def main():
    """Run the MCP server."""
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    finally:
        # Clean up connections on shutdown
        await cleanup_services()


if __name__ == "__main__":
    asyncio.run(main())
