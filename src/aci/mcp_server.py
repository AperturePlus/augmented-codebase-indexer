"""
MCP Server for Project ACI.

Provides Model Context Protocol interface for semantic code search and indexing.

This module serves as the entry point for the MCP server. The implementation
is split across submodules:
- mcp/context.py: MCPContext dataclass and factory functions
- mcp/tools.py: Tool definitions and schemas
- mcp/handlers.py: Tool handler implementations
"""

import asyncio
import os
import sys

# Load .env BEFORE importing handlers (so ACI_ENV is available)
# Try multiple locations: CWD, then script directory
from pathlib import Path
from dotenv import load_dotenv

# First try CWD
if not load_dotenv():
    # Fallback: try to find .env relative to this file's package
    # This helps when MCP is started from a different working directory
    _pkg_dir = Path(__file__).parent.parent.parent  # src/aci -> src -> project root
    _env_file = _pkg_dir / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
        print(f"[ACI-MCP] Loaded .env from {_env_file}", file=__import__('sys').stderr, flush=True)

from mcp.server import Server
from mcp.server.stdio import stdio_server

from aci.mcp.context import MCPContext, create_mcp_context, cleanup_context
from aci.mcp.tools import list_tools
from aci.mcp.handlers import call_tool

# Re-export for backward compatibility
__all__ = ["app", "list_tools", "call_tool", "main"]

# Initialize MCP server
app = Server("aci-mcp-server")

# Module-level context, created at startup
_ctx: MCPContext | None = None


@app.list_tools()
async def _list_tools():
    """List available MCP tools."""
    return list_tools()


@app.call_tool()
async def _call_tool(name: str, arguments):
    """Handle tool calls from MCP clients."""
    return await call_tool(name, arguments, _ctx)


async def _run_server():
    """Run the MCP server (async implementation)."""
    global _ctx
    
    # Print startup info to stderr (won't interfere with stdio protocol)
    env = os.environ.get("ACI_ENV", "production")
    if env == "development":
        print(f"[ACI-MCP] Starting in {env} mode (debug enabled)", file=sys.stderr, flush=True)
    
    # Create context with all services at startup
    _ctx = create_mcp_context()
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    finally:
        # Clean up connections on shutdown
        if _ctx is not None:
            await cleanup_context(_ctx)
            _ctx = None


def main():
    """Entry point for the MCP server."""
    asyncio.run(_run_server())


if __name__ == "__main__":
    main()
