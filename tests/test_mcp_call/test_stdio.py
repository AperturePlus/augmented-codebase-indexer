"""
Test MCP server via stdio communication.

Usage: uv run python tests/test_mcp_call/test_stdio.py
"""

import asyncio
import json
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_stdio():
    """Test MCP server through actual stdio communication."""
    
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "aci-mcp"],
        env=None,
    )
    
    print("Starting MCP server via stdio...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            print("✓ Server initialized")
            
            # List tools
            print("\n=== Available Tools ===")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description[:60]}...")
            
            # Test list_indexed_repos
            print("\n=== Testing list_indexed_repos ===")
            result = await session.call_tool("list_indexed_repos", {})
            for content in result.content:
                if hasattr(content, 'text'):
                    print(content.text)
            
            # Test get_index_status
            print("\n=== Testing get_index_status ===")
            result = await session.call_tool("get_index_status", {})
            for content in result.content:
                if hasattr(content, 'text'):
                    data = json.loads(content.text)
                    print(json.dumps(data, indent=2))
            
            # Test update_index on current codebase
            print("\n=== Testing update_index ===")
            import os
            codebase_path = os.getcwd()
            print(f"Updating index for: {codebase_path}")
            result = await session.call_tool("update_index", {"path": codebase_path})
            for content in result.content:
                if hasattr(content, 'text'):
                    try:
                        data = json.loads(content.text)
                        print(json.dumps(data, indent=2))
                    except json.JSONDecodeError:
                        print(content.text)
            
            print("\n✓ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_mcp_stdio())
