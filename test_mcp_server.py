#!/usr/bin/env python3
"""
Simple test script for ACI MCP server.

This script tests the MCP server's tool listing functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for local testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aci.mcp_server import list_tools


async def test_list_tools():
    """Test that all expected tools are available."""
    print("Testing MCP server tool listing...\n")
    
    tools = await list_tools()
    
    print(f"Found {len(tools)} tools:\n")
    
    expected_tools = {
        "index_codebase",
        "search_code",
        "get_index_status",
        "update_index",
        "list_indexed_repos",
    }
    
    found_tools = {tool.name for tool in tools}
    
    for tool in tools:
        status = "✓" if tool.name in expected_tools else "?"
        print(f"{status} {tool.name}")
        print(f"  Description: {tool.description[:80]}...")
        print(f"  Required params: {tool.inputSchema.get('required', [])}")
        print()
    
    # Check if all expected tools are present
    missing = expected_tools - found_tools
    extra = found_tools - expected_tools
    
    if missing:
        print(f"❌ Missing tools: {missing}")
        return False
    
    if extra:
        print(f"⚠️  Extra tools: {extra}")
    
    print("✅ All expected tools are available!")
    return True


async def main():
    """Run tests."""
    try:
        success = await test_list_tools()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
