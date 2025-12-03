"""
Tests for ACI MCP server.

Tests the MCP server's tool listing and basic functionality.
"""

import asyncio

from aci.mcp_server import list_tools


def _run_async(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def test_list_tools_returns_expected_tools():
    """Test that all expected tools are available."""
    tools = _run_async(list_tools())

    expected_tools = {
        "index_codebase",
        "search_code",
        "get_index_status",
        "update_index",
        "list_indexed_repos",
    }

    found_tools = {tool.name for tool in tools}

    # Check all expected tools are present
    missing = expected_tools - found_tools
    assert not missing, f"Missing tools: {missing}"

    # Verify we have at least the expected number of tools
    assert len(tools) >= len(expected_tools)


def test_list_tools_have_descriptions():
    """Test that all tools have descriptions."""
    tools = _run_async(list_tools())

    for tool in tools:
        assert tool.description, f"Tool {tool.name} has no description"
        assert len(tool.description) > 10, f"Tool {tool.name} has too short description"


def test_list_tools_have_input_schemas():
    """Test that all tools have input schemas."""
    tools = _run_async(list_tools())

    for tool in tools:
        assert tool.inputSchema, f"Tool {tool.name} has no input schema"
        assert "type" in tool.inputSchema, f"Tool {tool.name} schema missing 'type'"
        assert tool.inputSchema["type"] == "object"


def test_search_code_tool_has_mode_parameter():
    """Test that search_code tool has the mode parameter for hybrid search."""
    tools = _run_async(list_tools())

    search_tool = next((t for t in tools if t.name == "search_code"), None)
    assert search_tool is not None, "search_code tool not found"

    properties = search_tool.inputSchema.get("properties", {})
    assert "mode" in properties, "search_code tool missing 'mode' parameter"

    mode_schema = properties["mode"]
    assert "enum" in mode_schema, "mode parameter should have enum values"
    assert set(mode_schema["enum"]) == {"hybrid", "vector", "grep"}


def test_index_codebase_requires_path():
    """Test that index_codebase tool requires path parameter."""
    tools = _run_async(list_tools())

    index_tool = next((t for t in tools if t.name == "index_codebase"), None)
    assert index_tool is not None, "index_codebase tool not found"

    required = index_tool.inputSchema.get("required", [])
    assert "path" in required, "index_codebase should require 'path' parameter"
