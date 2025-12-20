"""
Tests for ACI MCP server.

Tests the MCP server's tool listing and basic functionality.
"""

from aci.mcp_server import list_tools


def test_list_tools_returns_expected_tools():
    """Test that all expected tools are available."""
    tools = list_tools()

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
    tools = list_tools()

    for tool in tools:
        assert tool.description, f"Tool {tool.name} has no description"
        assert len(tool.description) > 10, f"Tool {tool.name} has too short description"


def test_list_tools_have_input_schemas():
    """Test that all tools have input schemas."""
    tools = list_tools()

    for tool in tools:
        assert tool.inputSchema, f"Tool {tool.name} has no input schema"
        assert "type" in tool.inputSchema, f"Tool {tool.name} schema missing 'type'"
        assert tool.inputSchema["type"] == "object"


def test_search_code_tool_has_mode_parameter():
    """Test that search_code tool has the mode parameter for hybrid search."""
    tools = list_tools()

    search_tool = next((t for t in tools if t.name == "search_code"), None)
    assert search_tool is not None, "search_code tool not found"

    properties = search_tool.inputSchema.get("properties", {})
    assert "mode" in properties, "search_code tool missing 'mode' parameter"

    mode_schema = properties["mode"]
    assert "enum" in mode_schema, "mode parameter should have enum values"
    assert set(mode_schema["enum"]) == {"hybrid", "vector", "grep"}


def test_index_codebase_requires_path():
    """Test that index_codebase tool requires path parameter."""
    tools = list_tools()

    index_tool = next((t for t in tools if t.name == "index_codebase"), None)
    assert index_tool is not None, "index_codebase tool not found"

    required = index_tool.inputSchema.get("required", [])
    assert "path" in required, "index_codebase should require 'path' parameter"


# API Compatibility Tests for MCP Server Refactoring
# These tests verify that the refactored module structure maintains backward compatibility


def test_mcp_package_exports():
    """Test that the mcp package exports the expected public API."""
    from aci import mcp

    # Verify DI-based exports are available
    assert hasattr(mcp, "MCPContext")
    assert hasattr(mcp, "create_mcp_context")
    assert hasattr(mcp, "cleanup_context")

    # Verify core exports
    assert hasattr(mcp, "list_tools")
    assert hasattr(mcp, "call_tool")

    # Verify backward compatibility alias
    assert hasattr(mcp, "cleanup_services")


def test_mcp_server_module_exports():
    """Test that mcp_server.py maintains backward compatible exports."""
    from aci import mcp_server

    # Verify the app is exported
    assert hasattr(mcp_server, "app")

    # Verify list_tools is exported
    assert hasattr(mcp_server, "list_tools")

    # Verify call_tool is exported
    assert hasattr(mcp_server, "call_tool")

    # Verify main is exported
    assert hasattr(mcp_server, "main")


def test_tool_schemas_unchanged():
    """Test that tool schemas match the expected structure after refactoring."""
    tools = list_tools()

    # Build a dict of tool schemas for easy comparison
    tool_schemas = {tool.name: tool.inputSchema for tool in tools}

    # Verify index_codebase schema
    assert "path" in tool_schemas["index_codebase"]["properties"]
    assert "workers" in tool_schemas["index_codebase"]["properties"]
    assert tool_schemas["index_codebase"]["required"] == ["path"]

    # Verify search_code schema
    search_props = tool_schemas["search_code"]["properties"]
    assert "query" in search_props
    assert "path" in search_props
    assert "limit" in search_props
    assert "file_filter" in search_props
    assert "use_rerank" in search_props
    assert "mode" in search_props
    assert set(tool_schemas["search_code"]["required"]) == {"query", "path"}

    # Verify update_index schema
    assert "path" in tool_schemas["update_index"]["properties"]
    assert tool_schemas["update_index"]["required"] == ["path"]

    # Verify get_index_status schema (optional path param for per-repo stats)
    assert "path" in tool_schemas["get_index_status"]["properties"]
    assert "required" not in tool_schemas["get_index_status"]  # path is optional

    # Verify list_indexed_repos schema (no required params)
    assert tool_schemas["list_indexed_repos"]["properties"] == {}


def test_tool_count_unchanged():
    """Test that the number of tools remains the same after refactoring."""
    tools = list_tools()

    # Should have exactly 5 tools
    assert len(tools) == 5


def test_services_module_exports():
    """Test that services module exports the expected constants."""
    from aci.mcp import services

    # Service Locator functions have been removed
    # Only MAX_WORKERS constant remains
    assert hasattr(services, "MAX_WORKERS")
    assert services.MAX_WORKERS == 32

    # Verify deprecated functions are removed
    assert not hasattr(services, "get_initialized_services")
    assert not hasattr(services, "cleanup_services")
    assert not hasattr(services, "get_indexing_lock")


def test_handlers_module_exports():
    """Test that handlers module exports the expected functions."""
    from aci.mcp import handlers

    assert hasattr(handlers, "call_tool")


def test_tools_module_exports():
    """Test that tools module exports the expected functions."""
    from aci.mcp import tools

    assert hasattr(tools, "list_tools")
