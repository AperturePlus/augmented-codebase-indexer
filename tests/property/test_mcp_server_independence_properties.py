"""
Property-based tests for MCP server independence from CLI.

**Feature: service-initialization-refactor, Property 2: MCP Server Independence from CLI**
**Validates: Requirements 1.2**
"""

import sys


def get_transitive_imports(module_name: str) -> set[str]:
    """
    Get all transitive imports for a module.

    This function imports the module and collects all modules
    that were loaded as a result.
    """
    aci_prefix = "aci"
    aci_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == aci_prefix or name.startswith(f"{aci_prefix}.")
    }
    for mod in aci_modules:
        del sys.modules[mod]

    # Record modules before import
    before_import = set(sys.modules.keys())

    try:
        # Import the module
        __import__(module_name)

        # Record modules after import
        after_import = set(sys.modules.keys())
        return after_import - before_import
    finally:
        loaded_aci_modules = [
            name
            for name in sys.modules.keys()
            if name == aci_prefix or name.startswith(f"{aci_prefix}.")
        ]
        for mod in loaded_aci_modules:
            del sys.modules[mod]
        sys.modules.update(aci_modules)


def test_mcp_services_does_not_import_cli():
    """
    **Feature: service-initialization-refactor, Property 2: MCP Server Independence from CLI**
    **Validates: Requirements 1.2**

    *For any* import of `aci.mcp.services`, the module's transitive imports
    SHALL NOT include `aci.cli`.

    This ensures the MCP server can be used without depending on CLI code.
    """
    # Get transitive imports of mcp.services
    transitive_imports = get_transitive_imports("aci.mcp.services")

    # Check that no aci.cli modules are imported
    cli_imports = [mod for mod in transitive_imports if mod.startswith("aci.cli")]

    assert len(cli_imports) == 0, (
        f"MCP services should not import CLI modules. "
        f"Found CLI imports: {cli_imports}"
    )


def test_mcp_services_imports_from_services_container():
    """
    **Feature: service-initialization-refactor, Property 2: MCP Server Independence from CLI**
    **Validates: Requirements 1.2**

    The MCP server SHALL import service creation from `aci.services.container`.
    """
    # Import mcp.services
    transitive_imports = get_transitive_imports("aci.mcp.services")

    # Verify services.container is imported
    assert "aci.services.container" in transitive_imports, (
        "MCP services should import from aci.services.container"
    )


def test_mcp_handlers_does_not_import_cli():
    """
    **Feature: service-initialization-refactor, Property 2: MCP Server Independence from CLI**
    **Validates: Requirements 1.2**

    *For any* import of `aci.mcp.handlers`, the module's transitive imports
    SHALL NOT include `aci.cli`.

    This ensures the MCP handlers can be used without depending on CLI code.
    """
    # Get transitive imports of mcp.handlers
    transitive_imports = get_transitive_imports("aci.mcp.handlers")

    # Check that no aci.cli modules are imported
    cli_imports = [mod for mod in transitive_imports if mod.startswith("aci.cli")]

    assert len(cli_imports) == 0, (
        f"MCP handlers should not import CLI modules. "
        f"Found CLI imports: {cli_imports}"
    )


def test_mcp_handlers_imports_repository_resolver():
    """
    **Feature: service-initialization-refactor, Property 2: MCP Server Independence from CLI**
    **Validates: Requirements 1.2**

    The MCP handlers SHALL import repository resolution from
    `aci.services.repository_resolver`.
    """
    # Import mcp.handlers
    transitive_imports = get_transitive_imports("aci.mcp.handlers")

    # Verify repository_resolver is imported
    assert "aci.services.repository_resolver" in transitive_imports, (
        "MCP handlers should import from aci.services.repository_resolver"
    )
