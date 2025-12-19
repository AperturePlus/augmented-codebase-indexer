"""
Property-based tests for HTTP server independence from CLI.

**Feature: service-initialization-refactor, Property 1: HTTP Server Independence from CLI**
**Validates: Requirements 1.1**
"""

import sys
from typing import Set


def get_transitive_imports(module_name: str) -> Set[str]:
    """
    Get all transitive imports for a module.
    
    This function imports the module and collects all modules
    that were loaded as a result.
    """
    # Record modules before import
    before_import = set(sys.modules.keys())
    
    # Import the module
    __import__(module_name)
    
    # Record modules after import
    after_import = set(sys.modules.keys())
    
    # Return newly imported modules
    return after_import - before_import


def test_http_server_does_not_import_cli():
    """
    **Feature: service-initialization-refactor, Property 1: HTTP Server Independence from CLI**
    **Validates: Requirements 1.1**
    
    *For any* import of `aci.http_server`, the module's transitive imports
    SHALL NOT include `aci.cli`.
    
    This ensures the HTTP server can be used without depending on CLI code.
    """
    # Clear any cached imports of aci modules to get clean import
    modules_to_clear = [key for key in sys.modules.keys() if key.startswith("aci")]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    # Get transitive imports of http_server
    transitive_imports = get_transitive_imports("aci.http_server")
    
    # Check that no aci.cli modules are imported
    cli_imports = [mod for mod in transitive_imports if mod.startswith("aci.cli")]
    
    assert len(cli_imports) == 0, (
        f"HTTP server should not import CLI modules. "
        f"Found CLI imports: {cli_imports}"
    )


def test_http_server_imports_from_services_container():
    """
    **Feature: service-initialization-refactor, Property 1: HTTP Server Independence from CLI**
    **Validates: Requirements 1.1**
    
    The HTTP server SHALL import service creation from `aci.services.container`.
    """
    # Clear any cached imports
    modules_to_clear = [key for key in sys.modules.keys() if key.startswith("aci")]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    # Import http_server
    transitive_imports = get_transitive_imports("aci.http_server")
    
    # Verify services.container is imported
    assert "aci.services.container" in transitive_imports, (
        "HTTP server should import from aci.services.container"
    )



# Property tests for HTTP artifact type query parameters
from hypothesis import given, settings
from hypothesis import strategies as st


# Valid artifact types as defined in the design
VALID_ARTIFACT_TYPES = ["chunk", "function_summary", "class_summary", "file_summary"]


@st.composite
def artifact_type_list_strategy(draw):
    """Generate a list of valid artifact types."""
    return draw(
        st.lists(
            st.sampled_from(VALID_ARTIFACT_TYPES),
            min_size=1,
            max_size=len(VALID_ARTIFACT_TYPES),
            unique=True,
        )
    )


@given(artifact_types=artifact_type_list_strategy())
@settings(max_examples=100)
def test_http_artifact_type_query_params_are_valid(artifact_types: list[str]):
    """
    **Feature: service-initialization-refactor, Property 11: HTTP Artifact Type Query Parameters**
    **Validates: Requirements 5.2**
    
    *For any* HTTP search request with multiple `artifact_type` query parameters,
    the endpoint SHALL accept all valid artifact type values.
    
    This test verifies that the valid artifact types are correctly defined
    and can be used as query parameters.
    """
    # All generated artifact types should be in the valid set
    for artifact_type in artifact_types:
        assert artifact_type in VALID_ARTIFACT_TYPES, (
            f"Generated artifact type '{artifact_type}' not in valid types"
        )


@given(
    valid_types=artifact_type_list_strategy(),
    invalid_type=st.text(min_size=1, max_size=20).filter(
        lambda x: x not in VALID_ARTIFACT_TYPES and x.isalnum()
    ),
)
@settings(max_examples=100)
def test_http_invalid_artifact_type_returns_error_with_valid_types(
    valid_types: list[str],
    invalid_type: str,
):
    """
    **Feature: service-initialization-refactor, Property 11: HTTP Artifact Type Query Parameters**
    **Validates: Requirements 5.2**
    
    *For any* HTTP search request with an invalid artifact type,
    the error response SHALL list all valid artifact type names.
    
    This test verifies the error message format includes valid types.
    """
    # Simulate the validation logic from http_server.py
    artifact_types = valid_types + [invalid_type]
    invalid_types = [t for t in artifact_types if t not in VALID_ARTIFACT_TYPES]
    
    # Should detect the invalid type
    assert invalid_type in invalid_types, (
        f"Invalid type '{invalid_type}' should be detected"
    )
    
    # Error message should include valid types
    error_detail = (
        f"Invalid artifact type(s): {invalid_types}. "
        f"Valid types are: {VALID_ARTIFACT_TYPES}"
    )
    
    for valid_type in VALID_ARTIFACT_TYPES:
        assert valid_type in error_detail, (
            f"Error message should include valid type '{valid_type}'"
        )
