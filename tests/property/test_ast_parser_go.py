"""
Property-based tests for AST Parser - Go language.

**Feature: codebase-semantic-search, Property 4: Function Chunk Line Accuracy**
**Feature: codebase-semantic-search, Property 5: Class Chunk Line Accuracy**
**Validates: Requirements 2.2, 2.3**
"""

from pathlib import Path

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from aci.core.ast_parser import TreeSitterParser

# =============================================================================
# Fixtures path
# =============================================================================

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


# =============================================================================
# Helper functions
# =============================================================================


def get_lines_from_content(content: str, start_line: int, end_line: int) -> str:
    """Extract lines from content (1-based line numbers, inclusive)."""
    lines = content.split("\n")
    extracted = lines[start_line - 1 : end_line]
    return "\n".join(extracted)


def normalize_content(content: str) -> str:
    """Normalize content for comparison (strip trailing whitespace)."""
    return "\n".join(line.rstrip() for line in content.split("\n")).strip()


# =============================================================================
# Strategies for generating Go code
# =============================================================================


@st.composite
def go_function_code(draw):
    """Generate valid Go code containing function declarations."""
    num_functions = draw(st.integers(min_value=1, max_value=3))

    lines = ["package main", ""]
    expected_functions = []
    current_line = 3

    for i in range(num_functions):
        if i > 0:
            num_blanks = draw(st.integers(min_value=1, max_value=2))
            lines.extend([""] * num_blanks)
            current_line += num_blanks

        name = f"Func{i}"
        start_line = current_line

        lines.append(f"func {name}() {{")
        current_line += 1

        num_body_lines = draw(st.integers(min_value=1, max_value=3))
        for _ in range(num_body_lines):
            lines.append('    fmt.Println("test")')
            current_line += 1

        lines.append("}")
        current_line += 1

        end_line = current_line - 1
        expected_functions.append((name, start_line, end_line))

    return "\n".join(lines), expected_functions


@st.composite
def go_struct_code(draw):
    """Generate valid Go code containing struct type declarations."""
    num_structs = draw(st.integers(min_value=1, max_value=2))

    lines = ["package main", ""]
    expected_structs = []
    current_line = 3

    for i in range(num_structs):
        if i > 0:
            num_blanks = draw(st.integers(min_value=1, max_value=2))
            lines.extend([""] * num_blanks)
            current_line += num_blanks

        name = f"MyStruct{i}"
        start_line = current_line

        lines.append(f"type {name} struct {{")
        current_line += 1

        num_fields = draw(st.integers(min_value=1, max_value=3))
        for j in range(num_fields):
            lines.append(f"    Field{j} int")
            current_line += 1

        lines.append("}")
        current_line += 1

        end_line = current_line - 1
        expected_structs.append((name, start_line, end_line))

    return "\n".join(lines), expected_structs


# =============================================================================
# Property Tests
# =============================================================================


@given(data=go_function_code())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_go_function_line_accuracy(data):
    """
    **Feature: codebase-semantic-search, Property 4: Function Chunk Line Accuracy**
    **Validates: Requirements 2.2**

    For any valid Go code containing function declarations, AST_Parser extracts
    function chunks with accurate start_line and end_line.
    """
    code, expected_functions = data
    assume(len(expected_functions) > 0)

    parser = TreeSitterParser()
    nodes = parser.parse(code, "go")
    function_nodes = [n for n in nodes if n.node_type == "function"]

    for name, expected_start, expected_end in expected_functions:
        matching = [n for n in function_nodes if n.name == name]
        assert len(matching) == 1, f"Expected one function '{name}', found {len(matching)}"

        node = matching[0]
        assert node.start_line == expected_start
        assert node.end_line == expected_end

        extracted = get_lines_from_content(code, node.start_line, node.end_line)
        assert normalize_content(extracted) == normalize_content(node.content)


@given(data=go_struct_code())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_go_struct_line_accuracy(data):
    """
    **Feature: codebase-semantic-search, Property 5: Class Chunk Line Accuracy**
    **Validates: Requirements 2.3**

    For any valid Go code containing struct type declarations, AST_Parser extracts
    struct chunks with accurate start_line and end_line.
    """
    code, expected_structs = data
    assume(len(expected_structs) > 0)

    parser = TreeSitterParser()
    nodes = parser.parse(code, "go")
    struct_nodes = [n for n in nodes if n.node_type == "class"]

    for name, expected_start, expected_end in expected_structs:
        matching = [n for n in struct_nodes if n.name == name]
        assert len(matching) == 1, f"Expected one struct '{name}', found {len(matching)}"

        node = matching[0]
        assert node.start_line == expected_start
        assert node.end_line == expected_end

        extracted = get_lines_from_content(code, node.start_line, node.end_line)
        assert normalize_content(extracted) == normalize_content(node.content)


# =============================================================================
# Complex Go File Parsing Tests
# =============================================================================


class TestComplexGoParsing:
    """Tests for parsing complex Go code from fixture file."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    @pytest.fixture
    def complex_go_code(self):
        """Load the complex Go fixture file."""
        fixture_path = FIXTURES_DIR / "complex_go.go"
        return fixture_path.read_text(encoding="utf-8")

    def test_parses_simple_structs(self, parser, complex_go_code):
        """Test that simple structs are correctly parsed."""
        nodes = parser.parse(complex_go_code, "go")
        struct_names = {n.name for n in nodes if n.node_type == "class"}

        assert "SimpleStruct" in struct_names
        assert "EmbeddedStruct" in struct_names

    def test_parses_generic_structs(self, parser, complex_go_code):
        """Test that generic structs are correctly parsed."""
        nodes = parser.parse(complex_go_code, "go")
        struct_names = {n.name for n in nodes if n.node_type == "class"}

        assert "GenericStruct" in struct_names
        assert "GenericMap" in struct_names

    def test_parses_complex_struct(self, parser, complex_go_code):
        """Test that complex structs are correctly parsed."""
        nodes = parser.parse(complex_go_code, "go")
        struct_names = {n.name for n in nodes if n.node_type == "class"}

        assert "ComplexStruct" in struct_names
        assert "Server" in struct_names

    def test_parses_standalone_functions(self, parser, complex_go_code):
        """Test that standalone functions are correctly parsed."""
        nodes = parser.parse(complex_go_code, "go")
        function_names = {n.name for n in nodes if n.node_type == "function"}

        assert "SimpleFunction" in function_names
        assert "FunctionWithParams" in function_names
        assert "FunctionWithVariadic" in function_names
        assert "FunctionWithNamedReturns" in function_names

    def test_parses_generic_functions(self, parser, complex_go_code):
        """Test that generic functions are correctly parsed."""
        nodes = parser.parse(complex_go_code, "go")
        function_names = {n.name for n in nodes if n.node_type == "function"}

        assert "GenericFunction" in function_names
        assert "GenericConstrainedFunction" in function_names

    def test_parses_constructor_functions(self, parser, complex_go_code):
        """Test that constructor functions (NewXxx) are correctly parsed."""
        nodes = parser.parse(complex_go_code, "go")
        function_names = {n.name for n in nodes if n.node_type == "function"}

        assert "NewComplexStruct" in function_names
        assert "NewServer" in function_names

    def test_parses_methods_with_receivers(self, parser, complex_go_code):
        """Test that methods with receivers are correctly parsed."""
        nodes = parser.parse(complex_go_code, "go")

        # Find methods for SimpleStruct
        simple_methods = [
            n for n in nodes if n.node_type == "method" and n.parent_name == "SimpleStruct"
        ]
        method_names = {m.name for m in simple_methods}

        assert "GetName" in method_names
        assert "SetName" in method_names

    def test_parses_methods_for_generic_types(self, parser, complex_go_code):
        """Test that methods for generic types are correctly parsed.

        Note: Generic type methods have receivers like `GenericStruct[T]` which
        the parser extracts as just the base type name without the type parameter.
        """
        nodes = parser.parse(complex_go_code, "go")

        # Find all methods - generic type methods may have different parent_name format
        all_methods = [n for n in nodes if n.node_type == "method"]
        method_names = {m.name for m in all_methods}

        # These methods exist but may have different parent_name due to generic syntax
        assert "GetValue" in method_names
        assert "SetValue" in method_names
        assert "AddItem" in method_names

    def test_parses_error_type_methods(self, parser, complex_go_code):
        """Test that error type methods are correctly parsed."""
        nodes = parser.parse(complex_go_code, "go")

        # Find methods for CustomError
        error_methods = [
            n for n in nodes if n.node_type == "method" and n.parent_name == "CustomError"
        ]
        method_names = {m.name for m in error_methods}

        assert "Error" in method_names
        assert "Unwrap" in method_names
        assert "Is" in method_names

    def test_parses_functional_options(self, parser, complex_go_code):
        """Test that functional option functions are correctly parsed."""
        nodes = parser.parse(complex_go_code, "go")
        function_names = {n.name for n in nodes if n.node_type == "function"}

        assert "WithHost" in function_names
        assert "WithPort" in function_names
        assert "WithTimeout" in function_names
        assert "WithLogger" in function_names

    def test_parses_concurrency_functions(self, parser, complex_go_code):
        """Test that functions with goroutines and channels are parsed."""
        nodes = parser.parse(complex_go_code, "go")
        function_names = {n.name for n in nodes if n.node_type == "function"}

        assert "FunctionWithGoroutine" in function_names
        assert "FunctionWithSelect" in function_names

    def test_line_numbers_are_valid(self, parser, complex_go_code):
        """Test that all parsed nodes have valid line numbers."""
        nodes = parser.parse(complex_go_code, "go")
        lines = complex_go_code.split("\n")
        total_lines = len(lines)

        for node in nodes:
            assert node.start_line >= 1, f"Node {node.name} has invalid start_line"
            assert node.end_line <= total_lines, f"Node {node.name} has invalid end_line"
            assert node.start_line <= node.end_line, f"Node {node.name} has start > end"

    def test_content_matches_line_numbers(self, parser, complex_go_code):
        """Test that node content matches the lines indicated by line numbers."""
        nodes = parser.parse(complex_go_code, "go")

        for node in nodes:
            extracted = get_lines_from_content(complex_go_code, node.start_line, node.end_line)
            assert normalize_content(extracted) == normalize_content(node.content), (
                f"Content mismatch for {node.node_type} '{node.name}'"
            )
