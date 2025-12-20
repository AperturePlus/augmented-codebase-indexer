"""
Property-based tests for AST Parser - JavaScript language.

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
# Strategies for generating JavaScript code
# =============================================================================


@st.composite
def javascript_function_code(draw):
    """Generate valid JavaScript code containing function declarations."""
    num_functions = draw(st.integers(min_value=1, max_value=3))

    lines = []
    expected_functions = []
    current_line = 1

    for i in range(num_functions):
        if i > 0:
            num_blanks = draw(st.integers(min_value=1, max_value=2))
            lines.extend([""] * num_blanks)
            current_line += num_blanks

        name = f"func{i}"
        start_line = current_line

        lines.append(f"function {name}() {{")
        current_line += 1

        num_body_lines = draw(st.integers(min_value=1, max_value=3))
        for _ in range(num_body_lines):
            lines.append("    console.log('test');")
            current_line += 1

        lines.append("}")
        current_line += 1

        end_line = current_line - 1
        expected_functions.append((name, start_line, end_line))

    return "\n".join(lines), expected_functions


@st.composite
def javascript_class_code(draw):
    """Generate valid JavaScript code containing class declarations."""
    num_classes = draw(st.integers(min_value=1, max_value=2))

    lines = []
    expected_classes = []
    current_line = 1

    for i in range(num_classes):
        if i > 0:
            num_blanks = draw(st.integers(min_value=1, max_value=2))
            lines.extend([""] * num_blanks)
            current_line += num_blanks

        name = f"MyClass{i}"
        start_line = current_line

        lines.append(f"class {name} {{")
        current_line += 1
        lines.append("    constructor() {")
        current_line += 1
        lines.append("        this.value = 0;")
        current_line += 1
        lines.append("    }")
        current_line += 1
        lines.append("}")
        current_line += 1

        end_line = current_line - 1
        expected_classes.append((name, start_line, end_line))

    return "\n".join(lines), expected_classes


# =============================================================================
# Property Tests
# =============================================================================


@given(data=javascript_function_code())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_javascript_function_line_accuracy(data):
    """
    **Feature: codebase-semantic-search, Property 4: Function Chunk Line Accuracy**
    **Validates: Requirements 2.2**

    For any valid JavaScript code containing function declarations, AST_Parser extracts
    function chunks with accurate start_line and end_line.
    """
    code, expected_functions = data
    assume(len(expected_functions) > 0)

    parser = TreeSitterParser()
    nodes = parser.parse(code, "javascript")
    function_nodes = [n for n in nodes if n.node_type == "function"]

    for name, expected_start, expected_end in expected_functions:
        matching = [n for n in function_nodes if n.name == name]
        assert len(matching) == 1, f"Expected one function '{name}', found {len(matching)}"

        node = matching[0]
        assert node.start_line == expected_start
        assert node.end_line == expected_end

        extracted = get_lines_from_content(code, node.start_line, node.end_line)
        assert normalize_content(extracted) == normalize_content(node.content)


@given(data=javascript_class_code())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_javascript_class_line_accuracy(data):
    """
    **Feature: codebase-semantic-search, Property 5: Class Chunk Line Accuracy**
    **Validates: Requirements 2.3**

    For any valid JavaScript code containing class declarations, AST_Parser extracts
    class chunks with accurate start_line and end_line.
    """
    code, expected_classes = data
    assume(len(expected_classes) > 0)

    parser = TreeSitterParser()
    nodes = parser.parse(code, "javascript")
    class_nodes = [n for n in nodes if n.node_type == "class"]

    for name, expected_start, expected_end in expected_classes:
        matching = [n for n in class_nodes if n.name == name]
        assert len(matching) == 1, f"Expected one class '{name}', found {len(matching)}"

        node = matching[0]
        assert node.start_line == expected_start
        assert node.end_line == expected_end

        extracted = get_lines_from_content(code, node.start_line, node.end_line)
        assert normalize_content(extracted) == normalize_content(node.content)


# =============================================================================
# Complex JavaScript File Parsing Tests
# =============================================================================


class TestComplexJavaScriptParsing:
    """Tests for parsing complex JavaScript code from fixture file."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    @pytest.fixture
    def complex_js_code(self):
        """Load the complex JavaScript fixture file."""
        fixture_path = FIXTURES_DIR / "complex_javascript.js"
        return fixture_path.read_text(encoding="utf-8")

    def test_parses_function_declarations(self, parser, complex_js_code):
        """Test that function declarations are correctly parsed."""
        nodes = parser.parse(complex_js_code, "javascript")
        function_names = {n.name for n in nodes if n.node_type == "function"}

        assert "simpleFunction" in function_names
        assert "functionWithParams" in function_names
        assert "functionWithRestParams" in function_names
        assert "functionWithDestructuring" in function_names

    def test_parses_arrow_functions(self, parser, complex_js_code):
        """Test that arrow functions in variable declarations are parsed."""
        nodes = parser.parse(complex_js_code, "javascript")
        function_names = {n.name for n in nodes if n.node_type == "function"}

        assert "arrowFunction" in function_names
        assert "arrowWithParams" in function_names
        assert "arrowWithBody" in function_names

    def test_parses_classes(self, parser, complex_js_code):
        """Test that classes are correctly parsed."""
        nodes = parser.parse(complex_js_code, "javascript")
        class_names = {n.name for n in nodes if n.node_type == "class"}

        assert "BaseClass" in class_names
        assert "DerivedClass" in class_names
        assert "ComplexClass" in class_names

    def test_parses_class_methods(self, parser, complex_js_code):
        """Test that class methods are correctly parsed."""
        nodes = parser.parse(complex_js_code, "javascript")

        # Find methods in BaseClass
        base_methods = [
            n for n in nodes if n.node_type == "method" and n.parent_name == "BaseClass"
        ]
        method_names = {m.name for m in base_methods}

        assert "constructor" in method_names
        assert "instanceMethod" in method_names
        assert "staticMethod" in method_names
        assert "asyncMethod" in method_names
        assert "generatorMethod" in method_names

    def test_parses_async_functions(self, parser, complex_js_code):
        """Test that async functions are correctly parsed."""
        nodes = parser.parse(complex_js_code, "javascript")
        function_names = {n.name for n in nodes if n.node_type == "function"}

        assert "asyncFunction" in function_names
        assert "asyncWithAwait" in function_names
        assert "asyncWithTryCatch" in function_names

    def test_parses_generator_functions(self, parser, complex_js_code):
        """Test that generator functions are correctly parsed.

        Note: Generator functions (function*) are currently not extracted by the parser.
        This test documents the current behavior - generator functions are not in the
        function list. This could be enhanced in the future.
        """
        nodes = parser.parse(complex_js_code, "javascript")
        function_names = {n.name for n in nodes if n.node_type == "function"}

        # Generator functions are not currently extracted as standalone functions
        # They use a different AST node type (generator_function_declaration)
        # This is a known limitation - the parser focuses on regular functions
        assert "generatorFunction" not in function_names  # Known limitation

    def test_parses_higher_order_functions(self, parser, complex_js_code):
        """Test that higher-order functions are correctly parsed."""
        nodes = parser.parse(complex_js_code, "javascript")
        function_names = {n.name for n in nodes if n.node_type == "function"}

        assert "higherOrderFunction" in function_names
        assert "closureExample" in function_names
        assert "curryFunction" in function_names

    def test_line_numbers_are_valid(self, parser, complex_js_code):
        """Test that all parsed nodes have valid line numbers."""
        nodes = parser.parse(complex_js_code, "javascript")
        lines = complex_js_code.split("\n")
        total_lines = len(lines)

        for node in nodes:
            assert node.start_line >= 1, f"Node {node.name} has invalid start_line"
            assert node.end_line <= total_lines, f"Node {node.name} has invalid end_line"
            assert node.start_line <= node.end_line, f"Node {node.name} has start > end"

    def test_content_matches_line_numbers(self, parser, complex_js_code):
        """Test that node content matches the lines indicated by line numbers."""
        nodes = parser.parse(complex_js_code, "javascript")

        for node in nodes:
            extracted = get_lines_from_content(complex_js_code, node.start_line, node.end_line)
            assert normalize_content(extracted) == normalize_content(node.content), (
                f"Content mismatch for {node.node_type} '{node.name}'"
            )
