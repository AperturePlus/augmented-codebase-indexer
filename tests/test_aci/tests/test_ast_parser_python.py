"""
Property-based tests for AST Parser - Python language.

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

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
# Strategies for generating Python code
# =============================================================================

simple_body_line = st.sampled_from(["pass", "x = 1", "return None", "print('hello')"])
body_line_count = st.integers(min_value=1, max_value=3)


@st.composite
def python_function_code(draw):
    """Generate valid Python code containing function definitions."""
    num_functions = draw(st.integers(min_value=1, max_value=3))

    lines = []
    expected_functions = []
    current_line = 1

    for i in range(num_functions):
        if i > 0:
            num_blanks = draw(st.integers(min_value=1, max_value=2))
            lines.extend([""] * num_blanks)
            current_line += num_blanks

        name = f"func_{i}"
        start_line = current_line

        lines.append(f"def {name}():")
        current_line += 1

        num_body_lines = draw(body_line_count)
        for _ in range(num_body_lines):
            body = draw(simple_body_line)
            lines.append(f"    {body}")
            current_line += 1

        end_line = current_line - 1
        expected_functions.append((name, start_line, end_line))

    return "\n".join(lines), expected_functions


@st.composite
def python_class_code(draw):
    """Generate valid Python code containing class definitions."""
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

        lines.append(f"class {name}:")
        current_line += 1

        num_body_lines = draw(st.integers(min_value=1, max_value=3))
        for j in range(num_body_lines):
            if j == 0:
                lines.append("    pass")
            else:
                lines.append(f"    attr_{j} = {j}")
            current_line += 1

        end_line = current_line - 1
        expected_classes.append((name, start_line, end_line))

    return "\n".join(lines), expected_classes


# =============================================================================
# Property Tests
# =============================================================================


@given(data=python_function_code())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_python_function_line_accuracy(data):
    """
    **Feature: codebase-semantic-search, Property 4: Function Chunk Line Accuracy**
    **Validates: Requirements 2.2**

    For any valid Python code containing function definitions, AST_Parser extracts
    function chunks with accurate start_line and end_line.
    """
    code, expected_functions = data
    assume(len(expected_functions) > 0)

    parser = TreeSitterParser()
    nodes = parser.parse(code, "python")
    function_nodes = [n for n in nodes if n.node_type == "function"]

    for name, expected_start, expected_end in expected_functions:
        matching = [n for n in function_nodes if n.name == name]
        assert len(matching) == 1, f"Expected one function '{name}', found {len(matching)}"

        node = matching[0]
        assert node.start_line == expected_start
        assert node.end_line == expected_end

        extracted = get_lines_from_content(code, node.start_line, node.end_line)
        assert normalize_content(extracted) == normalize_content(node.content)


@given(data=python_class_code())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_python_class_line_accuracy(data):
    """
    **Feature: codebase-semantic-search, Property 5: Class Chunk Line Accuracy**
    **Validates: Requirements 2.3**

    For any valid Python code containing class definitions, AST_Parser extracts
    class chunks with accurate start_line and end_line.
    """
    code, expected_classes = data
    assume(len(expected_classes) > 0)

    parser = TreeSitterParser()
    nodes = parser.parse(code, "python")
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
# Complex Python File Parsing Tests
# =============================================================================


class TestComplexPythonParsing:
    """Tests for parsing complex Python code from fixture file."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    @pytest.fixture
    def complex_python_code(self):
        """Load the complex Python fixture file."""
        fixture_path = FIXTURES_DIR / "complex_python.py"
        return fixture_path.read_text(encoding="utf-8")

    def test_parses_decorated_functions(self, parser, complex_python_code):
        """Test that decorated functions are correctly parsed."""
        nodes = parser.parse(complex_python_code, "python")
        function_names = {n.name for n in nodes if n.node_type == "function"}

        assert "decorated_function" in function_names
        assert "function_with_parameterized_decorator" in function_names
        assert "multi_decorated_function" in function_names

    def test_parses_classes_with_inheritance(self, parser, complex_python_code):
        """Test that classes with inheritance are correctly parsed."""
        nodes = parser.parse(complex_python_code, "python")
        class_names = {n.name for n in nodes if n.node_type == "class"}

        assert "BaseClass" in class_names
        assert "DerivedClass" in class_names
        assert "MultipleInheritance" in class_names

    def test_parses_nested_classes(self, parser, complex_python_code):
        """Test that nested classes are correctly parsed."""
        nodes = parser.parse(complex_python_code, "python")
        class_names = {n.name for n in nodes if n.node_type == "class"}

        assert "OuterClass" in class_names
        assert "InnerClass" in class_names
        assert "DeeplyNestedClass" in class_names

    def test_parses_methods_with_decorators(self, parser, complex_python_code):
        """Test that methods with decorators (property, classmethod, etc.) are parsed."""
        nodes = parser.parse(complex_python_code, "python")

        # Find methods in BaseClass
        base_methods = [
            n for n in nodes if n.node_type == "method" and n.parent_name == "BaseClass"
        ]
        method_names = {m.name for m in base_methods}

        assert "__init__" in method_names
        assert "value" in method_names  # property getter
        assert "abstract_method" in method_names
        assert "class_method" in method_names
        assert "static_method" in method_names

    def test_parses_async_functions(self, parser, complex_python_code):
        """Test that async functions are correctly parsed."""
        nodes = parser.parse(complex_python_code, "python")
        function_names = {n.name for n in nodes if n.node_type == "function"}

        assert "async_function" in function_names
        assert "async_function_with_await" in function_names

    def test_parses_generator_functions(self, parser, complex_python_code):
        """Test that generator functions are correctly parsed."""
        nodes = parser.parse(complex_python_code, "python")
        function_names = {n.name for n in nodes if n.node_type == "function"}

        assert "generator_function" in function_names
        assert "generator_with_send" in function_names

    def test_parses_dataclasses(self, parser, complex_python_code):
        """Test that dataclasses are correctly parsed."""
        nodes = parser.parse(complex_python_code, "python")
        class_names = {n.name for n in nodes if n.node_type == "class"}

        assert "SimpleDataclass" in class_names
        assert "DataclassWithMethods" in class_names

    def test_parses_complex_function_signatures(self, parser, complex_python_code):
        """Test that functions with complex signatures are parsed."""
        nodes = parser.parse(complex_python_code, "python")
        function_names = {n.name for n in nodes if n.node_type == "function"}

        assert "function_with_complex_signature" in function_names
        assert "function_with_type_hints" in function_names

    def test_line_numbers_are_valid(self, parser, complex_python_code):
        """Test that all parsed nodes have valid line numbers."""
        nodes = parser.parse(complex_python_code, "python")
        lines = complex_python_code.split("\n")
        total_lines = len(lines)

        for node in nodes:
            assert node.start_line >= 1, f"Node {node.name} has invalid start_line"
            assert node.end_line <= total_lines, f"Node {node.name} has invalid end_line"
            assert node.start_line <= node.end_line, f"Node {node.name} has start > end"

    def test_content_matches_line_numbers(self, parser, complex_python_code):
        """Test that node content matches the lines indicated by line numbers."""
        nodes = parser.parse(complex_python_code, "python")

        for node in nodes:
            extracted = get_lines_from_content(complex_python_code, node.start_line, node.end_line)
            assert normalize_content(extracted) == normalize_content(node.content), (
                f"Content mismatch for {node.node_type} '{node.name}'"
            )
