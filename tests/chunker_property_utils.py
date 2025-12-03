"""Shared strategies and helpers for chunker property-based tests."""

from pathlib import Path

from hypothesis import strategies as st

from aci.core.file_scanner import ScannedFile

# Strategy for generating valid Python identifiers
python_identifier = st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True)

# Strategy for generating simple Python statements
simple_statement = st.sampled_from(
    [
        "x = 1",
        "y = 2",
        "result = x + y",
        "print(result)",
        "return result",
        "pass",
        "continue",
        "break",
    ]
)

# Strategy for generating indented statements
indented_statement = st.builds(
    lambda indent, stmt: " " * indent + stmt,
    indent=st.sampled_from([4, 8]),
    stmt=simple_statement,
)


@st.composite
def python_function_strategy(draw):
    """Generate a valid Python function with random body."""
    func_name = draw(python_identifier)
    num_body_lines = draw(st.integers(min_value=1, max_value=10))

    body_lines = []
    for _ in range(num_body_lines):
        stmt = draw(simple_statement)
        body_lines.append(f"    {stmt}")

    lines = [f"def {func_name}():"]
    lines.extend(body_lines)

    content = "\n".join(lines)
    return content, func_name, len(lines)


@st.composite
def python_class_with_methods_strategy(draw):
    """Generate a valid Python class with methods."""
    class_name = draw(st.from_regex(r"[A-Z][a-zA-Z0-9]{0,15}", fullmatch=True))
    num_methods = draw(st.integers(min_value=1, max_value=3))

    lines = [f"class {class_name}:"]
    method_names = []

    for i in range(num_methods):
        method_name = f"method_{i}"
        method_names.append(method_name)

        lines.append(f"    def {method_name}(self):")
        num_body_lines = draw(st.integers(min_value=1, max_value=3))
        for _ in range(num_body_lines):
            stmt = draw(simple_statement)
            lines.append(f"        {stmt}")
        lines.append("")

    content = "\n".join(lines)
    return content, class_name, method_names


@st.composite
def multi_line_text_strategy(draw):
    """Generate multi-line text content for fixed-size chunking tests."""
    num_lines = draw(st.integers(min_value=10, max_value=200))
    lines = []

    for i in range(num_lines):
        line_type = draw(st.sampled_from(["short", "medium", "long", "empty"]))
        if line_type == "short":
            lines.append(f"line_{i}")
        elif line_type == "medium":
            lines.append(f"# This is line number {i} with some content")
        elif line_type == "long":
            lines.append(f"# Line {i}: " + "x" * draw(st.integers(min_value=20, max_value=80)))
        else:
            lines.append("")

    content = "\n".join(lines)
    return content, num_lines


@st.composite
def large_function_strategy(draw):
    """Generate a large Python function that may exceed token limits."""
    func_name = draw(python_identifier)
    num_body_lines = draw(st.integers(min_value=50, max_value=150))

    lines = [f"def {func_name}():"]

    for i in range(num_body_lines):
        stmt_type = draw(st.sampled_from(["assign", "comment", "empty", "call"]))
        if stmt_type == "assign":
            lines.append(f"    var_{i} = {i}")
        elif stmt_type == "comment":
            lines.append(f"    # Comment line {i}")
        elif stmt_type == "empty":
            lines.append("")
        else:
            lines.append(f"    print(var_{max(0, i - 1)})")

    lines.append("    return None")

    content = "\n".join(lines)
    return content, func_name


def create_scanned_file(
    content: str,
    language: str = "python",
    path: str = "/test/file.py",
) -> ScannedFile:
    """Helper to create a ScannedFile for testing."""
    return ScannedFile(
        path=Path(path),
        content=content,
        language=language,
        size_bytes=len(content),
        modified_time=0.0,
        content_hash="test-hash-" + str(hash(content))[:8],
    )
