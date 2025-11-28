"""
Property-based tests for Chunker module.

**Feature: codebase-semantic-search, Property 2: Line Number Accuracy Invariant**
**Feature: codebase-semantic-search, Property 6: Fixed-Size Chunk Bounds**
**Feature: codebase-semantic-search, Property 7: Token Limit Compliance**
**Feature: codebase-semantic-search, Property 7a: Smart Split Syntax Preservation**
**Feature: codebase-semantic-search, Property 8: Metadata Completeness**
**Validates: Requirements 1.3, 2.4, 2.5, 2.6**
"""

from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aci.core.ast_parser import ASTNode, TreeSitterParser
from aci.core.chunker import (
    Chunker,
    SmartChunkSplitter,
)
from aci.core.file_scanner import ScannedFile
from aci.core.tokenizer import get_default_tokenizer

# =============================================================================
# Strategies for generating test data
# =============================================================================

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
    """
    Generate a valid Python function with random body.

    Returns:
        tuple: (function_content, function_name, num_lines)
    """
    func_name = draw(python_identifier)
    num_body_lines = draw(st.integers(min_value=1, max_value=10))

    # Generate function body
    body_lines = []
    for _ in range(num_body_lines):
        stmt = draw(simple_statement)
        body_lines.append(f"    {stmt}")

    # Build function
    lines = [f"def {func_name}():"]
    lines.extend(body_lines)

    content = "\n".join(lines)
    return content, func_name, len(lines)


@st.composite
def python_class_with_methods_strategy(draw):
    """
    Generate a valid Python class with methods.

    Returns:
        tuple: (class_content, class_name, method_names)
    """
    class_name = draw(st.from_regex(r"[A-Z][a-zA-Z0-9]{0,15}", fullmatch=True))
    num_methods = draw(st.integers(min_value=1, max_value=3))

    lines = [f"class {class_name}:"]
    method_names = []

    for i in range(num_methods):
        method_name = f"method_{i}"
        method_names.append(method_name)

        # Add method with body
        lines.append(f"    def {method_name}(self):")
        num_body_lines = draw(st.integers(min_value=1, max_value=3))
        for _ in range(num_body_lines):
            stmt = draw(simple_statement)
            lines.append(f"        {stmt}")
        lines.append("")  # Empty line between methods

    content = "\n".join(lines)
    return content, class_name, method_names


@st.composite
def multi_line_text_strategy(draw):
    """
    Generate multi-line text content for fixed-size chunking tests.

    Returns:
        tuple: (content, num_lines)
    """
    num_lines = draw(st.integers(min_value=10, max_value=200))
    lines = []

    for i in range(num_lines):
        # Generate lines of varying lengths
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
    """
    Generate a large Python function that may exceed token limits.

    Returns:
        tuple: (function_content, function_name)
    """
    func_name = draw(python_identifier)
    num_body_lines = draw(st.integers(min_value=50, max_value=150))

    lines = [f"def {func_name}():"]

    for i in range(num_body_lines):
        # Mix of different statement types
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


# =============================================================================
# Property 2: Line Number Accuracy Invariant
# =============================================================================


@given(func_data=python_function_strategy())
@settings(max_examples=100)
def test_line_number_accuracy_ast_chunks(func_data):
    """
    **Feature: codebase-semantic-search, Property 2: Line Number Accuracy Invariant**
    **Validates: Requirements 1.3**

    For any Chunker-produced CodeChunk, extracting [start_line, end_line] from
    the original file content should match the chunk's content field (ignoring
    trailing whitespace).
    """
    content, func_name, _ = func_data

    tokenizer = get_default_tokenizer()
    chunker = Chunker(tokenizer=tokenizer, max_tokens=8192)
    parser = TreeSitterParser()

    file = create_scanned_file(content)
    ast_nodes = parser.parse(content, "python")

    # Skip if no AST nodes were extracted
    assume(len(ast_nodes) > 0)

    chunks = chunker.chunk(file, ast_nodes)

    # Property: For each chunk, extracted lines should match content
    lines = content.split("\n")
    for chunk in chunks:
        # Extract lines from original content (1-based to 0-based conversion)
        extracted_lines = lines[chunk.start_line - 1 : chunk.end_line]
        extracted_content = "\n".join(extracted_lines)

        # Compare (strip trailing whitespace for robustness)
        # Note: For partial chunks with context prefix, we need to handle differently
        if chunk.metadata.get("has_context_prefix"):
            # Skip context prefix comparison for continuation chunks
            continue

        assert chunk.content.rstrip() == extracted_content.rstrip(), (
            f"Line number mismatch for chunk {chunk.chunk_id}:\n"
            f"Expected (lines {chunk.start_line}-{chunk.end_line}):\n{extracted_content}\n"
            f"Got:\n{chunk.content}"
        )


@given(text_data=multi_line_text_strategy())
@settings(max_examples=100)
def test_line_number_accuracy_fixed_chunks(text_data):
    """
    **Feature: codebase-semantic-search, Property 2: Line Number Accuracy Invariant**
    **Validates: Requirements 1.3**

    For fixed-size chunks, the line numbers should accurately reflect the
    content extracted from the original file.
    """
    content, num_lines = text_data
    assume(num_lines > 0)

    tokenizer = get_default_tokenizer()
    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=8192,
        fixed_chunk_lines=50,
        overlap_lines=5,
    )

    # Use unknown language to force fixed-size chunking
    file = create_scanned_file(content, language="unknown", path="/test/file.txt")

    chunks = chunker.chunk(file, [])  # No AST nodes

    # Property: For each chunk, extracted lines should match content
    lines = content.split("\n")
    for chunk in chunks:
        # Extract lines from original content
        extracted_lines = lines[chunk.start_line - 1 : chunk.end_line]
        extracted_content = "\n".join(extracted_lines)

        assert chunk.content.rstrip() == extracted_content.rstrip(), (
            f"Line number mismatch for fixed chunk:\n"
            f"Expected (lines {chunk.start_line}-{chunk.end_line}):\n{extracted_content}\n"
            f"Got:\n{chunk.content}"
        )


# =============================================================================
# Property 6: Fixed-Size Chunk Bounds
# =============================================================================


@given(
    text_data=multi_line_text_strategy(),
    chunk_lines=st.integers(min_value=10, max_value=100),
    overlap_lines=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=100)
def test_fixed_size_chunk_bounds(text_data, chunk_lines, overlap_lines):
    """
    **Feature: codebase-semantic-search, Property 6: Fixed-Size Chunk Bounds**
    **Validates: Requirements 2.4**

    For unsupported languages, fixed-size chunks should:
    1. Not exceed the configured maximum line count
    2. Have the configured overlap between consecutive chunks
    """
    content, num_lines = text_data
    assume(num_lines > 0)
    assume(overlap_lines < chunk_lines)  # Overlap must be less than chunk size

    tokenizer = get_default_tokenizer()
    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=8192,
        fixed_chunk_lines=chunk_lines,
        overlap_lines=overlap_lines,
    )

    file = create_scanned_file(content, language="unknown", path="/test/file.txt")
    chunks = chunker.chunk(file, [])

    assume(len(chunks) > 0)

    # Property 1: Each chunk should not exceed max lines
    for chunk in chunks:
        chunk_line_count = chunk.end_line - chunk.start_line + 1
        assert chunk_line_count <= chunk_lines, (
            f"Chunk exceeds max lines: {chunk_line_count} > {chunk_lines}"
        )

    # Property 2: Consecutive chunks should have proper overlap
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            # Calculate actual overlap
            actual_overlap = current_chunk.end_line - next_chunk.start_line + 1

            # Overlap should be approximately equal to configured overlap
            # (may be less at boundaries or when token limits apply)
            assert actual_overlap <= overlap_lines + 1, (
                f"Overlap between chunks {i} and {i + 1} is {actual_overlap}, "
                f"expected <= {overlap_lines + 1}"
            )


# =============================================================================
# Property 7: Token Limit Compliance
# =============================================================================


@given(func_data=large_function_strategy())
@settings(max_examples=100)
def test_token_limit_compliance_ast_chunks(func_data):
    """
    **Feature: codebase-semantic-search, Property 7: Token Limit Compliance**
    **Validates: Requirements 2.5**

    For any Chunker-produced CodeChunk, the token count should not exceed
    the configured Token_Window limit.
    """
    content, func_name = func_data

    tokenizer = get_default_tokenizer()
    max_tokens = 500  # Small limit to force splitting

    chunker = Chunker(tokenizer=tokenizer, max_tokens=max_tokens)
    parser = TreeSitterParser()

    file = create_scanned_file(content)
    ast_nodes = parser.parse(content, "python")

    assume(len(ast_nodes) > 0)

    chunks = chunker.chunk(file, ast_nodes)

    # Property: All chunks should be within token limit
    for chunk in chunks:
        token_count = tokenizer.count_tokens(chunk.content)
        assert token_count <= max_tokens, (
            f"Chunk exceeds token limit: {token_count} > {max_tokens}\n"
            f"Chunk content:\n{chunk.content[:200]}..."
        )


@given(text_data=multi_line_text_strategy())
@settings(max_examples=100)
def test_token_limit_compliance_fixed_chunks(text_data):
    """
    **Feature: codebase-semantic-search, Property 7: Token Limit Compliance**
    **Validates: Requirements 2.5**

    For fixed-size chunks, the token count should not exceed the configured limit.
    """
    content, num_lines = text_data
    assume(num_lines > 0)

    tokenizer = get_default_tokenizer()
    max_tokens = 500  # Small limit

    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        fixed_chunk_lines=50,
        overlap_lines=5,
    )

    file = create_scanned_file(content, language="unknown", path="/test/file.txt")
    chunks = chunker.chunk(file, [])

    # Property: All chunks should be within token limit
    for chunk in chunks:
        token_count = tokenizer.count_tokens(chunk.content)
        assert token_count <= max_tokens, (
            f"Fixed chunk exceeds token limit: {token_count} > {max_tokens}"
        )


# =============================================================================
# Property 7a: Smart Split Syntax Preservation
# =============================================================================


@given(func_data=large_function_strategy())
@settings(max_examples=100)
def test_smart_split_syntax_preservation(func_data):
    """
    **Feature: codebase-semantic-search, Property 7a: Smart Split Syntax Preservation**
    **Validates: Requirements 2.5**

    For oversized code units, SmartChunkSplitter should split at syntax boundaries
    (empty lines, statement boundaries, or complete lines) rather than mid-statement.
    """
    content, func_name = func_data

    tokenizer = get_default_tokenizer()
    splitter = SmartChunkSplitter(tokenizer)

    # Create an AST node for the function
    lines = content.split("\n")
    node = ASTNode(
        node_type="function",
        name=func_name,
        start_line=1,
        end_line=len(lines),
        content=content,
    )

    # Use small token limit to force splitting
    max_tokens = 100

    chunks = splitter.split_oversized_node(
        node=node,
        max_tokens=max_tokens,
        file_path="/test/file.py",
        language="python",
        base_metadata={},
    )

    # Property: Each chunk should end at a complete line
    # (no partial lines or mid-statement cuts)
    for chunk in chunks:
        chunk_content = chunk.content

        # Remove context prefix if present
        if chunk.metadata.get("has_context_prefix"):
            # Find the end of the context prefix line
            first_newline = chunk_content.find("\n")
            if first_newline > 0:
                chunk_content = chunk_content[first_newline + 1 :]

        # Check that content doesn't end mid-line (unless it's the last line)
        # A proper split should end with a complete line
        if chunk_content:
            # The content should be composed of complete lines
            # (each line should be a valid Python statement or empty)
            chunk_lines = chunk_content.split("\n")
            for line in chunk_lines:
                # Lines should not be truncated mid-word
                # (simple heuristic: no line should end with incomplete syntax)
                stripped = line.rstrip()
                if stripped:
                    # Should not end with incomplete operators or keywords
                    assert not stripped.endswith(("=", "+", "-", "*", "/", "and", "or", "not")), (
                        f"Line appears to be truncated mid-statement: '{stripped}'"
                    )


# =============================================================================
# Property 8: Metadata Completeness
# =============================================================================


@given(class_data=python_class_with_methods_strategy())
@settings(max_examples=100)
def test_metadata_completeness_methods(class_data):
    """
    **Feature: codebase-semantic-search, Property 8: Metadata Completeness**
    **Validates: Requirements 2.6**

    For any CodeChunk extracted from a class method, the metadata should
    include the parent_class field.
    """
    content, class_name, method_names = class_data

    tokenizer = get_default_tokenizer()
    chunker = Chunker(tokenizer=tokenizer, max_tokens=8192)
    parser = TreeSitterParser()

    file = create_scanned_file(content)
    ast_nodes = parser.parse(content, "python")

    chunks = chunker.chunk(file, ast_nodes)

    # Find method chunks
    method_chunks = [c for c in chunks if c.chunk_type == "method"]

    # Property: All method chunks should have parent_class metadata
    for chunk in method_chunks:
        assert "parent_class" in chunk.metadata, (
            f"Method chunk missing parent_class metadata: {chunk.metadata}"
        )
        assert chunk.metadata["parent_class"] == class_name, (
            f"Method chunk has wrong parent_class: "
            f"expected {class_name}, got {chunk.metadata['parent_class']}"
        )

        # Also verify function_name is present
        assert "function_name" in chunk.metadata, (
            f"Method chunk missing function_name metadata: {chunk.metadata}"
        )


@given(func_data=python_function_strategy())
@settings(max_examples=100)
def test_metadata_completeness_functions(func_data):
    """
    **Feature: codebase-semantic-search, Property 8: Metadata Completeness**
    **Validates: Requirements 2.6**

    For any CodeChunk extracted from a function, the metadata should
    include the function_name field.
    """
    content, func_name, _ = func_data

    tokenizer = get_default_tokenizer()
    chunker = Chunker(tokenizer=tokenizer, max_tokens=8192)
    parser = TreeSitterParser()

    file = create_scanned_file(content)
    ast_nodes = parser.parse(content, "python")

    assume(len(ast_nodes) > 0)

    chunks = chunker.chunk(file, ast_nodes)

    # Find function chunks
    function_chunks = [c for c in chunks if c.chunk_type == "function"]

    # Property: All function chunks should have function_name metadata
    for chunk in function_chunks:
        assert "function_name" in chunk.metadata, (
            f"Function chunk missing function_name metadata: {chunk.metadata}"
        )
        assert chunk.metadata["function_name"] == func_name, (
            f"Function chunk has wrong function_name: "
            f"expected {func_name}, got {chunk.metadata['function_name']}"
        )


@given(func_data=python_function_strategy())
@settings(max_examples=100)
def test_metadata_includes_file_hash(func_data):
    """
    **Feature: codebase-semantic-search, Property 8: Metadata Completeness**
    **Validates: Requirements 2.6**

    All chunks should include the file_hash in their metadata.
    """
    content, _, _ = func_data

    tokenizer = get_default_tokenizer()
    chunker = Chunker(tokenizer=tokenizer, max_tokens=8192)
    parser = TreeSitterParser()

    file = create_scanned_file(content)
    ast_nodes = parser.parse(content, "python")

    chunks = chunker.chunk(file, ast_nodes)

    # Property: All chunks should have file_hash metadata
    for chunk in chunks:
        assert "file_hash" in chunk.metadata, f"Chunk missing file_hash metadata: {chunk.metadata}"
        assert chunk.metadata["file_hash"] == file.content_hash, (
            f"Chunk has wrong file_hash: "
            f"expected {file.content_hash}, got {chunk.metadata['file_hash']}"
        )
