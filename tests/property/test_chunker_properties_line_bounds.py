"""Property-based tests for chunker line accuracy and fixed-size bounds."""

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from aci.core.ast_parser import TreeSitterParser
from aci.core.chunker import Chunker
from aci.core.tokenizer import get_default_tokenizer
from tests.support.chunker_property_utils import (
    create_scanned_file,
    multi_line_text_strategy,
    python_function_strategy,
)


@given(func_data=python_function_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_line_number_accuracy_ast_chunks(func_data):
    """
    **Feature: codebase-semantic-search, Property 2: Line Number Accuracy Invariant**
    **Validates: Requirements 1.3**

    For any Chunker-produced CodeChunk, extracting [start_line, end_line] from
    the original file content should match the chunk's content field.
    """
    content, func_name, _ = func_data

    tokenizer = get_default_tokenizer()
    chunker = Chunker(tokenizer=tokenizer, max_tokens=8192)
    parser = TreeSitterParser()

    file = create_scanned_file(content)
    ast_nodes = parser.parse(content, "python")

    assume(len(ast_nodes) > 0)

    result = chunker.chunk(file, ast_nodes)
    chunks = result.chunks

    lines = content.split("\n")
    for chunk in chunks:
        extracted_lines = lines[chunk.start_line - 1 : chunk.end_line]
        extracted_content = "\n".join(extracted_lines)

        if chunk.metadata.get("has_context_prefix"):
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

    file = create_scanned_file(content, language="unknown", path="/test/file.txt")

    result = chunker.chunk(file, [])
    chunks = result.chunks

    lines = content.split("\n")
    for chunk in chunks:
        extracted_lines = lines[chunk.start_line - 1 : chunk.end_line]
        extracted_content = "\n".join(extracted_lines)

        assert chunk.content.rstrip() == extracted_content.rstrip(), (
            f"Line number mismatch for fixed chunk:\n"
            f"Expected (lines {chunk.start_line}-{chunk.end_line}):\n{extracted_content}\n"
            f"Got:\n{chunk.content}"
        )


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

    For unsupported languages, fixed-size chunks should not exceed the
    configured line count and should respect configured overlap.
    """
    content, num_lines = text_data
    assume(num_lines > 0)
    assume(overlap_lines < chunk_lines)

    tokenizer = get_default_tokenizer()
    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=8192,
        fixed_chunk_lines=chunk_lines,
        overlap_lines=overlap_lines,
    )

    file = create_scanned_file(content, language="unknown", path="/test/file.txt")
    result = chunker.chunk(file, [])
    chunks = result.chunks

    assume(len(chunks) > 0)

    for chunk in chunks:
        chunk_line_count = chunk.end_line - chunk.start_line + 1
        assert chunk_line_count <= chunk_lines, (
            f"Chunk exceeds max lines: {chunk_line_count} > {chunk_lines}"
        )

    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]

            actual_overlap = current_chunk.end_line - next_chunk.start_line + 1
            assert actual_overlap <= overlap_lines + 1, (
                f"Overlap between chunks {i} and {i + 1} is {actual_overlap}, "
                f"expected <= {overlap_lines + 1}"
            )
