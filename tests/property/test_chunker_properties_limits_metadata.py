"""Property-based tests for chunker limits, splitting, and metadata."""

from hypothesis import assume, given, settings

from aci.core.ast_parser import ASTNode, TreeSitterParser
from aci.core.chunker import Chunker, SmartChunkSplitter
from aci.core.tokenizer import get_default_tokenizer
from tests.support.chunker_property_utils import (
    create_scanned_file,
    large_function_strategy,
    multi_line_text_strategy,
    python_class_with_methods_strategy,
    python_function_strategy,
)


@given(func_data=large_function_strategy())
@settings(max_examples=100)
def test_token_limit_compliance_ast_chunks(func_data):
    """
    **Feature: codebase-semantic-search, Property 7: Token Limit Compliance**
    **Validates: Requirements 2.5**
    """
    content, func_name = func_data

    tokenizer = get_default_tokenizer()
    max_tokens = 500

    chunker = Chunker(tokenizer=tokenizer, max_tokens=max_tokens)
    parser = TreeSitterParser()

    file = create_scanned_file(content)
    ast_nodes = parser.parse(content, "python")

    assume(len(ast_nodes) > 0)

    result = chunker.chunk(file, ast_nodes)
    chunks = result.chunks

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
    """
    content, num_lines = text_data
    assume(num_lines > 0)

    tokenizer = get_default_tokenizer()
    max_tokens = 500

    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        fixed_chunk_lines=50,
        overlap_lines=5,
    )

    file = create_scanned_file(content, language="unknown", path="/test/file.txt")
    result = chunker.chunk(file, [])
    chunks = result.chunks

    for chunk in chunks:
        token_count = tokenizer.count_tokens(chunk.content)
        assert token_count <= max_tokens, (
            f"Fixed chunk exceeds token limit: {token_count} > {max_tokens}"
        )


@given(func_data=large_function_strategy())
@settings(max_examples=100)
def test_smart_split_syntax_preservation(func_data):
    """
    **Feature: codebase-semantic-search, Property 7a: Smart Split Syntax Preservation**
    **Validates: Requirements 2.5**
    """
    content, func_name = func_data

    tokenizer = get_default_tokenizer()
    splitter = SmartChunkSplitter(tokenizer)

    lines = content.split("\n")
    node = ASTNode(
        node_type="function",
        name=func_name,
        start_line=1,
        end_line=len(lines),
        content=content,
    )

    max_tokens = 100

    chunks = splitter.split_oversized_node(
        node=node,
        max_tokens=max_tokens,
        file_path="/test/file.py",
        language="python",
        base_metadata={},
    )

    for chunk in chunks:
        chunk_content = chunk.content

        if chunk.metadata.get("has_context_prefix"):
            first_newline = chunk_content.find("\n")
            if first_newline > 0:
                chunk_content = chunk_content[first_newline + 1 :]

        if chunk_content:
            chunk_lines = chunk_content.split("\n")
            for line in chunk_lines:
                stripped = line.rstrip()
                if stripped:
                    assert not stripped.endswith(("=", "+", "-", "*", "/", "and", "or", "not")), (
                        f"Line appears to be truncated mid-statement: '{stripped}'"
                    )


@given(class_data=python_class_with_methods_strategy())
@settings(max_examples=100)
def test_metadata_completeness_methods(class_data):
    """
    **Feature: codebase-semantic-search, Property 8: Metadata Completeness**
    **Validates: Requirements 2.6**
    """
    content, class_name, method_names = class_data

    tokenizer = get_default_tokenizer()
    chunker = Chunker(tokenizer=tokenizer, max_tokens=8192)
    parser = TreeSitterParser()

    file = create_scanned_file(content)
    ast_nodes = parser.parse(content, "python")

    result = chunker.chunk(file, ast_nodes)
    chunks = result.chunks

    method_chunks = [c for c in chunks if c.chunk_type == "method"]

    for chunk in method_chunks:
        assert "parent_class" in chunk.metadata
        assert chunk.metadata["parent_class"] == class_name
        assert "function_name" in chunk.metadata


@given(func_data=python_function_strategy())
@settings(max_examples=100)
def test_metadata_completeness_functions(func_data):
    """
    **Feature: codebase-semantic-search, Property 8: Metadata Completeness**
    **Validates: Requirements 2.6**
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

    function_chunks = [c for c in chunks if c.chunk_type == "function"]

    for chunk in function_chunks:
        assert "function_name" in chunk.metadata
        assert chunk.metadata["function_name"] == func_name


@given(func_data=python_function_strategy())
@settings(max_examples=100)
def test_metadata_includes_file_hash(func_data):
    """
    **Feature: codebase-semantic-search, Property 8: Metadata Completeness**
    **Validates: Requirements 2.6**
    """
    content, _, _ = func_data

    tokenizer = get_default_tokenizer()
    chunker = Chunker(tokenizer=tokenizer, max_tokens=8192)
    parser = TreeSitterParser()

    file = create_scanned_file(content)
    ast_nodes = parser.parse(content, "python")

    result = chunker.chunk(file, ast_nodes)
    chunks = result.chunks

    for chunk in chunks:
        assert "file_hash" in chunk.metadata
        assert chunk.metadata["file_hash"] == file.content_hash
