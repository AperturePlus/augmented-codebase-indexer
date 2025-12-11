"""Integration tests between Chunker and SummaryGenerator."""

from hypothesis import given, settings

from aci.core.summary_artifact import ArtifactType
from tests.summary_generator_strategies import python_code_with_functions_and_classes


@given(code_data=python_code_with_functions_and_classes())
@settings(max_examples=100, deadline=None)
def test_chunker_produces_both_chunks_and_summaries(code_data):
    """Chunker should emit both code chunks and summary artifacts when generator provided."""
    from pathlib import Path

    from aci.core.ast_parser import TreeSitterParser
    from aci.core.chunker import Chunker
    from aci.core.file_scanner import ScannedFile
    from aci.core.summary_generator import SummaryGenerator
    from aci.core.tokenizer import get_default_tokenizer

    content, _, _ = code_data
    tokenizer = get_default_tokenizer()
    summary_generator = SummaryGenerator(tokenizer=tokenizer)
    chunker = Chunker(tokenizer=tokenizer, max_tokens=8192, summary_generator=summary_generator)

    parser = TreeSitterParser()
    ast_nodes = parser.parse(content, "python")
    scanned_file = ScannedFile(
        path=Path("/test/file.py"),
        content=content,
        language="python",
        size_bytes=len(content),
        modified_time=0.0,
        content_hash="test-hash",
    )

    result = chunker.chunk(scanned_file, ast_nodes)

    assert len(result.chunks) > 0
    assert len(result.summaries) > 0
    file_summaries = [s for s in result.summaries if s.artifact_type == ArtifactType.FILE_SUMMARY]
    assert len(file_summaries) == 1


@given(code_data=python_code_with_functions_and_classes())
@settings(max_examples=100, deadline=None)
def test_chunker_summary_count_matches_ast_nodes(code_data):
    """Chunker should emit summaries that match AST node counts."""
    from pathlib import Path

    from aci.core.ast_parser import TreeSitterParser
    from aci.core.chunker import Chunker
    from aci.core.file_scanner import ScannedFile
    from aci.core.summary_generator import SummaryGenerator
    from aci.core.tokenizer import get_default_tokenizer

    content, _, _ = code_data
    tokenizer = get_default_tokenizer()
    summary_generator = SummaryGenerator(tokenizer=tokenizer)
    chunker = Chunker(tokenizer=tokenizer, max_tokens=8192, summary_generator=summary_generator)

    parser = TreeSitterParser()
    ast_nodes = parser.parse(content, "python")
    scanned_file = ScannedFile(
        path=Path("/test/file.py"),
        content=content,
        language="python",
        size_bytes=len(content),
        modified_time=0.0,
        content_hash="test-hash",
    )

    result = chunker.chunk(scanned_file, ast_nodes)

    function_summaries = [s for s in result.summaries if s.artifact_type == ArtifactType.FUNCTION_SUMMARY]
    class_summaries = [s for s in result.summaries if s.artifact_type == ArtifactType.CLASS_SUMMARY]
    file_summaries = [s for s in result.summaries if s.artifact_type == ArtifactType.FILE_SUMMARY]

    function_nodes = [n for n in ast_nodes if n.node_type == "function"]
    method_nodes = [n for n in ast_nodes if n.node_type == "method"]
    class_nodes = [n for n in ast_nodes if n.node_type == "class"]

    expected_function_summaries = len(function_nodes) + len(method_nodes)
    assert len(function_summaries) == expected_function_summaries
    assert len(class_summaries) == len(class_nodes)
    assert len(file_summaries) == 1


@given(code_data=python_code_with_functions_and_classes())
@settings(max_examples=100, deadline=None)
def test_chunker_without_summary_generator_produces_no_summaries(code_data):
    """Chunker should not emit summaries when generator is absent."""
    from pathlib import Path

    from aci.core.ast_parser import TreeSitterParser
    from aci.core.chunker import Chunker
    from aci.core.file_scanner import ScannedFile
    from aci.core.tokenizer import get_default_tokenizer

    content, _, _ = code_data
    tokenizer = get_default_tokenizer()
    chunker = Chunker(tokenizer=tokenizer, max_tokens=8192, summary_generator=None)

    parser = TreeSitterParser()
    ast_nodes = parser.parse(content, "python")
    scanned_file = ScannedFile(
        path=Path("/test/file.py"),
        content=content,
        language="python",
        size_bytes=len(content),
        modified_time=0.0,
        content_hash="test-hash",
    )

    result = chunker.chunk(scanned_file, ast_nodes)

    assert len(result.chunks) > 0
    assert len(result.summaries) == 0

