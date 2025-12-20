"""Property-based tests for class summary generation."""

from hypothesis import given, settings

from aci.core.summary_artifact import ArtifactType
from aci.core.summary_generator import DEFAULT_MAX_SUMMARY_TOKENS, SummaryGenerator
from tests.support.summary_generator_strategies import (
    class_ast_node,
    class_with_methods,
    file_path,
)


@given(node=class_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_class_summary_contains_class_name(node, path: str):
    """Class summaries should include the class name and correct artifact type."""
    generator = SummaryGenerator()
    summary = generator.generate_class_summary(node, [], path)

    assert node.name in summary.content
    assert summary.artifact_type == ArtifactType.CLASS_SUMMARY


@given(node=class_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_class_summary_contains_base_classes(node, path: str):
    """Class summaries should include base classes when present."""
    generator = SummaryGenerator()
    summary = generator.generate_class_summary(node, [], path)

    expected_bases = generator._extract_base_classes(node.content, node.name)
    for base in expected_bases:
        assert base in summary.content or base in summary.metadata.get("base_classes", [])


@given(data=class_with_methods(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_class_summary_contains_method_names(data, path: str):
    """Class summaries should include method names."""
    class_node, methods = data
    generator = SummaryGenerator()
    summary = generator.generate_class_summary(class_node, methods, path)

    for method in methods:
        assert method.name in summary.content or method.name in summary.metadata.get("method_names", [])


@given(node=class_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_class_summary_preserves_location(node, path: str):
    """Class summaries should preserve source location metadata."""
    generator = SummaryGenerator()
    summary = generator.generate_class_summary(node, [], path)

    assert summary.start_line == node.start_line
    assert summary.end_line == node.end_line
    assert summary.file_path == path


@given(node=class_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_class_summary_format_consistency(node, path: str):
    """Class summaries should be non-empty, include the name, and respect token limits."""
    generator = SummaryGenerator()
    summary = generator.generate_class_summary(node, [], path)

    assert summary.content and len(summary.content.strip()) > 0
    assert node.name in summary.content

    from aci.core.tokenizer import get_default_tokenizer

    tokenizer = get_default_tokenizer()
    token_count = tokenizer.count_tokens(summary.content)
    assert token_count <= DEFAULT_MAX_SUMMARY_TOKENS

