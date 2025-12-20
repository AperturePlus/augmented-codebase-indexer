"""Property-based tests for function summary generation."""

from hypothesis import given, settings

from aci.core.summary_artifact import ArtifactType
from aci.core.summary_generator import DEFAULT_MAX_SUMMARY_TOKENS, SummaryGenerator
from tests.support.summary_generator_strategies import file_path, function_ast_node


@given(node=function_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_function_summary_contains_function_name(node, path: str):
    """Function summaries should include the function name and correct artifact type."""
    generator = SummaryGenerator()
    summary = generator.generate_function_summary(node, path)

    assert node.name in summary.content
    assert summary.artifact_type == ArtifactType.FUNCTION_SUMMARY


@given(node=function_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_function_summary_contains_parameters(node, path: str):
    """Function summaries should include parameter names."""
    generator = SummaryGenerator()
    summary = generator.generate_function_summary(node, path)

    expected_params = generator._extract_parameters(node.content, node.name)
    for param in expected_params:
        param_name = param.split(":")[0].strip()
        assert param_name in summary.content or any(
            param_name in str(p) for p in summary.metadata.get("parameters", [])
        )


@given(node=function_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_function_summary_contains_return_type_when_present(node, path: str):
    """Function summaries should include return type when provided."""
    generator = SummaryGenerator()
    summary = generator.generate_function_summary(node, path)

    return_type = generator._extract_return_type(node.content)
    if return_type:
        assert return_type in summary.content or summary.metadata.get("return_type") == return_type


@given(node=function_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_function_summary_preserves_location(node, path: str):
    """Function summaries should preserve source location metadata."""
    generator = SummaryGenerator()
    summary = generator.generate_function_summary(node, path)

    assert summary.start_line == node.start_line
    assert summary.end_line == node.end_line
    assert summary.file_path == path


@given(node=function_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_function_summary_format_consistency(node, path: str):
    """Function summaries should be non-empty, include the name, and respect token limits."""
    generator = SummaryGenerator()
    summary = generator.generate_function_summary(node, path)

    assert summary.content and len(summary.content.strip()) > 0
    assert node.name in summary.content

    from aci.core.tokenizer import get_default_tokenizer

    tokenizer = get_default_tokenizer()
    token_count = tokenizer.count_tokens(summary.content)
    assert token_count <= DEFAULT_MAX_SUMMARY_TOKENS

