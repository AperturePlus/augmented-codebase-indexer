"""Property-based tests for file summary generation."""

from hypothesis import given, settings

from aci.core.summary_artifact import ArtifactType
from aci.core.summary_generator import DEFAULT_MAX_SUMMARY_TOKENS, SummaryGenerator
from tests.support.summary_generator_strategies import file_with_nodes, large_file_data


@given(data=file_with_nodes())
@settings(max_examples=100, deadline=None)
def test_file_summary_contains_file_path(data):
    """File summaries should include the file path and correct artifact type."""
    path, lang, imports, nodes = data
    generator = SummaryGenerator()
    summary = generator.generate_file_summary(path, lang, imports, nodes)

    import os

    file_name = os.path.basename(path)
    assert file_name in summary.content
    assert summary.artifact_type == ArtifactType.FILE_SUMMARY


@given(data=file_with_nodes())
@settings(max_examples=100, deadline=None)
def test_file_summary_contains_language(data):
    """File summaries should include language information."""
    path, lang, imports, nodes = data
    generator = SummaryGenerator()
    summary = generator.generate_file_summary(path, lang, imports, nodes)

    assert lang in summary.content or summary.metadata.get("language") == lang


@given(data=file_with_nodes())
@settings(max_examples=100, deadline=None)
def test_file_summary_contains_definitions(data):
    """File summaries should list top-level definitions."""
    path, lang, imports, nodes = data
    generator = SummaryGenerator()
    summary = generator.generate_file_summary(path, lang, imports, nodes)

    for node in nodes:
        if node.node_type in ("function", "class"):
            assert node.name in summary.content


@given(data=file_with_nodes())
@settings(max_examples=100, deadline=None)
def test_file_summary_has_file_location(data):
    """File summary line numbers should be zero for file-level summaries."""
    path, lang, imports, nodes = data
    generator = SummaryGenerator()
    summary = generator.generate_file_summary(path, lang, imports, nodes)

    assert summary.start_line == 0
    assert summary.end_line == 0
    assert summary.file_path == path


@given(data=large_file_data())
@settings(max_examples=50, deadline=None)
def test_file_summary_respects_token_limit(data):
    """File summaries should respect configured token limits."""
    path, lang, imports, nodes = data

    max_tokens = DEFAULT_MAX_SUMMARY_TOKENS
    generator = SummaryGenerator(max_summary_tokens=max_tokens)
    summary = generator.generate_file_summary(path, lang, imports, nodes)

    from aci.core.tokenizer import get_default_tokenizer

    tokenizer = get_default_tokenizer()
    token_count = tokenizer.count_tokens(summary.content)
    assert token_count <= max_tokens


@given(data=file_with_nodes())
@settings(max_examples=100, deadline=None)
def test_file_summary_content_not_empty(data):
    """File summaries should contain non-empty content."""
    path, lang, imports, nodes = data
    generator = SummaryGenerator()
    summary = generator.generate_file_summary(path, lang, imports, nodes)

    assert summary.content and len(summary.content.strip()) > 0


@given(data=file_with_nodes())
@settings(max_examples=100, deadline=None)
def test_file_summary_format_consistency(data):
    """File summaries should include file name and respect token limits."""
    path, lang, imports, nodes = data
    generator = SummaryGenerator()
    summary = generator.generate_file_summary(path, lang, imports, nodes)

    import os

    file_name = os.path.basename(path)
    assert summary.content and len(summary.content.strip()) > 0
    assert file_name in summary.content

    from aci.core.tokenizer import get_default_tokenizer

    tokenizer = get_default_tokenizer()
    token_count = tokenizer.count_tokens(summary.content)
    assert token_count <= DEFAULT_MAX_SUMMARY_TOKENS

