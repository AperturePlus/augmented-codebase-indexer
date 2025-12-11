"""
Property-based tests for SummaryGenerator.

Tests the correctness properties for function, class, and file summary generation.
"""

import string

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from aci.core.parsers.base import ASTNode
from aci.core.summary_artifact import ArtifactType
from aci.core.summary_generator import SummaryGenerator, DEFAULT_MAX_SUMMARY_TOKENS


# Strategies for generating valid identifiers
identifier = st.text(
    alphabet=string.ascii_letters + "_",
    min_size=1,
    max_size=30,
).filter(lambda x: x[0].isalpha() or x[0] == "_")

# Strategy for parameter names with optional type hints
param_with_type = st.one_of(
    identifier,  # Just name
    st.tuples(identifier, identifier).map(lambda t: f"{t[0]}: {t[1]}"),  # name: type
)

# Strategy for return types
return_type = st.one_of(
    st.none(),
    st.sampled_from(["int", "str", "bool", "float", "None", "List", "Dict", "Optional[str]"]),
)

# Strategy for docstrings
docstring = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "Z"),
            blacklist_characters='\x00"\'',
        ),
        min_size=1,
        max_size=200,
    ),
)

# Strategy for base classes
base_class = st.text(
    alphabet=string.ascii_letters + "_",
    min_size=1,
    max_size=20,
).filter(lambda x: x[0].isupper())

# Strategy for file paths
file_path = st.from_regex(
    r"[a-zA-Z0-9_]+(/[a-zA-Z0-9_]+)*\.(py|js|ts|go|java|c|cpp)",
    fullmatch=True,
)

# Strategy for languages
language = st.sampled_from(["python", "javascript", "typescript", "go", "java", "c", "cpp"])

# Strategy for import statements
python_import = st.one_of(
    identifier.map(lambda x: f"import {x}"),
    st.tuples(identifier, identifier).map(lambda t: f"from {t[0]} import {t[1]}"),
)


@st.composite
def function_ast_node(draw):
    """Generate a valid function AST node."""
    name = draw(identifier)
    params = draw(st.lists(param_with_type, max_size=5))
    ret_type = draw(return_type)
    doc = draw(docstring)
    is_async = draw(st.booleans())
    
    # Build function content
    async_prefix = "async " if is_async else ""
    params_str = ", ".join(params)
    ret_annotation = f" -> {ret_type}" if ret_type else ""
    
    content_lines = [f"{async_prefix}def {name}({params_str}){ret_annotation}:"]
    if doc:
        content_lines.append(f'    """{doc}"""')
    content_lines.append("    pass")
    
    content = "\n".join(content_lines)
    start_line = draw(st.integers(min_value=1, max_value=1000))
    end_line = start_line + len(content_lines) - 1
    
    return ASTNode(
        node_type="function",
        name=name,
        start_line=start_line,
        end_line=end_line,
        content=content,
        docstring=doc,
    )


@st.composite
def class_ast_node(draw):
    """Generate a valid class AST node."""
    name = draw(identifier.filter(lambda x: x[0].isupper() or x[0] == "_"))
    # Ensure class name starts with uppercase for convention
    if name[0].islower():
        name = name[0].upper() + name[1:]
    
    bases = draw(st.lists(base_class, max_size=3))
    doc = draw(docstring)
    
    # Build class content
    bases_str = f"({', '.join(bases)})" if bases else ""
    content_lines = [f"class {name}{bases_str}:"]
    if doc:
        content_lines.append(f'    """{doc}"""')
    content_lines.append("    pass")
    
    content = "\n".join(content_lines)
    start_line = draw(st.integers(min_value=1, max_value=1000))
    end_line = start_line + len(content_lines) - 1
    
    return ASTNode(
        node_type="class",
        name=name,
        start_line=start_line,
        end_line=end_line,
        content=content,
        docstring=doc,
    )


@st.composite
def method_ast_node(draw, parent_name: str = None):
    """Generate a valid method AST node."""
    name = draw(identifier)
    params = draw(st.lists(param_with_type, max_size=4))
    ret_type = draw(return_type)
    doc = draw(docstring)
    
    # Methods always have self as first param
    all_params = ["self"] + params
    params_str = ", ".join(all_params)
    ret_annotation = f" -> {ret_type}" if ret_type else ""
    
    content_lines = [f"def {name}({params_str}){ret_annotation}:"]
    if doc:
        content_lines.append(f'    """{doc}"""')
    content_lines.append("    pass")
    
    content = "\n".join(content_lines)
    start_line = draw(st.integers(min_value=1, max_value=1000))
    end_line = start_line + len(content_lines) - 1
    
    return ASTNode(
        node_type="method",
        name=name,
        start_line=start_line,
        end_line=end_line,
        content=content,
        parent_name=parent_name or draw(identifier),
        docstring=doc,
    )


# ============================================================================
# Property 1: Function summary contains required fields
# ============================================================================

@given(node=function_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_function_summary_contains_function_name(node: ASTNode, path: str):
    """
    **Feature: multi-granularity-indexing, Property 1: Function summary contains required fields**
    **Validates: Requirements 1.1**

    For any valid function AST node, the generated function summary SHALL contain
    the function name in the output content.
    """
    generator = SummaryGenerator()
    summary = generator.generate_function_summary(node, path)
    
    # Function name must be in the content
    assert node.name in summary.content, f"Function name '{node.name}' not found in summary"
    
    # Artifact type must be correct
    assert summary.artifact_type == ArtifactType.FUNCTION_SUMMARY


@given(node=function_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_function_summary_contains_parameters(node: ASTNode, path: str):
    """
    **Feature: multi-granularity-indexing, Property 1: Function summary contains required fields**
    **Validates: Requirements 1.1**

    For any valid function AST node with parameters, the generated function summary
    SHALL contain all parameter names in the output content.
    """
    generator = SummaryGenerator()
    summary = generator.generate_function_summary(node, path)
    
    # Extract expected parameters from the node content
    expected_params = generator._extract_parameters(node.content, node.name)
    
    # Each parameter name should appear in the summary content or metadata
    for param in expected_params:
        # Get just the parameter name (without type annotation)
        param_name = param.split(":")[0].strip()
        # Parameter should be in content or metadata
        assert (
            param_name in summary.content or 
            any(param_name in str(p) for p in summary.metadata.get("parameters", []))
        ), f"Parameter '{param_name}' not found in summary"


@given(node=function_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_function_summary_contains_return_type_when_present(node: ASTNode, path: str):
    """
    **Feature: multi-granularity-indexing, Property 1: Function summary contains required fields**
    **Validates: Requirements 1.1**

    For any valid function AST node with return type hint, the generated function
    summary SHALL contain the return type in the output content.
    """
    generator = SummaryGenerator()
    summary = generator.generate_function_summary(node, path)
    
    # Extract return type from node content
    return_type = generator._extract_return_type(node.content)
    
    if return_type:
        # Return type should be in content or metadata
        assert (
            return_type in summary.content or
            summary.metadata.get("return_type") == return_type
        ), f"Return type '{return_type}' not found in summary"


@given(node=function_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_function_summary_preserves_location(node: ASTNode, path: str):
    """
    **Feature: multi-granularity-indexing, Property 1: Function summary contains required fields**
    **Validates: Requirements 1.1**

    For any valid function AST node, the generated summary SHALL preserve
    the source location (start_line, end_line) from the original node.
    """
    generator = SummaryGenerator()
    summary = generator.generate_function_summary(node, path)
    
    assert summary.start_line == node.start_line
    assert summary.end_line == node.end_line
    assert summary.file_path == path


# ============================================================================
# Property 2: Class summary contains required fields
# ============================================================================

@given(node=class_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_class_summary_contains_class_name(node: ASTNode, path: str):
    """
    **Feature: multi-granularity-indexing, Property 2: Class summary contains required fields**
    **Validates: Requirements 1.2**

    For any valid class AST node, the generated class summary SHALL contain
    the class name in the output content.
    """
    generator = SummaryGenerator()
    summary = generator.generate_class_summary(node, [], path)
    
    # Class name must be in the content
    assert node.name in summary.content, f"Class name '{node.name}' not found in summary"
    
    # Artifact type must be correct
    assert summary.artifact_type == ArtifactType.CLASS_SUMMARY


@given(node=class_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_class_summary_contains_base_classes(node: ASTNode, path: str):
    """
    **Feature: multi-granularity-indexing, Property 2: Class summary contains required fields**
    **Validates: Requirements 1.2**

    For any valid class AST node with base classes, the generated class summary
    SHALL contain all base class names in the output content.
    """
    generator = SummaryGenerator()
    summary = generator.generate_class_summary(node, [], path)
    
    # Extract expected base classes from the node content
    expected_bases = generator._extract_base_classes(node.content, node.name)
    
    # Each base class should appear in the summary content or metadata
    for base in expected_bases:
        assert (
            base in summary.content or
            base in summary.metadata.get("base_classes", [])
        ), f"Base class '{base}' not found in summary"


@st.composite
def class_with_methods(draw):
    """Generate a class AST node with associated method nodes."""
    class_node = draw(class_ast_node())
    num_methods = draw(st.integers(min_value=0, max_value=5))
    
    methods = []
    for _ in range(num_methods):
        method = draw(method_ast_node(parent_name=class_node.name))
        methods.append(method)
    
    return class_node, methods


@given(data=class_with_methods(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_class_summary_contains_method_names(data, path: str):
    """
    **Feature: multi-granularity-indexing, Property 2: Class summary contains required fields**
    **Validates: Requirements 1.2**

    For any valid class AST node with methods, the generated class summary
    SHALL contain all method names in the output content.
    """
    class_node, methods = data
    generator = SummaryGenerator()
    summary = generator.generate_class_summary(class_node, methods, path)
    
    # Each method name should appear in the summary content or metadata
    for method in methods:
        assert (
            method.name in summary.content or
            method.name in summary.metadata.get("method_names", [])
        ), f"Method name '{method.name}' not found in summary"


@given(node=class_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_class_summary_preserves_location(node: ASTNode, path: str):
    """
    **Feature: multi-granularity-indexing, Property 2: Class summary contains required fields**
    **Validates: Requirements 1.2**

    For any valid class AST node, the generated summary SHALL preserve
    the source location (start_line, end_line) from the original node.
    """
    generator = SummaryGenerator()
    summary = generator.generate_class_summary(node, [], path)
    
    assert summary.start_line == node.start_line
    assert summary.end_line == node.end_line
    assert summary.file_path == path


# ============================================================================
# Property 4: File summary contains required information
# ============================================================================

@st.composite
def file_with_nodes(draw):
    """Generate file data with AST nodes and imports."""
    path = draw(file_path)
    lang = draw(language)
    
    # Generate imports
    num_imports = draw(st.integers(min_value=0, max_value=10))
    imports = [draw(python_import) for _ in range(num_imports)]
    
    # Generate nodes (mix of functions and classes)
    num_functions = draw(st.integers(min_value=0, max_value=5))
    num_classes = draw(st.integers(min_value=0, max_value=3))
    
    nodes = []
    for _ in range(num_functions):
        nodes.append(draw(function_ast_node()))
    for _ in range(num_classes):
        nodes.append(draw(class_ast_node()))
    
    return path, lang, imports, nodes


@given(data=file_with_nodes())
@settings(max_examples=100, deadline=None)
def test_file_summary_contains_file_path(data):
    """
    **Feature: multi-granularity-indexing, Property 4: File summary contains required information**
    **Validates: Requirements 2.1, 2.2, 2.3**

    For any file, the generated file summary SHALL contain the file path
    in the output content.
    """
    path, lang, imports, nodes = data
    generator = SummaryGenerator()
    summary = generator.generate_file_summary(path, lang, imports, nodes)
    
    import os
    file_name = os.path.basename(path)
    
    # File name must be in the content
    assert file_name in summary.content, f"File name '{file_name}' not found in summary"
    
    # Artifact type must be correct
    assert summary.artifact_type == ArtifactType.FILE_SUMMARY


@given(data=file_with_nodes())
@settings(max_examples=100, deadline=None)
def test_file_summary_contains_language(data):
    """
    **Feature: multi-granularity-indexing, Property 4: File summary contains required information**
    **Validates: Requirements 2.1, 2.2, 2.3**

    For any file, the generated file summary SHALL contain the language identifier
    in the output content.
    """
    path, lang, imports, nodes = data
    generator = SummaryGenerator()
    summary = generator.generate_file_summary(path, lang, imports, nodes)
    
    # Language must be in the content or metadata
    assert (
        lang in summary.content or
        summary.metadata.get("language") == lang
    ), f"Language '{lang}' not found in summary"


@given(data=file_with_nodes())
@settings(max_examples=100, deadline=None)
def test_file_summary_contains_definitions(data):
    """
    **Feature: multi-granularity-indexing, Property 4: File summary contains required information**
    **Validates: Requirements 2.1, 2.2, 2.3**

    For any file with AST nodes, the generated file summary SHALL contain
    the names of top-level definitions (functions and classes).
    """
    path, lang, imports, nodes = data
    generator = SummaryGenerator()
    summary = generator.generate_file_summary(path, lang, imports, nodes)
    
    # Each function and class name should appear in the summary
    for node in nodes:
        if node.node_type in ("function", "class"):
            assert (
                node.name in summary.content
            ), f"Definition '{node.name}' not found in summary"


@given(data=file_with_nodes())
@settings(max_examples=100, deadline=None)
def test_file_summary_has_file_location(data):
    """
    **Feature: multi-granularity-indexing, Property 4: File summary contains required information**
    **Validates: Requirements 2.1, 2.2, 2.3**

    For any file summary, start_line and end_line SHALL be 0 (file-level).
    """
    path, lang, imports, nodes = data
    generator = SummaryGenerator()
    summary = generator.generate_file_summary(path, lang, imports, nodes)
    
    # File summaries have 0 for line numbers
    assert summary.start_line == 0
    assert summary.end_line == 0
    assert summary.file_path == path


# ============================================================================
# Property 5: File summary respects token limits
# ============================================================================

@st.composite
def large_file_data(draw):
    """Generate file data with many definitions to test token limits."""
    # Use simpler strategies to avoid filtering issues
    path = draw(st.just("test_file.py"))
    lang = draw(st.just("python"))
    
    # Generate many imports using simple format
    num_imports = draw(st.integers(min_value=20, max_value=50))
    imports = [f"import module_{i}" for i in range(num_imports)]
    
    # Generate many nodes with simple names
    num_functions = draw(st.integers(min_value=10, max_value=30))
    num_classes = draw(st.integers(min_value=5, max_value=15))
    
    nodes = []
    for i in range(num_functions):
        nodes.append(ASTNode(
            node_type="function",
            name=f"function_{i}",
            start_line=i * 10 + 1,
            end_line=i * 10 + 5,
            content=f"def function_{i}():\n    pass",
        ))
    for i in range(num_classes):
        nodes.append(ASTNode(
            node_type="class",
            name=f"Class_{i}",
            start_line=1000 + i * 10,
            end_line=1000 + i * 10 + 5,
            content=f"class Class_{i}:\n    pass",
        ))
    
    return path, lang, imports, nodes


@given(data=large_file_data())
@settings(max_examples=50, deadline=None)
def test_file_summary_respects_token_limit(data):
    """
    **Feature: multi-granularity-indexing, Property 5: File summary respects token limits**
    **Validates: Requirements 2.4**

    For any file regardless of the number of definitions, the generated file
    summary content SHALL not exceed the configured maximum token limit.
    """
    path, lang, imports, nodes = data
    
    max_tokens = DEFAULT_MAX_SUMMARY_TOKENS
    generator = SummaryGenerator(max_summary_tokens=max_tokens)
    summary = generator.generate_file_summary(path, lang, imports, nodes)
    
    # Count tokens in the summary content
    from aci.core.tokenizer import get_default_tokenizer
    tokenizer = get_default_tokenizer()
    token_count = tokenizer.count_tokens(summary.content)
    
    # Token count must not exceed limit
    assert token_count <= max_tokens, (
        f"Summary has {token_count} tokens, exceeds limit of {max_tokens}"
    )


@given(data=file_with_nodes())
@settings(max_examples=100, deadline=None)
def test_file_summary_content_not_empty(data):
    """
    **Feature: multi-granularity-indexing, Property 5: File summary respects token limits**
    **Validates: Requirements 2.4**

    For any file, the generated file summary SHALL have non-empty content.
    """
    path, lang, imports, nodes = data
    generator = SummaryGenerator()
    summary = generator.generate_file_summary(path, lang, imports, nodes)
    
    # Content must not be empty
    assert summary.content, "File summary content should not be empty"
    assert len(summary.content.strip()) > 0


# ============================================================================
# Property 3: Summary format consistency
# ============================================================================

@given(node=function_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_function_summary_format_consistency(node: ASTNode, path: str):
    """
    **Feature: multi-granularity-indexing, Property 3: Summary format consistency**
    **Validates: Requirements 1.4**

    For any generated function summary, the content SHALL be non-empty,
    contain the entity name, and have length within token limits (≤ 512 tokens).
    """
    generator = SummaryGenerator()
    summary = generator.generate_function_summary(node, path)
    
    # Content must be non-empty
    assert summary.content, "Summary content should not be empty"
    assert len(summary.content.strip()) > 0
    
    # Content must contain the entity name
    assert node.name in summary.content, f"Entity name '{node.name}' not in summary"
    
    # Content must be within token limits
    from aci.core.tokenizer import get_default_tokenizer
    tokenizer = get_default_tokenizer()
    token_count = tokenizer.count_tokens(summary.content)
    assert token_count <= DEFAULT_MAX_SUMMARY_TOKENS, (
        f"Summary has {token_count} tokens, exceeds limit of {DEFAULT_MAX_SUMMARY_TOKENS}"
    )


@given(node=class_ast_node(), path=file_path)
@settings(max_examples=100, deadline=None)
def test_class_summary_format_consistency(node: ASTNode, path: str):
    """
    **Feature: multi-granularity-indexing, Property 3: Summary format consistency**
    **Validates: Requirements 1.4**

    For any generated class summary, the content SHALL be non-empty,
    contain the entity name, and have length within token limits (≤ 512 tokens).
    """
    generator = SummaryGenerator()
    summary = generator.generate_class_summary(node, [], path)
    
    # Content must be non-empty
    assert summary.content, "Summary content should not be empty"
    assert len(summary.content.strip()) > 0
    
    # Content must contain the entity name
    assert node.name in summary.content, f"Entity name '{node.name}' not in summary"
    
    # Content must be within token limits
    from aci.core.tokenizer import get_default_tokenizer
    tokenizer = get_default_tokenizer()
    token_count = tokenizer.count_tokens(summary.content)
    assert token_count <= DEFAULT_MAX_SUMMARY_TOKENS, (
        f"Summary has {token_count} tokens, exceeds limit of {DEFAULT_MAX_SUMMARY_TOKENS}"
    )


@given(data=file_with_nodes())
@settings(max_examples=100, deadline=None)
def test_file_summary_format_consistency(data):
    """
    **Feature: multi-granularity-indexing, Property 3: Summary format consistency**
    **Validates: Requirements 1.4**

    For any generated file summary, the content SHALL be non-empty,
    contain the file name, and have length within token limits (≤ 512 tokens).
    """
    path, lang, imports, nodes = data
    generator = SummaryGenerator()
    summary = generator.generate_file_summary(path, lang, imports, nodes)
    
    import os
    file_name = os.path.basename(path)
    
    # Content must be non-empty
    assert summary.content, "Summary content should not be empty"
    assert len(summary.content.strip()) > 0
    
    # Content must contain the file name
    assert file_name in summary.content, f"File name '{file_name}' not in summary"
    
    # Content must be within token limits
    from aci.core.tokenizer import get_default_tokenizer
    tokenizer = get_default_tokenizer()
    token_count = tokenizer.count_tokens(summary.content)
    assert token_count <= DEFAULT_MAX_SUMMARY_TOKENS, (
        f"Summary has {token_count} tokens, exceeds limit of {DEFAULT_MAX_SUMMARY_TOKENS}"
    )


# ============================================================================
# Chunker Summary Integration Tests
# ============================================================================

@st.composite
def python_code_with_functions_and_classes(draw):
    """Generate Python code with functions and classes for chunker integration tests."""
    num_functions = draw(st.integers(min_value=1, max_value=3))
    num_classes = draw(st.integers(min_value=0, max_value=2))
    
    lines = []
    
    # Add some imports
    lines.append("import os")
    lines.append("from typing import List")
    lines.append("")
    
    # Add functions
    for i in range(num_functions):
        func_name = f"function_{i}"
        lines.append(f"def {func_name}():")
        lines.append(f'    """Docstring for {func_name}."""')
        lines.append("    pass")
        lines.append("")
    
    # Add classes with methods
    for i in range(num_classes):
        class_name = f"Class_{i}"
        lines.append(f"class {class_name}:")
        lines.append(f'    """Docstring for {class_name}."""')
        lines.append("")
        # Add a method
        lines.append(f"    def method_{i}(self):")
        lines.append(f'        """Method docstring."""')
        lines.append("        pass")
        lines.append("")
    
    content = "\n".join(lines)
    return content, num_functions, num_classes


@given(code_data=python_code_with_functions_and_classes())
@settings(max_examples=100, deadline=None)
def test_chunker_produces_both_chunks_and_summaries(code_data):
    """
    **Feature: multi-granularity-indexing, Property: Chunker summary integration**
    **Validates: Requirements 1.1, 1.2, 2.1**

    For any file with AST nodes, when a SummaryGenerator is provided to the Chunker,
    the ChunkingResult SHALL contain both code chunks and summary artifacts.
    """
    from aci.core.ast_parser import TreeSitterParser
    from aci.core.chunker import Chunker
    from aci.core.file_scanner import ScannedFile
    from aci.core.summary_generator import SummaryGenerator
    from aci.core.tokenizer import get_default_tokenizer
    from pathlib import Path
    
    content, num_functions, num_classes = code_data
    
    # Create chunker with summary generator
    tokenizer = get_default_tokenizer()
    summary_generator = SummaryGenerator(tokenizer=tokenizer)
    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=8192,
        summary_generator=summary_generator,
    )
    
    # Parse and chunk
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
    
    # Should have chunks
    assert len(result.chunks) > 0, "ChunkingResult should contain chunks"
    
    # Should have summaries (at least file summary)
    assert len(result.summaries) > 0, "ChunkingResult should contain summaries"
    
    # Should have file summary
    file_summaries = [s for s in result.summaries if s.artifact_type == ArtifactType.FILE_SUMMARY]
    assert len(file_summaries) == 1, "Should have exactly one file summary"


@given(code_data=python_code_with_functions_and_classes())
@settings(max_examples=100, deadline=None)
def test_chunker_summary_count_matches_ast_nodes(code_data):
    """
    **Feature: multi-granularity-indexing, Property: Chunker summary integration**
    **Validates: Requirements 1.1, 1.2, 2.1**

    For any file with AST nodes, the number of function/class summaries SHALL match
    the number of function/class AST nodes, plus one file summary.
    """
    from aci.core.ast_parser import TreeSitterParser
    from aci.core.chunker import Chunker
    from aci.core.file_scanner import ScannedFile
    from aci.core.summary_generator import SummaryGenerator
    from aci.core.tokenizer import get_default_tokenizer
    from pathlib import Path
    
    content, num_functions, num_classes = code_data
    
    # Create chunker with summary generator
    tokenizer = get_default_tokenizer()
    summary_generator = SummaryGenerator(tokenizer=tokenizer)
    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=8192,
        summary_generator=summary_generator,
    )
    
    # Parse and chunk
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
    
    # Count summaries by type
    function_summaries = [s for s in result.summaries if s.artifact_type == ArtifactType.FUNCTION_SUMMARY]
    class_summaries = [s for s in result.summaries if s.artifact_type == ArtifactType.CLASS_SUMMARY]
    file_summaries = [s for s in result.summaries if s.artifact_type == ArtifactType.FILE_SUMMARY]
    
    # Count AST nodes by type
    function_nodes = [n for n in ast_nodes if n.node_type == "function"]
    method_nodes = [n for n in ast_nodes if n.node_type == "method"]
    class_nodes = [n for n in ast_nodes if n.node_type == "class"]
    
    # Function summaries should match function + method nodes
    expected_function_summaries = len(function_nodes) + len(method_nodes)
    assert len(function_summaries) == expected_function_summaries, (
        f"Expected {expected_function_summaries} function summaries, got {len(function_summaries)}"
    )
    
    # Class summaries should match class nodes
    assert len(class_summaries) == len(class_nodes), (
        f"Expected {len(class_nodes)} class summaries, got {len(class_summaries)}"
    )
    
    # Should have exactly one file summary
    assert len(file_summaries) == 1, "Should have exactly one file summary"


@given(code_data=python_code_with_functions_and_classes())
@settings(max_examples=100, deadline=None)
def test_chunker_without_summary_generator_produces_no_summaries(code_data):
    """
    **Feature: multi-granularity-indexing, Property: Chunker summary integration**
    **Validates: Requirements 1.1, 1.2, 2.1**

    For any file, when no SummaryGenerator is provided to the Chunker,
    the ChunkingResult SHALL contain chunks but no summary artifacts.
    """
    from aci.core.ast_parser import TreeSitterParser
    from aci.core.chunker import Chunker
    from aci.core.file_scanner import ScannedFile
    from aci.core.tokenizer import get_default_tokenizer
    from pathlib import Path
    
    content, num_functions, num_classes = code_data
    
    # Create chunker WITHOUT summary generator
    tokenizer = get_default_tokenizer()
    chunker = Chunker(
        tokenizer=tokenizer,
        max_tokens=8192,
        summary_generator=None,  # No summary generator
    )
    
    # Parse and chunk
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
    
    # Should have chunks
    assert len(result.chunks) > 0, "ChunkingResult should contain chunks"
    
    # Should NOT have summaries
    assert len(result.summaries) == 0, "ChunkingResult should not contain summaries when no generator provided"
