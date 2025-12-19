"""Shared strategies for SummaryGenerator property tests."""

import string

from hypothesis import strategies as st

from aci.core.parsers.base import ASTNode

# Strategy primitives
identifier = st.text(alphabet=string.ascii_letters + "_", min_size=1, max_size=30).filter(
    lambda x: x[0].isalpha() or x[0] == "_"
)
param_with_type = st.one_of(
    identifier,
    st.tuples(identifier, identifier).map(lambda t: f"{t[0]}: {t[1]}"),
)
return_type = st.one_of(
    st.none(),
    st.sampled_from(["int", "str", "bool", "float", "None", "List", "Dict", "Optional[str]"]),
)
docstring = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z"), blacklist_characters='\x00"\''),
        min_size=1,
        max_size=200,
    ),
)
base_class = st.text(alphabet=string.ascii_letters + "_", min_size=1, max_size=20).filter(lambda x: x[0].isupper())
file_path = st.from_regex(r"[a-zA-Z0-9_]+(/[a-zA-Z0-9_]+)*\.(py|js|ts|go|java|c|cpp)", fullmatch=True)
language = st.sampled_from(["python", "javascript", "typescript", "go", "java", "c", "cpp"])
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
    if name[0].islower():
        name = name[0].upper() + name[1:]

    bases = draw(st.lists(base_class, max_size=3))
    doc = draw(docstring)

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
def method_ast_node(draw, parent_name: str | None = None):
    """Generate a valid method AST node."""
    name = draw(identifier)
    params = draw(st.lists(param_with_type, max_size=4))
    ret_type = draw(return_type)
    doc = draw(docstring)

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


@st.composite
def class_with_methods(draw):
    """Generate a class AST node with associated method nodes."""
    class_node = draw(class_ast_node())
    num_methods = draw(st.integers(min_value=0, max_value=5))

    methods = []
    for _ in range(num_methods):
        methods.append(draw(method_ast_node(parent_name=class_node.name)))

    return class_node, methods


@st.composite
def file_with_nodes(draw):
    """Generate file data with AST nodes and imports."""
    path = draw(file_path)
    lang = draw(language)

    num_imports = draw(st.integers(min_value=0, max_value=10))
    imports = [draw(python_import) for _ in range(num_imports)]

    num_functions = draw(st.integers(min_value=0, max_value=5))
    num_classes = draw(st.integers(min_value=0, max_value=3))

    nodes = []
    for _ in range(num_functions):
        nodes.append(draw(function_ast_node()))
    for _ in range(num_classes):
        nodes.append(draw(class_ast_node()))

    return path, lang, imports, nodes


@st.composite
def large_file_data(draw):
    """Generate file data with many definitions to test token limits."""
    path = draw(st.just("test_file.py"))
    lang = draw(st.just("python"))

    num_imports = draw(st.integers(min_value=20, max_value=50))
    imports = [f"import module_{i}" for i in range(num_imports)]

    num_functions = draw(st.integers(min_value=10, max_value=30))
    num_classes = draw(st.integers(min_value=5, max_value=15))

    nodes = []
    for i in range(num_functions):
        nodes.append(
            ASTNode(
                node_type="function",
                name=f"function_{i}",
                start_line=i * 10 + 1,
                end_line=i * 10 + 5,
                content=f"def function_{i}():\n    pass",
            )
        )
    for i in range(num_classes):
        nodes.append(
            ASTNode(
                node_type="class",
                name=f"Class_{i}",
                start_line=1000 + i * 10,
                end_line=1000 + i * 10 + 5,
                content=f"class Class_{i}:\n    pass",
            )
        )

    return path, lang, imports, nodes


@st.composite
def python_code_with_functions_and_classes(draw):
    """Generate Python code with functions and classes for chunker integration tests."""
    num_functions = draw(st.integers(min_value=1, max_value=3))
    num_classes = draw(st.integers(min_value=0, max_value=2))

    lines: list[str] = []
    lines.append("import os")
    lines.append("from typing import List")
    lines.append("")

    for i in range(num_functions):
        func_name = f"function_{i}"
        lines.append(f"def {func_name}():")
        lines.append(f'    """Docstring for {func_name}."""')
        lines.append("    pass")
        lines.append("")

    for i in range(num_classes):
        class_name = f"Class_{i}"
        lines.append(f"class {class_name}:")
        lines.append(f'    """Docstring for {class_name}."""')
        lines.append("")
        lines.append(f"    def method_{i}(self):")
        lines.append('        """Method docstring."""')
        lines.append("        pass")
        lines.append("")

    content = "\n".join(lines)
    return content, num_functions, num_classes

