"""
Property-based tests for C/C++ Doxygen extraction.
"""

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from aci.core.ast_parser import TreeSitterParser

# C/C++ reserved keywords to avoid
CPP_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if", "int",
    "long", "register", "return", "short", "signed", "sizeof", "static",
    "struct", "switch", "typedef", "union", "unsigned", "void", "volatile",
    "while", "class", "public", "private", "protected", "virtual", "template",
    "typename", "namespace", "using", "new", "delete", "this", "throw", "try",
    "catch", "true", "false", "nullptr", "inline", "explicit", "friend",
    "operator", "mutable", "constexpr", "decltype", "noexcept", "override",
    "final", "alignas", "alignof", "asm", "bool", "wchar_t",
}

identifier = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    min_size=1,
    max_size=12,
)

doc_text = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;/@()-",
    min_size=1,
    max_size=60,
)


@given(lang=st.sampled_from(["c", "cpp"]), name=identifier, doc=doc_text)
@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
def test_c_cpp_doxygen_property(lang: str, name: str, doc: str):
    """Any C/C++ function with Doxygen comment should surface docstring."""
    assume(name[0].isalpha())
    # Avoid C/C++ reserved keywords
    assume(name.lower() not in CPP_KEYWORDS)
    parser = TreeSitterParser()
    code = f"""/** {doc} */
int {name}() {{
    return 0;
}}"""

    nodes = parser.parse(code, lang)
    func_node = next(n for n in nodes if n.node_type == "function" and n.name == name)

    assert func_node.docstring is not None
    assert doc in func_node.docstring
