"""
Property-based tests for C/C++ Doxygen extraction.
"""

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from aci.core.ast_parser import TreeSitterParser


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
    parser = TreeSitterParser()
    code = f"""/** {doc} */
int {name}() {{
    return 0;
}}"""

    nodes = parser.parse(code, lang)
    func_node = next(n for n in nodes if n.node_type == "function" and n.name == name)

    assert func_node.docstring is not None
    assert doc in func_node.docstring
