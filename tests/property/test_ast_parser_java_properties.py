"""
Property-based tests for Java docstring extraction.
"""

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


@given(name=identifier, doc=doc_text)
@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
def test_java_javadoc_property(name: str, doc: str):
    """Any Java class/method with Javadoc should surface docstring."""
    assume(name.isidentifier())
    parser = TreeSitterParser()
    class_name = name.capitalize()
    code = f"""/** {doc} */
public class {class_name} {{
    /** {doc} */
    public void doWork() {{}}
}}"""

    nodes = parser.parse(code, "java")
    class_node = next(n for n in nodes if n.node_type == "class" and n.name == class_name)
    method_node = next(n for n in nodes if n.node_type == "method" and n.name == "doWork")

    assert class_node.docstring is not None and doc in class_node.docstring
    assert method_node.docstring is not None and doc in method_node.docstring
