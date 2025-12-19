"""
Property-based tests for JSDoc extraction.

Feature: comment-aware-search, Property 1: JavaScript Docstring Extraction
Validates: Requirements 1.1, 1.2, 1.3, 1.4
"""

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from aci.core.ast_parser import TreeSitterParser


# Simple strategy for generating valid JavaScript identifiers (no reserved words)
js_identifier = st.sampled_from([
    "foo", "bar", "baz", "myTest", "hello", "world",
    "addNum", "subNum", "mulNum", "divNum", "calc", "processData",
    "handleEvent", "createItem", "updateItem", "removeItem", "getValue", "setValue",
])

# Simple strategy for generating JSDoc content (ASCII only)
jsdoc_content = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?",
    min_size=5,
    max_size=50,
).filter(lambda x: x.strip())


class TestJSDocExtractionProperty:
    """
    Property 1: JavaScript Docstring Extraction
    
    For any JavaScript/TypeScript code containing a function, class, or method
    with a preceding JSDoc comment (/** ... */), when parsed by the AST parser,
    the resulting ASTNode SHALL have its docstring field populated.
    
    Feature: comment-aware-search, Property 1: JavaScript Docstring Extraction
    **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
    """

    def setup_method(self):
        self.parser = TreeSitterParser()

    @given(name=js_identifier, content=jsdoc_content)
    @settings(max_examples=50)
    def test_function_with_jsdoc_has_docstring(self, name: str, content: str):
        """Property: Any function with JSDoc SHALL have docstring populated."""
        code = f'''/** {content} */
function {name}() {{
    return 1;
}}'''
        nodes = self.parser.parse(code, "javascript")
        
        func_nodes = [n for n in nodes if n.name == name]
        assert len(func_nodes) == 1, f"Expected 1 function, got {len(func_nodes)}"
        assert func_nodes[0].docstring is not None
        assert content in func_nodes[0].docstring

    @given(name=js_identifier, content=jsdoc_content)
    @settings(max_examples=50)
    def test_class_with_jsdoc_has_docstring(self, name: str, content: str):
        """Property: Any class with JSDoc SHALL have docstring populated."""
        class_name = name.capitalize()
        
        code = f'''/** {content} */
class {class_name} {{
    constructor() {{}}
}}'''
        nodes = self.parser.parse(code, "javascript")
        
        class_nodes = [n for n in nodes if n.node_type == "class"]
        assert len(class_nodes) == 1
        assert class_nodes[0].docstring is not None
        assert content in class_nodes[0].docstring

    @given(name=js_identifier, content=jsdoc_content)
    @settings(max_examples=50)
    def test_arrow_function_with_jsdoc_has_docstring(self, name: str, content: str):
        """Property: Any arrow function with JSDoc SHALL have docstring populated."""
        code = f'''/** {content} */
const {name} = () => 1;'''
        nodes = self.parser.parse(code, "javascript")
        
        func_nodes = [n for n in nodes if n.name == name]
        assert len(func_nodes) == 1
        assert func_nodes[0].docstring is not None
        assert content in func_nodes[0].docstring

    @given(name=js_identifier)
    @settings(max_examples=30)
    def test_function_without_jsdoc_has_no_docstring(self, name: str):
        """Property: Function without JSDoc SHALL have None docstring."""
        code = f'''function {name}() {{
    return 1;
}}'''
        nodes = self.parser.parse(code, "javascript")
        
        func_nodes = [n for n in nodes if n.name == name]
        assert len(func_nodes) == 1
        assert func_nodes[0].docstring is None

    @given(content=jsdoc_content)
    @settings(max_examples=30)
    def test_jsdoc_with_param_tag_preserved(self, content: str):
        """Property: JSDoc @param tags SHALL be preserved."""
        code = f'''/**
 * {content}
 * @param x Input value
 */
function test(x) {{
    return x;
}}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].docstring is not None
        assert "@param" in nodes[0].docstring

    @given(content=jsdoc_content)
    @settings(max_examples=30)
    def test_jsdoc_with_returns_tag_preserved(self, content: str):
        """Property: JSDoc @returns tags SHALL be preserved."""
        code = f'''/**
 * {content}
 * @returns Result value
 */
function test() {{
    return 1;
}}'''
        nodes = self.parser.parse(code, "javascript")
        
        assert len(nodes) == 1
        assert nodes[0].docstring is not None
        assert "@returns" in nodes[0].docstring
