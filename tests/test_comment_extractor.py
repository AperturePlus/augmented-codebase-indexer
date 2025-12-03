"""
Direct tests for the comment extractor heuristics.
"""

from aci.core.ast_parser import TreeSitterParser


def test_jsdoc_allows_export_prefix():
    """JSDoc should survive common export modifiers."""
    code = """/** Adds numbers */
export function add(a, b) {
    return a + b;
}
"""
    parser = TreeSitterParser()
    nodes = parser.parse(code, "javascript")

    assert len(nodes) == 1
    assert nodes[0].docstring is not None
    assert "Adds numbers" in nodes[0].docstring


def test_jsdoc_rejects_distant_comment():
    """Distant comments should not attach as docstrings."""
    code = """/** Not a doc */


function real() {
    return 1;
}
"""
    parser = TreeSitterParser()
    nodes = parser.parse(code, "javascript")

    assert len(nodes) == 1
    assert nodes[0].docstring is None


def test_go_doc_requires_name_prefix():
    """Go doc comment must start with the identifier name."""
    code = """package main
// does something unrelated
func Add(a, b int) int {
    return a + b
}
"""
    parser = TreeSitterParser()
    nodes = parser.parse(code, "go")
    func_node = next(n for n in nodes if n.name == "Add")

    assert func_node.docstring is None


def test_go_block_comment_supported():
    """Block comments immediately before a Go declaration are accepted."""
    code = """package main
/* Add sums two numbers. */
func Add(a, b int) int {
    return a + b
}
"""
    parser = TreeSitterParser()
    nodes = parser.parse(code, "go")
    func_node = next(n for n in nodes if n.name == "Add")

    assert func_node.docstring is not None
    assert "sums two numbers" in func_node.docstring
