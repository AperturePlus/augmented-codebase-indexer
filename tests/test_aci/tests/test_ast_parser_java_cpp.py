"""
Tests for Java and C/C++ parsing support.
"""

from aci.core.ast_parser import TreeSitterParser


def test_java_class_and_method_docstrings():
    """Java parser should extract classes/methods with Javadoc."""
    code = """/** Represents a user */
public class User {
    /** Greets with a name */
    public String greet(String name) {
        return "Hello " + name;
    }
}
"""
    parser = TreeSitterParser()
    nodes = parser.parse(code, "java")

    class_node = next(n for n in nodes if n.node_type == "class")
    method_node = next(n for n in nodes if n.node_type == "method")

    assert class_node.name == "User"
    assert class_node.docstring and "Represents a user" in class_node.docstring

    assert method_node.name == "greet"
    assert method_node.docstring and "Greets with a name" in method_node.docstring


def test_cpp_class_and_function_docstrings():
    """C++ parser should extract class/struct and free functions with Doxygen comments."""
    code = """/** Utility helpers */
class Util {
public:
    /// Computes a value
    int compute();
};

/** Adds numbers together */
int add(int a, int b) {
    return a + b;
}
"""
    parser = TreeSitterParser()
    nodes = parser.parse(code, "cpp")

    class_node = next(n for n in nodes if n.node_type == "class")
    method_node = next(n for n in nodes if n.node_type == "method")
    func_node = next(n for n in nodes if n.node_type == "function")

    assert class_node.name == "Util"
    assert class_node.docstring and "Utility helpers" in class_node.docstring

    assert method_node.name == "compute"
    assert method_node.parent_name == "Util"
    assert method_node.docstring and "Computes a value" in method_node.docstring

    assert func_node.name == "add"
    assert func_node.docstring and "Adds numbers together" in func_node.docstring
