"""
Unit tests for AST Parser - Tree-sitter based code structure parsing.

Tests cover Python, JavaScript/TypeScript, and Go language parsing.
"""

import pytest
from aci.core.ast_parser import TreeSitterParser, ASTNode, SUPPORTED_LANGUAGES


class TestTreeSitterParser:
    """Tests for TreeSitterParser class."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance for tests."""
        return TreeSitterParser()

    def test_supports_language(self, parser):
        """Test language support detection."""
        assert parser.supports_language("python")
        assert parser.supports_language("javascript")
        assert parser.supports_language("typescript")
        assert parser.supports_language("go")
        assert not parser.supports_language("rust")
        assert not parser.supports_language("unknown")

    def test_parse_unsupported_language(self, parser):
        """Test parsing unsupported language returns empty list."""
        result = parser.parse("some code", "rust")
        assert result == []


class TestPythonParsing:
    """Tests for Python language parsing."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    def test_extract_standalone_functions(self, parser):
        """Test extraction of standalone Python functions."""
        code = '''
def standalone_function():
    """A standalone function."""
    pass

def another_function(arg1, arg2):
    """Function with arguments."""
    return arg1 + arg2
'''
        nodes = parser.parse(code, "python")
        functions = [n for n in nodes if n.node_type == "function"]

        assert len(functions) == 2
        func_names = {f.name for f in functions}
        assert func_names == {"standalone_function", "another_function"}

    def test_extract_classes(self, parser):
        """Test extraction of Python class definitions."""
        code = '''
class MyClass:
    """A sample class."""
    pass

class AnotherClass:
    pass
'''
        nodes = parser.parse(code, "python")
        classes = [n for n in nodes if n.node_type == "class"]

        assert len(classes) == 2
        class_names = {c.name for c in classes}
        assert class_names == {"MyClass", "AnotherClass"}

    def test_extract_methods_with_parent(self, parser):
        """Test extraction of methods with correct parent class."""
        code = '''
class MyClass:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value
'''
        nodes = parser.parse(code, "python")
        methods = [n for n in nodes if n.node_type == "method"]

        assert len(methods) == 3
        for method in methods:
            assert method.parent_name == "MyClass"

        method_names = {m.name for m in methods}
        assert method_names == {"__init__", "get_value", "set_value"}

    def test_extract_docstrings(self, parser):
        """Test extraction of Python docstrings."""
        code = '''
def func_with_doc():
    """This is a docstring."""
    pass

class ClassWithDoc:
    """Class docstring."""
    pass
'''
        nodes = parser.parse(code, "python")

        func = next(n for n in nodes if n.name == "func_with_doc")
        assert func.docstring == "This is a docstring."

        cls = next(n for n in nodes if n.name == "ClassWithDoc")
        assert cls.docstring == "Class docstring."

    def test_line_numbers_accuracy(self, parser):
        """Test that line numbers are accurate (1-based)."""
        code = '''def first():
    pass

def second():
    pass
'''
        nodes = parser.parse(code, "python")

        first = next(n for n in nodes if n.name == "first")
        assert first.start_line == 1
        assert first.end_line == 2

        second = next(n for n in nodes if n.name == "second")
        assert second.start_line == 4
        assert second.end_line == 5

    def test_full_python_parsing(self, parser):
        """Comprehensive test for Python parsing."""
        code = '''
def standalone_function():
    """A standalone function."""
    pass

def another_function(arg1, arg2):
    """Function with arguments."""
    return arg1 + arg2

class MyClass:
    """A sample class."""

    def __init__(self, value):
        """Initialize the class."""
        self.value = value

    def get_value(self):
        """Get the value."""
        return self.value

    def set_value(self, value):
        """Set the value."""
        self.value = value

class AnotherClass:
    def method_one(self):
        pass
'''
        nodes = parser.parse(code, "python")

        functions = [n for n in nodes if n.node_type == "function"]
        classes = [n for n in nodes if n.node_type == "class"]
        methods = [n for n in nodes if n.node_type == "method"]

        assert len(functions) == 2
        assert len(classes) == 2
        assert len(methods) == 4

        # Check MyClass methods have correct parent
        myclass_methods = [m for m in methods if m.parent_name == "MyClass"]
        assert len(myclass_methods) == 3


class TestJavaScriptParsing:
    """Tests for JavaScript/TypeScript language parsing."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    def test_extract_function_declarations(self, parser):
        """Test extraction of JavaScript function declarations."""
        code = '''
function greet(name) {
    return "Hello, " + name;
}

function anotherFunction() {
    console.log("test");
}
'''
        nodes = parser.parse(code, "javascript")
        functions = [n for n in nodes if n.node_type == "function"]

        assert len(functions) == 2
        func_names = {f.name for f in functions}
        assert func_names == {"greet", "anotherFunction"}

    def test_extract_arrow_functions(self, parser):
        """Test extraction of arrow functions in variable declarations."""
        code = '''
const sayHi = () => {
    console.log("Hi!");
};

const arrowWithBody = (x, y) => {
    return x + y;
};
'''
        nodes = parser.parse(code, "javascript")
        functions = [n for n in nodes if n.node_type == "function"]

        assert len(functions) == 2
        func_names = {f.name for f in functions}
        assert func_names == {"sayHi", "arrowWithBody"}

    def test_extract_classes_and_methods(self, parser):
        """Test extraction of JavaScript classes and methods."""
        code = '''
class Greeter {
    constructor(name) {
        this.name = name;
    }

    greet() {
        return "Hello, " + this.name;
    }

    static create(name) {
        return new Greeter(name);
    }
}
'''
        nodes = parser.parse(code, "javascript")

        classes = [n for n in nodes if n.node_type == "class"]
        assert len(classes) == 1
        assert classes[0].name == "Greeter"

        methods = [n for n in nodes if n.node_type == "method"]
        assert len(methods) == 3

        for method in methods:
            assert method.parent_name == "Greeter"

        method_names = {m.name for m in methods}
        assert method_names == {"constructor", "greet", "create"}

    def test_typescript_parsing(self, parser):
        """Test TypeScript parsing (uses same parser as JavaScript)."""
        code = '''
function add(a, b) {
    return a + b;
}

class Calculator {
    add(a, b) {
        return a + b;
    }
}
'''
        nodes = parser.parse(code, "typescript")

        assert len(nodes) >= 3  # function, class, method

        functions = [n for n in nodes if n.node_type == "function"]
        classes = [n for n in nodes if n.node_type == "class"]
        methods = [n for n in nodes if n.node_type == "method"]

        assert len(functions) == 1
        assert len(classes) == 1
        assert len(methods) == 1


class TestGoParsing:
    """Tests for Go language parsing."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    def test_extract_functions(self, parser):
        """Test extraction of Go function declarations."""
        code = '''
package main

func hello() {
    fmt.Println("Hello!")
}

func add(a, b int) int {
    return a + b
}
'''
        nodes = parser.parse(code, "go")
        functions = [n for n in nodes if n.node_type == "function"]

        assert len(functions) == 2
        func_names = {f.name for f in functions}
        assert func_names == {"hello", "add"}

    def test_extract_structs(self, parser):
        """Test extraction of Go struct type declarations."""
        code = '''
package main

type Greeter struct {
    Name string
    Age  int
}

type Calculator struct {
    Value int
}
'''
        nodes = parser.parse(code, "go")
        # Structs are represented as 'class' for consistency
        structs = [n for n in nodes if n.node_type == "class"]

        assert len(structs) == 2
        struct_names = {s.name for s in structs}
        assert struct_names == {"Greeter", "Calculator"}

    def test_extract_methods_with_receiver(self, parser):
        """Test extraction of Go methods with correct receiver type."""
        code = '''
package main

type Greeter struct {
    Name string
}

func (g *Greeter) Greet() string {
    return "Hello, " + g.Name
}

func (g Greeter) GetName() string {
    return g.Name
}
'''
        nodes = parser.parse(code, "go")
        methods = [n for n in nodes if n.node_type == "method"]

        assert len(methods) == 2

        for method in methods:
            assert method.parent_name == "Greeter"

        method_names = {m.name for m in methods}
        assert method_names == {"Greet", "GetName"}

    def test_full_go_parsing(self, parser):
        """Comprehensive test for Go parsing."""
        code = '''
package main

import "fmt"

func hello() {
    fmt.Println("Hello!")
}

func add(a, b int) int {
    return a + b
}

type Greeter struct {
    Name string
    Age  int
}

func (g *Greeter) Greet() string {
    return "Hello, " + g.Name
}

func (g Greeter) GetAge() int {
    return g.Age
}

type Calculator struct {
    Value int
}

func (c *Calculator) Add(x int) {
    c.Value += x
}
'''
        nodes = parser.parse(code, "go")

        functions = [n for n in nodes if n.node_type == "function"]
        structs = [n for n in nodes if n.node_type == "class"]
        methods = [n for n in nodes if n.node_type == "method"]

        assert len(functions) == 2
        assert len(structs) == 2
        assert len(methods) == 3

        # Check receiver types
        greeter_methods = [m for m in methods if m.parent_name == "Greeter"]
        assert len(greeter_methods) == 2

        calculator_methods = [m for m in methods if m.parent_name == "Calculator"]
        assert len(calculator_methods) == 1
