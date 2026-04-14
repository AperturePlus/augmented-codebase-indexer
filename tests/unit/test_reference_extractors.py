"""
Unit tests for reference extractors — language-specific symbol reference extraction.

Tests cover Python, JavaScript, Go, Java, and C++ reference extraction including
calls, imports, type annotations, inheritance, and parent_symbol tracking.
"""

import pytest

from aci.core.ast_parser import TreeSitterParser
from aci.core.parsers.cpp_reference_extractor import CppReferenceExtractor
from aci.core.parsers.go_reference_extractor import GoReferenceExtractor
from aci.core.parsers.java_reference_extractor import JavaReferenceExtractor
from aci.core.parsers.javascript_reference_extractor import JavaScriptReferenceExtractor
from aci.core.parsers.python_reference_extractor import PythonReferenceExtractor

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


class TestPythonReferenceExtractor:
    """Tests for PythonReferenceExtractor."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    @pytest.fixture
    def extractor(self):
        return PythonReferenceExtractor()

    def test_extract_function_calls(self, parser, extractor):
        """Test foo(), obj.method(), a.b.c() → ref_type='call'."""
        code = """
foo()
obj.method()
a.b.c()
"""
        tree = parser.parse_tree(code, "python")
        refs = extractor.extract_references(tree.root_node, code, "test.py")
        call_names = {r.name for r in refs if r.ref_type == "call"}
        assert "foo" in call_names
        assert "obj.method" in call_names
        assert "a.b.c" in call_names

    def test_extract_imports(self, parser, extractor):
        """Test import os, from os import path, from . import module → ref_type='import'."""
        code = """
import os
from os import path
from . import module
"""
        tree = parser.parse_tree(code, "python")
        refs = extractor.extract_imports(tree.root_node, code, "test.py")
        import_names = [r.name for r in refs if r.ref_type == "import"]
        assert "os" in import_names
        assert "path" in import_names
        assert any("module" in n for n in import_names)

    def test_extract_type_annotations(self, parser, extractor):
        """Test def foo(x: int) -> str → ref_type='type_annotation'."""
        code = """
def foo(x: int) -> str:
    pass
"""
        tree = parser.parse_tree(code, "python")
        refs = extractor.extract_references(tree.root_node, code, "test.py")
        type_refs = [r for r in refs if r.ref_type == "type_annotation"]
        type_names = {r.name for r in type_refs}
        assert "int" in type_names
        assert "str" in type_names

    def test_extract_inheritance(self, parser, extractor):
        """Test class Foo(Bar, Baz) → ref_type='inheritance'."""
        code = """
class Foo(Bar, Baz):
    pass
"""
        tree = parser.parse_tree(code, "python")
        refs = extractor.extract_references(tree.root_node, code, "test.py")
        inh_names = {r.name for r in refs if r.ref_type == "inheritance"}
        assert "Bar" in inh_names
        assert "Baz" in inh_names

    def test_parent_symbol_tracking(self, parser, extractor):
        """Test that calls inside a class method have parent_symbol='ClassName.method_name'."""
        code = """
class MyClass:
    def my_method(self):
        helper()
"""
        tree = parser.parse_tree(code, "python")
        refs = extractor.extract_references(tree.root_node, code, "test.py")
        helper_refs = [r for r in refs if r.name == "helper" and r.ref_type == "call"]
        assert len(helper_refs) == 1
        assert helper_refs[0].parent_symbol == "MyClass.my_method"


# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------


class TestJavaScriptReferenceExtractor:
    """Tests for JavaScriptReferenceExtractor."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    @pytest.fixture
    def extractor(self):
        return JavaScriptReferenceExtractor()

    def test_extract_calls(self, parser, extractor):
        """Test foo(), obj.method(), new ClassName() → ref_type='call'."""
        code = """
foo();
obj.method();
new ClassName();
"""
        tree = parser.parse_tree(code, "javascript")
        refs = extractor.extract_references(tree.root_node, code, "test.js")
        call_names = {r.name for r in refs if r.ref_type == "call"}
        assert "foo" in call_names
        assert "obj.method" in call_names
        assert "ClassName" in call_names

    def test_extract_es6_imports(self, parser, extractor):
        """Test import { foo } from 'module', import bar from 'module' → ref_type='import'."""
        code = """
import { foo } from 'mymodule';
import bar from 'othermodule';
"""
        tree = parser.parse_tree(code, "javascript")
        refs = extractor.extract_imports(tree.root_node, code, "test.js")
        import_names = [r.name for r in refs if r.ref_type == "import"]
        assert "mymodule" in import_names
        assert "foo" in import_names
        assert "othermodule" in import_names
        assert "bar" in import_names

    def test_extract_commonjs_requires(self, parser, extractor):
        """Test require('module') → ref_type='import'."""
        code = """
const fs = require('fs');
"""
        tree = parser.parse_tree(code, "javascript")
        refs = extractor.extract_imports(tree.root_node, code, "test.js")
        import_names = [r.name for r in refs if r.ref_type == "import"]
        assert "fs" in import_names

    def test_extract_inheritance(self, parser, extractor):
        """Test class Foo extends Bar → ref_type='inheritance'."""
        code = """
class Foo extends Bar {
    constructor() {
        super();
    }
}
"""
        tree = parser.parse_tree(code, "javascript")
        refs = extractor.extract_references(tree.root_node, code, "test.js")
        inh_refs = [r for r in refs if r.ref_type == "inheritance"]
        inh_names = {r.name for r in inh_refs}
        assert "Bar" in inh_names

    def test_parent_symbol_tracking(self, parser, extractor):
        """Test parent_symbol inside class methods."""
        code = """
class MyClass {
    doWork() {
        helper();
    }
}
"""
        tree = parser.parse_tree(code, "javascript")
        refs = extractor.extract_references(tree.root_node, code, "test.js")
        helper_refs = [r for r in refs if r.name == "helper" and r.ref_type == "call"]
        assert len(helper_refs) == 1
        assert helper_refs[0].parent_symbol == "MyClass.doWork"


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------


class TestGoReferenceExtractor:
    """Tests for GoReferenceExtractor."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    @pytest.fixture
    def extractor(self):
        return GoReferenceExtractor()

    def test_extract_calls(self, parser, extractor):
        """Test foo(), pkg.Func() → ref_type='call'."""
        code = """
package main

func main() {
    foo()
    pkg.Func()
}
"""
        tree = parser.parse_tree(code, "go")
        refs = extractor.extract_references(tree.root_node, code, "test.go")
        call_names = {r.name for r in refs if r.ref_type == "call"}
        assert "foo" in call_names
        assert "pkg.Func" in call_names

    def test_extract_imports(self, parser, extractor):
        """Test import 'fmt', grouped imports → ref_type='import'."""
        code = '''
package main

import (
    "fmt"
    "os"
)
'''
        tree = parser.parse_tree(code, "go")
        refs = extractor.extract_imports(tree.root_node, code, "test.go")
        import_names = {r.name for r in refs if r.ref_type == "import"}
        assert "fmt" in import_names
        assert "os" in import_names

    def test_extract_struct_embedding(self, parser, extractor):
        """Test type Foo struct { Bar } → ref_type='inheritance'."""
        code = """
package main

type Foo struct {
    Bar
}
"""
        tree = parser.parse_tree(code, "go")
        refs = extractor.extract_references(tree.root_node, code, "test.go")
        inh_refs = [r for r in refs if r.ref_type == "inheritance"]
        inh_names = {r.name for r in inh_refs}
        assert "Bar" in inh_names

    def test_extract_type_refs(self, parser, extractor):
        """Test function parameter/return types → ref_type='type_annotation'."""
        code = """
package main

func process(input MyType) (Result, error) {
    return Result{}, nil
}
"""
        tree = parser.parse_tree(code, "go")
        refs = extractor.extract_references(tree.root_node, code, "test.go")
        type_refs = [r for r in refs if r.ref_type == "type_annotation"]
        type_names = {r.name for r in type_refs}
        assert "MyType" in type_names
        assert "Result" in type_names

    def test_parent_symbol_tracking(self, parser, extractor):
        """Test parent_symbol for method receivers."""
        code = """
package main

type Server struct{}

func (s *Server) Handle() {
    process()
}
"""
        tree = parser.parse_tree(code, "go")
        refs = extractor.extract_references(tree.root_node, code, "test.go")
        process_refs = [r for r in refs if r.name == "process" and r.ref_type == "call"]
        assert len(process_refs) == 1
        assert process_refs[0].parent_symbol == "Server.Handle"


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------


class TestJavaReferenceExtractor:
    """Tests for JavaReferenceExtractor."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    @pytest.fixture
    def extractor(self):
        return JavaReferenceExtractor()

    def test_extract_calls(self, parser, extractor):
        """Test foo(), obj.method(), new ClassName() → ref_type='call'."""
        code = """
public class App {
    void run() {
        foo();
        obj.method();
        new ClassName();
    }
}
"""
        tree = parser.parse_tree(code, "java")
        refs = extractor.extract_references(tree.root_node, code, "Test.java")
        call_names = {r.name for r in refs if r.ref_type == "call"}
        assert "foo" in call_names
        assert "method" in call_names
        assert "ClassName" in call_names

    def test_extract_imports(self, parser, extractor):
        """Test import java.util.List; → ref_type='import'."""
        code = """
import java.util.List;
import java.io.File;

public class App {}
"""
        tree = parser.parse_tree(code, "java")
        refs = extractor.extract_imports(tree.root_node, code, "Test.java")
        import_names = {r.name for r in refs if r.ref_type == "import"}
        assert "java.util.List" in import_names
        assert "java.io.File" in import_names

    def test_extract_inheritance(self, parser, extractor):
        """Test class Foo extends Bar implements Baz → ref_type='inheritance'."""
        code = """
public class Foo extends Bar implements Baz {
}
"""
        tree = parser.parse_tree(code, "java")
        refs = extractor.extract_references(tree.root_node, code, "Test.java")
        inh_names = {r.name for r in refs if r.ref_type == "inheritance"}
        assert "Bar" in inh_names
        assert "Baz" in inh_names

    def test_extract_type_annotations(self, parser, extractor):
        """Test method param types, return types → ref_type='type_annotation'."""
        code = """
public class App {
    String process(MyType input) {
        return "";
    }
}
"""
        tree = parser.parse_tree(code, "java")
        refs = extractor.extract_references(tree.root_node, code, "Test.java")
        type_refs = [r for r in refs if r.ref_type == "type_annotation"]
        type_names = {r.name for r in type_refs}
        assert "String" in type_names
        assert "MyType" in type_names

    def test_parent_symbol_tracking(self, parser, extractor):
        """Test parent_symbol inside class methods."""
        code = """
public class MyService {
    void handle() {
        helper();
    }
}
"""
        tree = parser.parse_tree(code, "java")
        refs = extractor.extract_references(tree.root_node, code, "Test.java")
        helper_refs = [r for r in refs if r.name == "helper" and r.ref_type == "call"]
        assert len(helper_refs) == 1
        assert helper_refs[0].parent_symbol == "MyService.handle"


# ---------------------------------------------------------------------------
# C++
# ---------------------------------------------------------------------------


class TestCppReferenceExtractor:
    """Tests for CppReferenceExtractor."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    @pytest.fixture
    def extractor(self):
        return CppReferenceExtractor()

    def test_extract_calls(self, parser, extractor):
        """Test foo(), obj.method(), ns::func() → ref_type='call'."""
        code = """
void run() {
    foo();
    obj.method();
    ns::func();
}
"""
        tree = parser.parse_tree(code, "cpp")
        refs = extractor.extract_references(tree.root_node, code, "test.cpp")
        call_names = {r.name for r in refs if r.ref_type == "call"}
        assert "foo" in call_names
        assert "obj.method" in call_names
        assert "ns::func" in call_names

    def test_extract_includes(self, parser, extractor):
        """Test #include <iostream>, #include 'myheader.h' → ref_type='import'."""
        code = """
#include <iostream>
#include "myheader.h"
"""
        tree = parser.parse_tree(code, "cpp")
        refs = extractor.extract_imports(tree.root_node, code, "test.cpp")
        import_names = {r.name for r in refs if r.ref_type == "import"}
        assert "iostream" in import_names
        assert "myheader.h" in import_names

    def test_extract_inheritance(self, parser, extractor):
        """Test class Foo : public Bar → ref_type='inheritance'."""
        code = """
class Foo : public Bar {
};
"""
        tree = parser.parse_tree(code, "cpp")
        refs = extractor.extract_references(tree.root_node, code, "test.cpp")
        inh_names = {r.name for r in refs if r.ref_type == "inheritance"}
        assert "Bar" in inh_names

    def test_extract_type_annotations(self, parser, extractor):
        """Test function param types, return types → ref_type='type_annotation'."""
        code = """
MyResult process(MyInput input) {
    return MyResult();
}
"""
        tree = parser.parse_tree(code, "cpp")
        refs = extractor.extract_references(tree.root_node, code, "test.cpp")
        type_refs = [r for r in refs if r.ref_type == "type_annotation"]
        type_names = {r.name for r in type_refs}
        assert "MyResult" in type_names
        assert "MyInput" in type_names

    def test_parent_symbol_tracking(self, parser, extractor):
        """Test parent_symbol inside class methods."""
        code = """
class MyClass {
    void doWork() {
        helper();
    }
};
"""
        tree = parser.parse_tree(code, "cpp")
        refs = extractor.extract_references(tree.root_node, code, "test.cpp")
        helper_refs = [r for r in refs if r.name == "helper" and r.ref_type == "call"]
        assert len(helper_refs) == 1
        assert helper_refs[0].parent_symbol == "MyClass.doWork"
