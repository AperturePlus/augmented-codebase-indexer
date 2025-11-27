"""
Tests for the Chunker module.

Tests cover:
- AST-based semantic chunking
- Fixed-size chunking fallback
- Metadata extraction (function_name, parent_class, imports)
- Token limit enforcement
- SmartChunkSplitter for intelligent oversized node splitting
"""

import pytest
from pathlib import Path

from aci.core.chunker import (
    CodeChunk,
    ChunkerInterface,
    Chunker,
    create_chunker,
    ImportExtractorRegistry,
    PythonImportExtractor,
    JavaScriptImportExtractor,
    GoImportExtractor,
    get_import_registry,
    SmartChunkSplitter,
)
from aci.core.file_scanner import ScannedFile
from aci.core.ast_parser import ASTNode, TreeSitterParser
from aci.core.tokenizer import get_default_tokenizer


class TestCodeChunk:
    """Tests for CodeChunk dataclass."""
    
    def test_default_values(self):
        """Test that CodeChunk has sensible defaults."""
        chunk = CodeChunk()
        assert chunk.chunk_id  # Should have a UUID
        assert chunk.file_path == ""
        assert chunk.start_line == 0
        assert chunk.end_line == 0
        assert chunk.content == ""
        assert chunk.language == ""
        assert chunk.chunk_type == ""
        assert chunk.metadata == {}
    
    def test_custom_values(self):
        """Test CodeChunk with custom values."""
        chunk = CodeChunk(
            chunk_id="test-id",
            file_path="/path/to/file.py",
            start_line=10,
            end_line=20,
            content="def foo(): pass",
            language="python",
            chunk_type="function",
            metadata={"function_name": "foo"},
        )
        assert chunk.chunk_id == "test-id"
        assert chunk.file_path == "/path/to/file.py"
        assert chunk.start_line == 10
        assert chunk.end_line == 20
        assert chunk.content == "def foo(): pass"
        assert chunk.language == "python"
        assert chunk.chunk_type == "function"
        assert chunk.metadata == {"function_name": "foo"}


class TestImportExtractors:
    """Tests for import extraction using the registry pattern."""
    
    def test_python_import_extractor(self):
        """Test PythonImportExtractor."""
        content = '''import os
from pathlib import Path
import sys

def main():
    pass
'''
        extractor = PythonImportExtractor()
        imports = extractor.extract(content)
        assert "import os" in imports
        assert "from pathlib import Path" in imports
        assert "import sys" in imports
    
    def test_javascript_import_extractor(self):
        """Test JavaScriptImportExtractor."""
        content = '''import React from 'react';
import { useState } from 'react';

function App() {
    return null;
}
'''
        extractor = JavaScriptImportExtractor()
        imports = extractor.extract(content)
        assert len(imports) >= 1
        assert any("import" in imp for imp in imports)
    
    def test_go_import_extractor(self):
        """Test GoImportExtractor."""
        content = '''package main

import (
    "fmt"
    "os"
)

func main() {}
'''
        extractor = GoImportExtractor()
        imports = extractor.extract(content)
        assert "fmt" in imports
        assert "os" in imports
    
    def test_registry_returns_correct_extractor(self):
        """Test that registry returns correct extractor for each language."""
        registry = get_import_registry()
        
        assert isinstance(registry.get("python"), PythonImportExtractor)
        assert isinstance(registry.get("javascript"), JavaScriptImportExtractor)
        assert isinstance(registry.get("typescript"), JavaScriptImportExtractor)
        assert isinstance(registry.get("go"), GoImportExtractor)
    
    def test_registry_extract_imports(self):
        """Test registry's extract_imports convenience method."""
        registry = get_import_registry()
        content = "import os\nimport sys\n\ndef main(): pass"
        
        imports = registry.extract_imports(content, "python")
        assert "import os" in imports
        assert "import sys" in imports
    
    def test_registry_unknown_language_returns_empty(self):
        """Test that unknown languages return empty imports."""
        registry = get_import_registry()
        imports = registry.extract_imports("some content", "unknown_lang")
        assert imports == []
    
    def test_custom_registry(self):
        """Test creating a custom registry with custom extractors."""
        registry = ImportExtractorRegistry()
        registry.register("python", PythonImportExtractor())
        
        imports = registry.extract_imports("import os", "python")
        assert "import os" in imports


class TestChunker:
    """Tests for Chunker implementation."""
    
    @pytest.fixture
    def tokenizer(self):
        return get_default_tokenizer()
    
    @pytest.fixture
    def chunker(self, tokenizer):
        return Chunker(tokenizer=tokenizer, max_tokens=8192)
    
    @pytest.fixture
    def parser(self):
        return TreeSitterParser()
    
    def _create_scanned_file(
        self,
        content: str,
        language: str = "python",
        path: str = "/test/file.py",
    ) -> ScannedFile:
        """Helper to create a ScannedFile for testing."""
        return ScannedFile(
            path=Path(path),
            content=content,
            language=language,
            size_bytes=len(content),
            modified_time=0.0,
            content_hash="test-hash",
        )
    
    def test_implements_interface(self, chunker):
        """Test that Chunker implements ChunkerInterface."""
        assert isinstance(chunker, ChunkerInterface)
    
    def test_chunk_with_ast_nodes(self, chunker, parser):
        """Test chunking with AST nodes."""
        content = '''def hello():
    """Say hello."""
    print("Hello, World!")

def goodbye():
    print("Goodbye!")
'''
        file = self._create_scanned_file(content)
        ast_nodes = parser.parse(content, "python")
        
        chunks = chunker.chunk(file, ast_nodes)
        
        assert len(chunks) == 2
        assert all(isinstance(c, CodeChunk) for c in chunks)
        assert chunks[0].chunk_type == "function"
        assert chunks[0].metadata.get("function_name") == "hello"
        assert chunks[1].metadata.get("function_name") == "goodbye"
    
    def test_chunk_with_class_and_methods(self, chunker, parser):
        """Test chunking extracts class and method metadata."""
        content = '''class Calculator:
    """A simple calculator."""
    
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
'''
        file = self._create_scanned_file(content)
        ast_nodes = parser.parse(content, "python")
        
        chunks = chunker.chunk(file, ast_nodes)
        
        # Should have class + 2 methods
        assert len(chunks) == 3
        
        # Find method chunks
        method_chunks = [c for c in chunks if c.chunk_type == "method"]
        assert len(method_chunks) == 2
        
        for method_chunk in method_chunks:
            assert method_chunk.metadata.get("parent_class") == "Calculator"
            assert method_chunk.metadata.get("function_name") in ["add", "subtract"]
    
    def test_chunk_fixed_size_fallback(self, chunker):
        """Test fixed-size chunking when no AST nodes."""
        content = "line1\nline2\nline3\nline4\nline5\n" * 20
        file = self._create_scanned_file(content, language="unknown")
        
        chunks = chunker.chunk(file, [])  # No AST nodes
        
        assert len(chunks) >= 1
        assert all(c.chunk_type == "fixed" for c in chunks)
    
    def test_chunk_includes_file_hash(self, chunker, parser):
        """Test that chunks include file hash in metadata."""
        content = "def foo(): pass"
        file = self._create_scanned_file(content)
        ast_nodes = parser.parse(content, "python")
        
        chunks = chunker.chunk(file, ast_nodes)
        
        assert len(chunks) == 1
        assert chunks[0].metadata.get("file_hash") == "test-hash"
    
    def test_chunk_includes_imports(self, chunker, parser):
        """Test that chunks include imports in metadata."""
        content = '''import os
from pathlib import Path

def main():
    pass
'''
        file = self._create_scanned_file(content)
        ast_nodes = parser.parse(content, "python")
        
        chunks = chunker.chunk(file, ast_nodes)
        
        assert len(chunks) == 1
        imports = chunks[0].metadata.get("imports", [])
        assert "import os" in imports
        assert "from pathlib import Path" in imports
    
    def test_set_max_tokens(self, chunker):
        """Test setting max tokens."""
        chunker.set_max_tokens(4096)
        assert chunker._max_tokens == 4096
    
    def test_line_numbers_accuracy(self, chunker, parser):
        """Test that chunk line numbers match content."""
        content = '''# Comment line 1
# Comment line 2

def hello():
    """Docstring."""
    print("Hello")
    return True

# More comments
'''
        file = self._create_scanned_file(content)
        ast_nodes = parser.parse(content, "python")
        
        chunks = chunker.chunk(file, ast_nodes)
        
        assert len(chunks) == 1
        chunk = chunks[0]
        
        # Verify line numbers by extracting from original content
        lines = content.split("\n")
        extracted = "\n".join(lines[chunk.start_line - 1:chunk.end_line])
        
        assert chunk.content == extracted


class TestCreateChunker:
    """Tests for create_chunker factory function."""
    
    def test_creates_chunker_with_defaults(self):
        """Test factory creates chunker with default settings."""
        chunker = create_chunker()
        assert isinstance(chunker, Chunker)
        assert chunker._max_tokens == 8192
    
    def test_creates_chunker_with_custom_settings(self):
        """Test factory creates chunker with custom settings."""
        chunker = create_chunker(
            max_tokens=4096,
            fixed_chunk_lines=100,
            overlap_lines=10,
        )
        assert chunker._max_tokens == 4096
        assert chunker._fixed_chunk_lines == 100
        assert chunker._overlap_lines == 10


class TestSmartChunkSplitter:
    """Tests for SmartChunkSplitter implementation."""
    
    @pytest.fixture
    def tokenizer(self):
        return get_default_tokenizer()
    
    @pytest.fixture
    def splitter(self, tokenizer):
        return SmartChunkSplitter(tokenizer)
    
    def _create_ast_node(
        self,
        content: str,
        name: str = "test_func",
        node_type: str = "function",
        start_line: int = 1,
        parent_name: str = None,
    ) -> ASTNode:
        """Helper to create an ASTNode for testing."""
        lines = content.split("\n")
        end_line = start_line + len(lines) - 1
        return ASTNode(
            node_type=node_type,
            name=name,
            start_line=start_line,
            end_line=end_line,
            content=content,
            parent_name=parent_name,
        )
    
    def test_split_at_empty_lines(self, splitter, tokenizer):
        """Test that splitter prefers empty lines as split points."""
        # Create content with clear empty line boundaries
        content = '''def process_data():
    x = 1
    y = 2

    z = 3
    w = 4

    return x + y + z + w'''
        
        node = self._create_ast_node(content, name="process_data")
        
        # Use a very small token limit to force splitting
        chunks = splitter.split_oversized_node(
            node=node,
            max_tokens=30,  # Very small to force splits
            file_path="/test/file.py",
            language="python",
            base_metadata={"file_hash": "test"},
        )
        
        # Should produce multiple chunks
        assert len(chunks) >= 1
        # All chunks should have correct metadata
        for chunk in chunks:
            assert chunk.metadata.get("function_name") == "process_data"
    
    def test_context_prefix_for_methods(self, splitter, tokenizer):
        """Test that methods get class context prefix in continuation chunks."""
        # Create a large method content with enough lines to force multiple chunks
        lines = ["        result += {}".format(i) for i in range(30)]
        content = '''def calculate(self, a, b):
    result = 0
''' + "\n".join(lines) + '''
    return result'''
        
        node = self._create_ast_node(
            content,
            name="calculate",
            node_type="method",
            parent_name="Calculator",
        )
        
        chunks = splitter.split_oversized_node(
            node=node,
            max_tokens=100,  # Small enough to force splits but large enough for context
            file_path="/test/file.py",
            language="python",
            base_metadata={"file_hash": "test"},
        )
        
        # If there are multiple chunks, continuation chunks should have context
        if len(chunks) > 1:
            for i, chunk in enumerate(chunks):
                if i > 0:
                    # Continuation chunks should have context prefix marker
                    assert chunk.metadata.get("has_context_prefix") == True
                    # Content should start with context comment
                    assert "# Context:" in chunk.content
                    assert "Calculator" in chunk.content
    
    def test_line_numbers_accuracy(self, splitter, tokenizer):
        """Test that split chunks have accurate line numbers."""
        content = '''def long_function():
    line1 = 1
    line2 = 2
    line3 = 3

    line4 = 4
    line5 = 5
    line6 = 6

    return line1 + line6'''
        
        node = self._create_ast_node(content, name="long_function", start_line=10)
        
        chunks = splitter.split_oversized_node(
            node=node,
            max_tokens=50,
            file_path="/test/file.py",
            language="python",
            base_metadata={},
        )
        
        # Verify line numbers are within the original node's range
        for chunk in chunks:
            assert chunk.start_line >= node.start_line
            assert chunk.end_line <= node.end_line
            assert chunk.start_line <= chunk.end_line
    
    def test_partial_metadata(self, splitter, tokenizer):
        """Test that split chunks have correct partial metadata."""
        # Create content large enough to require splitting
        lines = ["    line{}".format(i) for i in range(50)]
        content = "def big_func():\n" + "\n".join(lines)
        
        node = self._create_ast_node(content, name="big_func")
        
        chunks = splitter.split_oversized_node(
            node=node,
            max_tokens=100,  # Force multiple chunks
            file_path="/test/file.py",
            language="python",
            base_metadata={},
        )
        
        if len(chunks) > 1:
            # All chunks should be marked as partial
            for i, chunk in enumerate(chunks):
                assert chunk.metadata.get("is_partial") == True
                assert chunk.metadata.get("part_index") == i
                assert chunk.metadata.get("total_parts") == len(chunks)
    
    def test_single_chunk_no_partial_markers(self, splitter, tokenizer):
        """Test that single chunks don't have partial markers."""
        content = '''def small_func():
    return 42'''
        
        node = self._create_ast_node(content, name="small_func")
        
        chunks = splitter.split_oversized_node(
            node=node,
            max_tokens=1000,  # Large enough to fit everything
            file_path="/test/file.py",
            language="python",
            base_metadata={},
        )
        
        assert len(chunks) == 1
        assert "is_partial" not in chunks[0].metadata
        assert "part_index" not in chunks[0].metadata
        assert "total_parts" not in chunks[0].metadata
    
    def test_statement_boundary_detection(self, splitter):
        """Test that statement boundaries are correctly detected."""
        # Test various statement patterns
        assert splitter._is_statement_boundary("def foo():") == True
        assert splitter._is_statement_boundary("class Bar:") == True
        assert splitter._is_statement_boundary("    if x > 0:") == True
        assert splitter._is_statement_boundary("    for i in range(10):") == True
        assert splitter._is_statement_boundary("    return x") == True
        assert splitter._is_statement_boundary("    @decorator") == True
        
        # Non-boundaries
        assert splitter._is_statement_boundary("    x = 1") == False
        assert splitter._is_statement_boundary("    print(x)") == False
    
    def test_indentation_detection(self, splitter):
        """Test indentation level detection."""
        assert splitter._get_indentation("def foo():") == 0
        assert splitter._get_indentation("    x = 1") == 4
        assert splitter._get_indentation("        y = 2") == 8
        assert splitter._get_indentation("\tx = 1") == 1
        assert splitter._get_indentation("") == 0


class TestChunkerWithSmartSplitter:
    """Integration tests for Chunker using SmartChunkSplitter."""
    
    @pytest.fixture
    def tokenizer(self):
        return get_default_tokenizer()
    
    @pytest.fixture
    def parser(self):
        return TreeSitterParser()
    
    def _create_scanned_file(
        self,
        content: str,
        language: str = "python",
        path: str = "/test/file.py",
    ) -> ScannedFile:
        """Helper to create a ScannedFile for testing."""
        return ScannedFile(
            path=Path(path),
            content=content,
            language=language,
            size_bytes=len(content),
            modified_time=0.0,
            content_hash="test-hash",
        )
    
    def test_oversized_function_uses_smart_splitter(self, tokenizer, parser):
        """Test that oversized functions are split using SmartChunkSplitter."""
        # Create a function that exceeds token limit
        lines = ["    x{} = {}".format(i, i) for i in range(100)]
        content = "def big_function():\n" + "\n".join(lines) + "\n    return sum([x0])"
        
        file = self._create_scanned_file(content)
        ast_nodes = parser.parse(content, "python")
        
        # Use small token limit to force splitting
        chunker = Chunker(tokenizer=tokenizer, max_tokens=200)
        chunks = chunker.chunk(file, ast_nodes)
        
        # Should produce multiple chunks
        assert len(chunks) > 1
        
        # All chunks should reference the same function
        for chunk in chunks:
            assert chunk.metadata.get("function_name") == "big_function"
            assert chunk.chunk_type == "function"
    
    def test_oversized_method_preserves_class_context(self, tokenizer, parser):
        """Test that oversized methods preserve class context."""
        # Create a class with a large method
        lines = ["        self.x{} = {}".format(i, i) for i in range(100)]
        content = '''class BigClass:
    def big_method(self):
''' + "\n".join(lines)
        
        file = self._create_scanned_file(content)
        ast_nodes = parser.parse(content, "python")
        
        # Filter to just the method node
        method_nodes = [n for n in ast_nodes if n.node_type == "method"]
        
        if method_nodes:
            chunker = Chunker(tokenizer=tokenizer, max_tokens=200)
            chunks = chunker.chunk(file, method_nodes)
            
            # Check that class context is preserved
            for chunk in chunks:
                assert chunk.metadata.get("parent_class") == "BigClass"
