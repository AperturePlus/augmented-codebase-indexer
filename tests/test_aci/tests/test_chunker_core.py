"""Unit tests for chunker core components and import extraction."""

from pathlib import Path

import pytest

from aci.core.ast_parser import TreeSitterParser
from aci.core.chunker import (
    Chunker,
    ChunkerInterface,
    CodeChunk,
    GoImportExtractor,
    ImportExtractorRegistry,
    JavaScriptImportExtractor,
    PythonImportExtractor,
    create_chunker,
    get_import_registry,
)
from aci.core.file_scanner import ScannedFile
from aci.core.tokenizer import get_default_tokenizer


class TestCodeChunk:
    """Tests for CodeChunk dataclass."""

    def test_default_values(self):
        """Test that CodeChunk has sensible defaults."""
        chunk = CodeChunk()
        assert chunk.chunk_id
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
        content = """import os
from pathlib import Path
import sys

def main():
    pass
"""
        extractor = PythonImportExtractor()
        imports = extractor.extract(content)
        assert "import os" in imports
        assert "from pathlib import Path" in imports
        assert "import sys" in imports

    def test_javascript_import_extractor(self):
        """Test JavaScriptImportExtractor."""
        content = """import React from 'react';
import { useState } from 'react';

function App() {
    return null;
}
"""
        extractor = JavaScriptImportExtractor()
        imports = extractor.extract(content)
        assert len(imports) >= 1
        assert any("import" in imp for imp in imports)

    def test_go_import_extractor(self):
        """Test GoImportExtractor."""
        content = """package main

import (
    "fmt"
    "os"
)

func main() {}
"""
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

        result = chunker.chunk(file, ast_nodes)
        chunks = result.chunks

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

        result = chunker.chunk(file, ast_nodes)
        chunks = result.chunks

        assert len(chunks) == 3

        method_chunks = [c for c in chunks if c.chunk_type == "method"]
        assert len(method_chunks) == 2

        for method_chunk in method_chunks:
            assert method_chunk.metadata.get("parent_class") == "Calculator"
            assert method_chunk.metadata.get("function_name") in ["add", "subtract"]

    def test_chunk_fixed_size_fallback(self, chunker):
        """Test fixed-size chunking when no AST nodes."""
        content = "line1\nline2\nline3\nline4\nline5\n" * 20
        file = self._create_scanned_file(content, language="unknown")

        result = chunker.chunk(file, [])
        chunks = result.chunks

        assert len(chunks) >= 1
        assert all(c.chunk_type == "fixed" for c in chunks)

    def test_chunk_includes_file_hash(self, chunker, parser):
        """Test that chunks include file hash in metadata."""
        content = "def foo(): pass"
        file = self._create_scanned_file(content)
        ast_nodes = parser.parse(content, "python")

        result = chunker.chunk(file, ast_nodes)
        chunks = result.chunks

        assert len(chunks) == 1
        assert chunks[0].metadata.get("file_hash") == "test-hash"

    def test_chunk_includes_imports(self, chunker, parser):
        """Test that chunks include imports in metadata."""
        content = """import os
from pathlib import Path

def main():
    pass
"""
        file = self._create_scanned_file(content)
        ast_nodes = parser.parse(content, "python")

        result = chunker.chunk(file, ast_nodes)
        chunks = result.chunks

        assert len(chunks) == 1
        imports = chunks[0].metadata.get("imports", [])
        assert "import os" in imports
        assert "from pathlib import Path" in imports

    def test_set_max_tokens(self, chunker):
        """Test setting max tokens."""
        chunker.set_max_tokens(4096)
        assert chunker._max_tokens == 4096

    def test_get_config_returns_chunker_config(self, tokenizer):
        """Test get_config returns ChunkerConfig with current settings."""
        from aci.core.chunker import ChunkerConfig

        chunker = Chunker(
            tokenizer=tokenizer,
            max_tokens=4096,
            fixed_chunk_lines=100,
            overlap_lines=10,
        )
        config = chunker.get_config()

        assert isinstance(config, ChunkerConfig)
        assert config.max_tokens == 4096
        assert config.fixed_chunk_lines == 100
        assert config.overlap_lines == 10

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

        result = chunker.chunk(file, ast_nodes)
        chunks = result.chunks

        assert len(chunks) == 1
        chunk = chunks[0]

        lines = content.split("\n")
        extracted = "\n".join(lines[chunk.start_line - 1 : chunk.end_line])

        assert chunk.content.endswith(extracted)
        assert chunk.content.split("\n---\n")[0].strip() == "Docstring."


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
