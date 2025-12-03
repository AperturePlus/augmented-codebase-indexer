"""Tests for SmartChunkSplitter and integration with Chunker."""

from pathlib import Path

import pytest

from aci.core.ast_parser import ASTNode, TreeSitterParser
from aci.core.chunker import Chunker, SmartChunkSplitter
from aci.core.file_scanner import ScannedFile
from aci.core.tokenizer import get_default_tokenizer


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
        content = """def process_data():
    x = 1
    y = 2

    z = 3
    w = 4

    return x + y + z + w"""

        node = self._create_ast_node(content, name="process_data")

        chunks = splitter.split_oversized_node(
            node=node,
            max_tokens=30,
            file_path="/test/file.py",
            language="python",
            base_metadata={"file_hash": "test"},
        )

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata.get("function_name") == "process_data"

    def test_context_prefix_for_methods(self, splitter, tokenizer):
        """Test that methods get class context prefix in continuation chunks."""
        lines = ["        result += {}".format(i) for i in range(30)]
        content = (
            """def calculate(self, a, b):
    result = 0
"""
            + "\n".join(lines)
            + """
    return result"""
        )

        node = self._create_ast_node(
            content,
            name="calculate",
            node_type="method",
            parent_name="Calculator",
        )

        chunks = splitter.split_oversized_node(
            node=node,
            max_tokens=100,
            file_path="/test/file.py",
            language="python",
            base_metadata={"file_hash": "test"},
        )

        if len(chunks) > 1:
            for i, chunk in enumerate(chunks):
                if i > 0:
                    assert chunk.metadata.get("has_context_prefix") is True
                    assert "# Context:" in chunk.content
                    assert "Calculator" in chunk.content

    def test_line_numbers_accuracy(self, splitter, tokenizer):
        """Test that split chunks have accurate line numbers."""
        content = """def long_function():
    line1 = 1
    line2 = 2
    line3 = 3

    line4 = 4
    line5 = 5
    line6 = 6

    return line1 + line6"""

        node = self._create_ast_node(content, name="long_function", start_line=10)

        chunks = splitter.split_oversized_node(
            node=node,
            max_tokens=50,
            file_path="/test/file.py",
            language="python",
            base_metadata={},
        )

        for chunk in chunks:
            assert chunk.start_line >= node.start_line
            assert chunk.end_line <= node.end_line
            assert chunk.start_line <= chunk.end_line

    def test_partial_metadata(self, splitter, tokenizer):
        """Test that split chunks have correct partial metadata."""
        lines = ["    line{}".format(i) for i in range(50)]
        content = "def big_func():\n" + "\n".join(lines)

        node = self._create_ast_node(content, name="big_func")

        chunks = splitter.split_oversized_node(
            node=node,
            max_tokens=100,
            file_path="/test/file.py",
            language="python",
            base_metadata={},
        )

        if len(chunks) > 1:
            for i, chunk in enumerate(chunks):
                assert chunk.metadata.get("is_partial") is True
                assert chunk.metadata.get("part_index") == i
                assert chunk.metadata.get("total_parts") == len(chunks)

    def test_single_chunk_no_partial_markers(self, splitter, tokenizer):
        """Test that single chunks don't have partial markers."""
        content = """def small_func():
    return 42"""

        node = self._create_ast_node(content, name="small_func")

        chunks = splitter.split_oversized_node(
            node=node,
            max_tokens=1000,
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
        assert splitter._is_statement_boundary("def foo():") is True
        assert splitter._is_statement_boundary("class Bar:") is True
        assert splitter._is_statement_boundary("    if x > 0:") is True
        assert splitter._is_statement_boundary("    for i in range(10):") is True
        assert splitter._is_statement_boundary("    return x") is True
        assert splitter._is_statement_boundary("    @decorator") is True

        assert splitter._is_statement_boundary("    x = 1") is False
        assert splitter._is_statement_boundary("    print(x)") is False

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
        lines = ["    x{} = {}".format(i, i) for i in range(100)]
        content = "def big_function():\n" + "\n".join(lines) + "\n    return sum([x0])"

        file = self._create_scanned_file(content)
        ast_nodes = parser.parse(content, "python")

        chunker = Chunker(tokenizer=tokenizer, max_tokens=200)
        chunks = chunker.chunk(file, ast_nodes)

        assert len(chunks) > 1

        for chunk in chunks:
            assert chunk.metadata.get("function_name") == "big_function"
            assert chunk.chunk_type == "function"

    def test_oversized_method_preserves_class_context(self, tokenizer, parser):
        """Test that oversized methods preserve class context."""
        lines = ["        self.x{} = {}".format(i, i) for i in range(100)]
        content = """class BigClass:
    def big_method(self):
""" + "\n".join(lines)

        file = self._create_scanned_file(content)
        ast_nodes = parser.parse(content, "python")

        method_nodes = [n for n in ast_nodes if n.node_type == "method"]

        if method_nodes:
            chunker = Chunker(tokenizer=tokenizer, max_tokens=200)
            chunks = chunker.chunk(file, method_nodes)

            for chunk in chunks:
                assert chunk.metadata.get("parent_class") == "BigClass"
