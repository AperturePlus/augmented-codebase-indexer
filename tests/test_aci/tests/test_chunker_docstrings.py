"""
Tests for Chunker docstring integration.
"""

from pathlib import Path

import pytest

from aci.core.ast_parser import ASTNode
from aci.core.chunker import Chunker
from aci.core.docstring_formatter import DocstringFormatter
from aci.core.file_scanner import ScannedFile
from aci.core.tokenizer import TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    """Lightweight tokenizer for tests (word-based)."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        words = text.split()
        truncated = " ".join(words[:max_tokens])
        return truncated


def _scanned_file(content: str, language: str) -> ScannedFile:
    return ScannedFile(
        path=Path(f"/tmp/test.{language}"),
        content=content,
        language=language,
        size_bytes=len(content),
        modified_time=0.0,
        content_hash="hash",
    )


@pytest.mark.parametrize(
    ("language", "docstring", "code"),
    [
        ("javascript", "/** Adds numbers */", "function add(a,b) { return a+b; }"),
        ("go", "// Add adds numbers\n// more detail", "func Add(a,b int) int { return a+b }"),
        ("java", "/** Greets */", "class G { void hi() {} }"),
        ("cpp", "/// Computes\n/// value", "int compute() { return 1; }"),
    ],
)
def test_chunker_formats_docstring_into_content(language: str, docstring: str, code: str):
    formatter = DocstringFormatter()
    tokenizer = DummyTokenizer()
    chunker = Chunker(tokenizer=tokenizer, max_tokens=200, docstring_formatter=formatter)

    lines = code.split("\n")
    node = ASTNode(
        node_type="function" if language != "java" else "class",
        name="Test",
        start_line=1,
        end_line=len(lines),
        content=code,
        parent_name=None,
        docstring=docstring,
    )
    file = _scanned_file(code, language)

    result = chunker.chunk(file, [node])
    chunks = result.chunks
    assert len(chunks) == 1

    normalized = formatter.normalize(docstring, language)
    assert chunks[0].content.startswith(f"{normalized}{formatter.DELIMITER}")
    assert chunks[0].metadata.get("docstring") == normalized


def test_docstring_only_in_first_split_chunk():
    formatter = DocstringFormatter()
    tokenizer = DummyTokenizer()
    # Force small max_tokens to trigger splitting
    chunker = Chunker(tokenizer=tokenizer, max_tokens=10, docstring_formatter=formatter)

    code_lines = ["line with many words" for _ in range(20)]
    code = "\n".join(code_lines)
    docstring = "/** Summary */"

    node = ASTNode(
        node_type="function",
        name="splitme",
        start_line=1,
        end_line=len(code_lines),
        content=code,
        parent_name=None,
        docstring=docstring,
    )
    file = _scanned_file(code, "javascript")

    result = chunker.chunk(file, [node])
    chunks = result.chunks
    assert len(chunks) > 1

    first, rest = chunks[0], chunks[1:]
    assert formatter.DELIMITER in first.content
    for chunk in rest:
        assert formatter.DELIMITER not in chunk.content
        assert chunk.metadata.get("docstring_included_in_chunk") is False

    assert first.metadata.get("docstring_included_in_chunk") is True
