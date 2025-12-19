"""
Fixture-based tests for AST parser docstring extraction.

Loads test cases from JSON fixtures and validates parsing results.
"""

import json
import pytest
from pathlib import Path

from aci.core.ast_parser import TreeSitterParser


# Load fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "docstring_samples"


def load_fixture(filename: str) -> dict:
    """Load a JSON fixture file."""
    filepath = FIXTURES_DIR / filename
    if not filepath.exists():
        pytest.skip(f"Fixture file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_test_cases(fixture_file: str):
    """Get test cases from a fixture file for parametrization."""
    try:
        fixture = load_fixture(fixture_file)
        return [
            (fixture["language"], tc["id"], tc["code"], tc["expected"])
            for tc in fixture["test_cases"]
        ]
    except FileNotFoundError:
        return []


class TestJavaScriptFixtures:
    """Test JavaScript parsing using fixtures."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    @pytest.mark.parametrize(
        "language,test_id,code,expected",
        get_test_cases("javascript_samples.json"),
        ids=lambda x: x if isinstance(x, str) and x.startswith("js_") else None,
    )
    def test_javascript_parsing(self, parser, language, test_id, code, expected):
        """Test JavaScript parsing against fixture expectations."""
        nodes = parser.parse(code, language)
        
        # Verify we got the expected number of nodes
        assert len(nodes) >= len(expected), (
            f"Test {test_id}: Expected at least {len(expected)} nodes, got {len(nodes)}"
        )
        
        # Check each expected node
        for exp in expected:
            matching = [n for n in nodes if n.name == exp["name"]]
            assert len(matching) > 0, f"Test {test_id}: Node '{exp['name']}' not found"
            
            node = matching[0]
            assert node.node_type == exp["node_type"], (
                f"Test {test_id}: Expected type '{exp['node_type']}', got '{node.node_type}'"
            )
            
            if exp["has_docstring"]:
                assert node.docstring is not None, (
                    f"Test {test_id}: Expected docstring for '{exp['name']}'"
                )
                for text in exp.get("docstring_contains", []):
                    assert text in node.docstring, (
                        f"Test {test_id}: Expected '{text}' in docstring"
                    )
            else:
                assert node.docstring is None, (
                    f"Test {test_id}: Expected no docstring for '{exp['name']}'"
                )


class TestPythonFixtures:
    """Test Python parsing using fixtures."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    @pytest.mark.parametrize(
        "language,test_id,code,expected",
        get_test_cases("python_samples.json"),
        ids=lambda x: x if isinstance(x, str) and x.startswith("py_") else None,
    )
    def test_python_parsing(self, parser, language, test_id, code, expected):
        """Test Python parsing against fixture expectations."""
        nodes = parser.parse(code, language)
        
        assert len(nodes) >= len(expected), (
            f"Test {test_id}: Expected at least {len(expected)} nodes, got {len(nodes)}"
        )
        
        for exp in expected:
            matching = [n for n in nodes if n.name == exp["name"]]
            assert len(matching) > 0, f"Test {test_id}: Node '{exp['name']}' not found"
            
            node = matching[0]
            assert node.node_type == exp["node_type"]
            
            if exp["has_docstring"]:
                assert node.docstring is not None
                for text in exp.get("docstring_contains", []):
                    assert text in node.docstring
            else:
                assert node.docstring is None


class TestGoFixtures:
    """Test Go parsing using fixtures."""

    @pytest.fixture
    def parser(self):
        return TreeSitterParser()

    @pytest.mark.parametrize(
        "language,test_id,code,expected",
        get_test_cases("go_samples.json"),
        ids=lambda x: x if isinstance(x, str) and x.startswith("go_") else None,
    )
    def test_go_parsing(self, parser, language, test_id, code, expected):
        """Test Go parsing against fixture expectations."""
        nodes = parser.parse(code, language)
        
        assert len(nodes) >= len(expected), (
            f"Test {test_id}: Expected at least {len(expected)} nodes, got {len(nodes)}"
        )
        
        for exp in expected:
            matching = [n for n in nodes if n.name == exp["name"]]
            assert len(matching) > 0, f"Test {test_id}: Node '{exp['name']}' not found"
            
            node = matching[0]
            assert node.node_type == exp["node_type"]
            
            if exp["has_docstring"]:
                assert node.docstring is not None
                for text in exp.get("docstring_contains", []):
                    assert text in node.docstring
            else:
                assert node.docstring is None
