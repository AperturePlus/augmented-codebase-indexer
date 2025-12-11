"""Property-based tests for MCP path security error message handling."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from hypothesis import given, settings, strategies as st

from tests.mcp_path_security_strategies import run_async, valid_directory_names


class TestErrorMessagePathInclusion:
    """
    **Feature: mcp-path-security, Property 4: Error Message Path Inclusion**
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    """

    @settings(max_examples=100, deadline=None)
    @given(dirname=valid_directory_names())
    def test_nonexistent_path_error_includes_path(self, dirname: str):
        """Non-existent paths should echo the original input path in errors."""
        from aci.mcp.handlers import _handle_index_codebase, _handle_update_index

        nonexistent_path = f"/nonexistent_test_dir_{dirname}/subdir"

        result = run_async(_handle_index_codebase({"path": nonexistent_path}))
        assert len(result) == 1
        error_text = result[0].text
        assert "Error" in error_text and nonexistent_path in error_text and "does not exist" in error_text

        result = run_async(_handle_update_index({"path": nonexistent_path}))
        assert len(result) == 1
        error_text = result[0].text
        assert "Error" in error_text and nonexistent_path in error_text and "does not exist" in error_text

    @settings(max_examples=100, deadline=None)
    @given(
        filename=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=65, max_codepoint=122),
            min_size=1,
            max_size=20,
        )
    )
    def test_file_path_error_includes_path(self, filename: str):
        """File paths (not directories) should be rejected with a helpful message."""
        from aci.mcp.handlers import _handle_index_codebase, _handle_update_index

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / f"file_{filename}.txt"
            file_path.write_text("test content")
            file_path_str = str(file_path)

            result = run_async(_handle_index_codebase({"path": file_path_str}))
            assert len(result) == 1
            error_text = result[0].text
            assert "Error" in error_text and file_path_str in error_text and "not a directory" in error_text

            result = run_async(_handle_update_index({"path": file_path_str}))
            assert len(result) == 1
            error_text = result[0].text
            assert "Error" in error_text and file_path_str in error_text and "not a directory" in error_text

    @settings(max_examples=100, deadline=None)
    @given(dirname=valid_directory_names())
    def test_system_dir_error_includes_path(self, dirname: str):
        """System directories should be rejected and include path + forbidden text."""
        from aci.mcp.handlers import _handle_index_codebase, _handle_update_index

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)
            test_path_str = str(test_path)

            with patch("aci.core.path_utils.is_system_directory", return_value=True):
                result = run_async(_handle_index_codebase({"path": test_path_str}))
                assert len(result) == 1
                error_text = result[0].text
                assert "Error" in error_text and test_path_str in error_text
                assert "forbidden" in error_text.lower()

                result = run_async(_handle_update_index({"path": test_path_str}))
                assert len(result) == 1
                error_text = result[0].text
                assert "Error" in error_text and test_path_str in error_text
                assert "forbidden" in error_text.lower()

    def test_error_format_consistency(self):
        """Error messages should follow the expected format with path included."""
        from aci.mcp.handlers import _handle_index_codebase

        test_cases = ["/nonexistent/path/xyz"]

        for path in test_cases:
            result = run_async(_handle_index_codebase({"path": path}))
            assert len(result) == 1
            error_text = result[0].text
            assert error_text.startswith("Error:")
            assert f"(path: {path})" in error_text

