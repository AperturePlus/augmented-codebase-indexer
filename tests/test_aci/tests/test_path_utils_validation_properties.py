"""Property-based tests for path validation utilities."""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aci.core.path_utils import (
    WINDOWS_SYSTEM_DIRS,
    ensure_directory_exists,
    is_system_directory,
    validate_indexable_path,
)
from tests.path_utils_strategies import (
    WINDOWS_RESERVED,
    non_existent_paths,
    posix_system_paths,
    windows_system_paths,
)


class TestPathExistenceValidation:
    """
    **Feature: path-validation-fixes, Property 1: Non-existent paths are rejected**
    **Validates: Requirements 2.1, 2.3**
    """

    @settings(max_examples=100)
    @given(path=non_existent_paths())
    def test_non_existent_paths_rejected(self, path: str):
        """validate_indexable_path should reject paths that do not exist."""
        if Path(path).exists():
            pytest.skip("Generated path unexpectedly exists")

        result = validate_indexable_path(path)

        assert result.valid is False
        assert result.error_message is not None
        assert "does not exist" in result.error_message


class TestDirectoryTypeValidation:
    """
    **Feature: path-validation-fixes, Property 2: Non-directory paths are rejected**
    **Validates: Requirements 2.2, 2.4**
    """

    @settings(max_examples=100)
    @given(
        filename=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=65, max_codepoint=122),
            min_size=1,
            max_size=20,
        )
    )
    def test_file_paths_rejected(self, filename: str):
        """validate_indexable_path should reject paths that are files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / f"file_{filename}.txt"
            file_path.write_text("test content")

            result = validate_indexable_path(file_path)

            assert result.valid is False
            assert result.error_message is not None
            assert "not a directory" in result.error_message


class TestSystemDirectoryDetection:
    """
    **Feature: path-validation-fixes, Property 3: System directories are detected across platforms**
    **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
    """

    @settings(max_examples=100)
    @given(path=posix_system_paths())
    def test_posix_system_directories_detected(self, path: str):
        """POSIX system directories should be detected."""
        from aci.core.path_utils import _is_posix_system_directory

        assert _is_posix_system_directory(path) is True

    @settings(max_examples=100)
    @given(path=windows_system_paths())
    def test_windows_system_directories_detected(self, path: str):
        """Windows system directories should be detected."""
        from aci.core.path_utils import _is_windows_system_directory

        mock_path = Path(path)
        assert _is_windows_system_directory(mock_path, path) is True

    @settings(max_examples=100)
    @given(
        dirname=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=65, max_codepoint=122),
            min_size=5,
            max_size=20,
        ).filter(lambda x: x.lower() not in WINDOWS_SYSTEM_DIRS)
    )
    def test_non_system_directories_not_flagged(self, dirname: str):
        """Non-system directories should not be flagged as system directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)

            assert is_system_directory(test_path) is False


class TestValidDirectoryAccepted:
    """Valid directories should pass validation."""

    @settings(max_examples=100)
    @given(
        dirname=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=65, max_codepoint=122),
            min_size=1,
            max_size=20,
        ).filter(lambda x: x.upper() not in WINDOWS_RESERVED)
    )
    def test_valid_directories_accepted(self, dirname: str):
        """validate_indexable_path should accept existing, non-system directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)

            result = validate_indexable_path(test_path)

            assert result.valid is True
            assert result.error_message is None


class TestEnsureDirectoryExists:
    """ensure_directory_exists should create nested directories."""

    @settings(max_examples=50)
    @given(
        parts=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N"), min_codepoint=65, max_codepoint=122),
                min_size=1,
                max_size=10,
            ).filter(lambda x: x.upper() not in WINDOWS_RESERVED),
            min_size=1,
            max_size=3,
        )
    )
    def test_creates_nested_directories(self, parts: list):
        """ensure_directory_exists should create requested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir).joinpath(*parts)

            result = ensure_directory_exists(nested_path)

            assert result is True
            assert nested_path.exists()
            assert nested_path.is_dir()
