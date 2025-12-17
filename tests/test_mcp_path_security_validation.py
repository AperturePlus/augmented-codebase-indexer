"""Property-based tests for MCP path validation and system directory rejection."""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from hypothesis import given, settings

from aci.core.path_utils import POSIX_SYSTEM_DIRS, WINDOWS_SYSTEM_DIRS, validate_indexable_path
from tests.mcp_path_security_strategies import (
    posix_system_paths,
    run_async,
    valid_directory_names,
    windows_system_paths,
)


def _create_mock_context(
    indexing_service=None,
    metadata_store=None,
):
    """Create a mock MCPContext for testing."""
    from aci.core.config import ACIConfig
    from aci.mcp.context import MCPContext

    # Create a real config for proper nested attribute access
    mock_cfg = ACIConfig()

    if indexing_service is None:
        indexing_service = MagicMock()

    if metadata_store is None:
        metadata_store = MagicMock()

    return MCPContext(
        config=mock_cfg,
        search_service=MagicMock(),
        indexing_service=indexing_service,
        metadata_store=metadata_store,
        vector_store=MagicMock(),
        indexing_lock=asyncio.Lock(),
    )


class TestMCPPathValidationConsistency:
    """
    **Feature: mcp-path-security, Property 1: MCP Path Validation Consistency**
    **Validates: Requirements 1.1, 1.2, 1.4**
    """

    @settings(max_examples=100, deadline=None)
    @given(dirname=valid_directory_names())
    def test_index_codebase_validates_before_indexing(self, dirname: str):
        """For valid paths, index_codebase should validate and then attempt indexing."""
        from aci.mcp.handlers import _handle_index_codebase

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)

            mock_indexing_service = MagicMock()
            mock_result = MagicMock(
                total_files=0, total_chunks=0, duration_seconds=0.1, failed_files=[]
            )
            mock_indexing_service.index_directory = AsyncMock(return_value=mock_result)

            ctx = _create_mock_context(indexing_service=mock_indexing_service)

            result = run_async(_handle_index_codebase({"path": str(test_path)}, ctx))
            assert mock_indexing_service.index_directory.called

    @settings(max_examples=100, deadline=None)
    @given(dirname=valid_directory_names())
    def test_update_index_validates_before_updating(self, dirname: str):
        """For valid paths, update_index should validate and then attempt update."""
        from aci.mcp.handlers import _handle_update_index

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)

            mock_indexing_service = MagicMock()
            mock_result = MagicMock(
                new_files=0, modified_files=0, deleted_files=0, duration_seconds=0.1
            )
            mock_indexing_service.update_incremental = AsyncMock(return_value=mock_result)

            mock_metadata_store = MagicMock()
            mock_metadata_store.get_index_info.return_value = {"root_path": str(test_path)}
            mock_metadata_store.get_all_file_hashes.return_value = {"dummy": "hash"}

            ctx = _create_mock_context(
                indexing_service=mock_indexing_service,
                metadata_store=mock_metadata_store,
            )

            result = run_async(_handle_update_index({"path": str(test_path)}, ctx))
            assert mock_indexing_service.update_incremental.called

    @settings(max_examples=100, deadline=None)
    @given(dirname=valid_directory_names())
    def test_invalid_path_prevents_indexing(self, dirname: str):
        """Invalid paths should short-circuit before indexing is attempted."""
        from aci.mcp.handlers import _handle_index_codebase

        nonexistent_path = f"/nonexistent/{dirname}/path"
        mock_indexing_service = MagicMock()

        ctx = _create_mock_context(indexing_service=mock_indexing_service)

        result = run_async(_handle_index_codebase({"path": nonexistent_path}, ctx))

        assert not mock_indexing_service.index_directory.called
        assert len(result) == 1
        assert "Error" in result[0].text

    @settings(max_examples=100, deadline=None)
    @given(dirname=valid_directory_names())
    def test_invalid_path_prevents_update(self, dirname: str):
        """Invalid paths should short-circuit before update is attempted."""
        from aci.mcp.handlers import _handle_update_index

        nonexistent_path = f"/nonexistent/{dirname}/path"
        mock_indexing_service = MagicMock()

        ctx = _create_mock_context(indexing_service=mock_indexing_service)

        result = run_async(_handle_update_index({"path": nonexistent_path}, ctx))

        assert not mock_indexing_service.update_incremental.called
        assert len(result) == 1
        assert "Error" in result[0].text


class TestSystemDirectoryRejection:
    """
    **Feature: mcp-path-security, Property 2: System Directory Rejection**
    **Validates: Requirements 1.3, 3.1, 4.3, 4.4**
    """

    @settings(max_examples=100, deadline=None)
    @given(path=posix_system_paths())
    def test_posix_system_paths_detected(self, path: str):
        """POSIX system directory detection should return True."""
        from aci.core.path_utils import _is_posix_system_directory

        assert _is_posix_system_directory(path) is True

    @settings(max_examples=100, deadline=None)
    @given(path=windows_system_paths())
    def test_windows_system_paths_detected(self, path: str):
        """Windows system directory detection should return True."""
        from aci.core.path_utils import _is_windows_system_directory

        mock_path = Path(path)
        assert _is_windows_system_directory(mock_path, path) is True

    @settings(max_examples=100, deadline=None)
    @given(dirname=valid_directory_names())
    def test_validate_indexable_path_rejects_system_dirs(self, dirname: str):
        """validate_indexable_path should reject directories inside system paths."""
        from aci.core.path_utils import _is_posix_system_directory

        system_path = f"/etc/{dirname}"
        assert _is_posix_system_directory(system_path) is True

        result = validate_indexable_path(system_path)
        assert result.valid is False
        assert result.error_message is not None

    @settings(max_examples=100, deadline=None)
    @given(dirname=valid_directory_names())
    def test_mcp_error_includes_path_for_system_dirs(self, dirname: str):
        """MCP handler should include original path when rejecting system dirs."""
        from aci.mcp.handlers import _handle_index_codebase

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)

            ctx = _create_mock_context()

            with patch("aci.core.path_utils.is_system_directory", return_value=True):
                result = run_async(_handle_index_codebase({"path": str(test_path)}, ctx))

                assert len(result) == 1
                error_text = result[0].text
                assert str(test_path) in error_text
                assert "forbidden" in error_text.lower()

    def test_all_posix_system_dirs_in_constant(self):
        """Ensure expected POSIX system directories are present."""
        expected_dirs = {"/etc", "/var", "/usr", "/bin", "/sbin", "/proc", "/sys", "/dev"}
        assert expected_dirs.issubset(POSIX_SYSTEM_DIRS)

    def test_all_windows_system_dirs_in_constant(self):
        """Ensure expected Windows system directories are present."""
        expected_dirs = {"windows", "program files", "system32"}
        assert expected_dirs.issubset(WINDOWS_SYSTEM_DIRS)
