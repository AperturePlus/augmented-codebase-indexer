"""
Property-based tests for MCP path security.

**Feature: mcp-path-security**

Uses Hypothesis for property-based testing with minimum 100 iterations.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, strategies as st

from aci.core.path_utils import (
    POSIX_SYSTEM_DIRS,
    WINDOWS_SYSTEM_DIRS,
    validate_indexable_path,
)


def _run_async(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# Windows reserved device names that can't be used as directory names
WINDOWS_RESERVED = frozenset([
    'CON', 'PRN', 'AUX', 'NUL',
    'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
    'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9',
])


# Strategies for generating test data
@st.composite
def valid_directory_names(draw):
    """Generate valid directory names that aren't Windows reserved or system dirs."""
    name = draw(st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
        min_size=1,
        max_size=20
    ))
    # Filter out Windows reserved names
    if name.upper() in WINDOWS_RESERVED:
        name = f"dir_{name}"
    # Filter out Windows system directory names (case-insensitive)
    if name.lower() in WINDOWS_SYSTEM_DIRS:
        name = f"test_{name}"
    return name


@st.composite
def posix_system_paths(draw):
    """Generate paths under POSIX system directories."""
    base = draw(st.sampled_from(list(POSIX_SYSTEM_DIRS)))
    subpath = draw(st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
        min_size=0,
        max_size=20
    ))
    if subpath:
        return f"{base}/{subpath}"
    return base


@st.composite
def windows_system_paths(draw):
    """Generate paths under Windows system directories."""
    drive = draw(st.sampled_from(['C:', 'D:']))
    sys_dir = draw(st.sampled_from(list(WINDOWS_SYSTEM_DIRS)))
    sys_dir_proper = sys_dir.title() if sys_dir != "program files (x86)" else "Program Files (x86)"
    subpath = draw(st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
        min_size=0,
        max_size=20
    ))
    if subpath:
        return f"{drive}\\{sys_dir_proper}\\{subpath}"
    return f"{drive}\\{sys_dir_proper}"


class TestMCPPathValidationConsistency:
    """
    **Feature: mcp-path-security, Property 1: MCP Path Validation Consistency**
    **Validates: Requirements 1.1, 1.2, 1.4**
    
    For any path passed to index_codebase or update_index MCP tools,
    the handler SHALL call validate_indexable_path before attempting
    any indexing operation, and if validation fails, no indexing
    operation SHALL occur.
    """

    @settings(max_examples=100, deadline=None)
    @given(dirname=valid_directory_names())
    def test_index_codebase_validates_before_indexing(self, dirname: str):
        """
        Property 1a: For any valid directory, index_codebase SHALL call
        validate_indexable_path and proceed with indexing only if valid.
        """
        from unittest.mock import AsyncMock
        from aci.mcp.handlers import _handle_index_codebase

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)

            # Mock the services to track if indexing was attempted
            mock_indexing_service = MagicMock()
            mock_result = MagicMock(
                total_files=0, total_chunks=0, duration_seconds=0.1, failed_files=[]
            )
            mock_indexing_service.index_directory = AsyncMock(return_value=mock_result)

            with patch('aci.mcp.handlers.get_initialized_services') as mock_services:
                mock_cfg = MagicMock()
                mock_cfg.indexing.max_workers = 4
                mock_services.return_value = (mock_cfg, None, mock_indexing_service, None, None)

                result = _run_async(_handle_index_codebase({"path": str(test_path)}))

                # For valid paths, indexing should be attempted
                assert mock_indexing_service.index_directory.called

    @settings(max_examples=100, deadline=None)
    @given(dirname=valid_directory_names())
    def test_update_index_validates_before_updating(self, dirname: str):
        """
        Property 1b: For any valid directory, update_index SHALL call
        validate_indexable_path and proceed with updating only if valid.
        """
        from unittest.mock import AsyncMock
        from aci.mcp.handlers import _handle_update_index

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)

            mock_indexing_service = MagicMock()
            mock_result = MagicMock(
                new_files=0, modified_files=0, deleted_files=0, duration_seconds=0.1
            )
            mock_indexing_service.update_incremental = AsyncMock(return_value=mock_result)

            with patch('aci.mcp.handlers.get_initialized_services') as mock_services:
                mock_cfg = MagicMock()
                mock_services.return_value = (mock_cfg, None, mock_indexing_service, None, None)

                result = _run_async(_handle_update_index({"path": str(test_path)}))

                # For valid paths, update should be attempted
                assert mock_indexing_service.update_incremental.called

    @settings(max_examples=100, deadline=None)
    @given(dirname=valid_directory_names())
    def test_invalid_path_prevents_indexing(self, dirname: str):
        """
        Property 1c: For any non-existent path, index_codebase SHALL NOT
        attempt any indexing operation.
        """
        from aci.mcp.handlers import _handle_index_codebase

        # Use a path that doesn't exist
        nonexistent_path = f"/nonexistent/{dirname}/path"

        mock_indexing_service = MagicMock()

        with patch('aci.mcp.handlers.get_initialized_services') as mock_services:
            mock_cfg = MagicMock()
            mock_services.return_value = (mock_cfg, None, mock_indexing_service, None, None)

            result = _run_async(_handle_index_codebase({"path": nonexistent_path}))

            # Indexing should NOT be attempted for invalid paths
            assert not mock_indexing_service.index_directory.called
            # Result should contain error
            assert len(result) == 1
            assert "Error" in result[0].text

    @settings(max_examples=100, deadline=None)
    @given(dirname=valid_directory_names())
    def test_invalid_path_prevents_update(self, dirname: str):
        """
        Property 1d: For any non-existent path, update_index SHALL NOT
        attempt any update operation.
        """
        from aci.mcp.handlers import _handle_update_index

        nonexistent_path = f"/nonexistent/{dirname}/path"

        mock_indexing_service = MagicMock()

        with patch('aci.mcp.handlers.get_initialized_services') as mock_services:
            mock_cfg = MagicMock()
            mock_services.return_value = (mock_cfg, None, mock_indexing_service, None, None)

            result = _run_async(_handle_update_index({"path": nonexistent_path}))

            # Update should NOT be attempted for invalid paths
            assert not mock_indexing_service.update_incremental.called
            # Result should contain error
            assert len(result) == 1
            assert "Error" in result[0].text



class TestSystemDirectoryRejection:
    """
    **Feature: mcp-path-security, Property 2: System Directory Rejection**
    **Validates: Requirements 1.3, 3.1, 4.3, 4.4**
    
    For any path that resolves to a system directory (as defined by
    is_system_directory), all interfaces (CLI, HTTP, MCP) SHALL reject
    the request with an error message containing "forbidden" and the
    original path.
    """

    @settings(max_examples=100, deadline=None)
    @given(path=posix_system_paths())
    def test_posix_system_paths_detected(self, path: str):
        """
        Property 2a: For any POSIX system directory path, the internal
        _is_posix_system_directory function SHALL return True.
        """
        from aci.core.path_utils import _is_posix_system_directory

        # Verify the path is detected as a system directory
        assert _is_posix_system_directory(path) is True

    @settings(max_examples=100, deadline=None)
    @given(path=windows_system_paths())
    def test_windows_system_paths_detected(self, path: str):
        """
        Property 2b: For any Windows system directory path, the internal
        _is_windows_system_directory function SHALL return True.
        """
        from aci.core.path_utils import _is_windows_system_directory

        mock_path = Path(path)
        result = _is_windows_system_directory(mock_path, path)
        assert result is True

    @settings(max_examples=100, deadline=None)
    @given(dirname=valid_directory_names())
    def test_validate_indexable_path_rejects_system_dirs(self, dirname: str):
        """
        Property 2c: For any path under a system directory, validate_indexable_path
        SHALL return valid=False with an appropriate error message.
        """
        from aci.core.path_utils import _is_posix_system_directory

        # Create a path under /etc (POSIX system dir)
        system_path = f"/etc/{dirname}"

        # Verify it's detected as system directory
        assert _is_posix_system_directory(system_path) is True

        # validate_indexable_path should reject it
        result = validate_indexable_path(system_path)
        assert result.valid is False
        assert result.error_message is not None

    @settings(max_examples=100, deadline=None)
    @given(dirname=valid_directory_names())
    def test_mcp_error_includes_path_for_system_dirs(self, dirname: str):
        """
        Property 2d: When MCP handlers reject a system directory, the error
        message SHALL include the original path.
        """
        import sys
        import tempfile
        from unittest.mock import patch
        from aci.mcp.handlers import _handle_index_codebase

        # We need to test with a path that:
        # 1. Exists (so we get past the "does not exist" check)
        # 2. Is a directory (so we get past the "not a directory" check)
        # 3. Is detected as a system directory
        
        # Mock is_system_directory to return True for our test path
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)
            
            with patch('aci.core.path_utils.is_system_directory', return_value=True):
                result = _run_async(_handle_index_codebase({"path": str(test_path)}))
                
                assert len(result) == 1
                error_text = result[0].text
                # Error should contain the path
                assert str(test_path) in error_text
                # Error should indicate forbidden
                assert "forbidden" in error_text.lower()

    def test_all_posix_system_dirs_in_constant(self):
        """
        Property 2e: All expected POSIX system directories should be in
        the POSIX_SYSTEM_DIRS constant.
        """
        expected_dirs = {"/etc", "/var", "/usr", "/bin", "/sbin", "/proc", "/sys", "/dev"}
        assert expected_dirs.issubset(POSIX_SYSTEM_DIRS)

    def test_all_windows_system_dirs_in_constant(self):
        """
        Property 2f: All expected Windows system directories should be in
        the WINDOWS_SYSTEM_DIRS constant.
        """
        expected_dirs = {"windows", "program files", "system32"}
        assert expected_dirs.issubset(WINDOWS_SYSTEM_DIRS)



class TestSensitiveDenylistEnforcement:
    """
    **Feature: mcp-path-security, Property 3: Sensitive Denylist Enforcement**
    **Validates: Requirements 2.1, 2.2, 2.3**
    
    For any FileScanner scan operation and for any user-provided ignore_patterns
    configuration, files matching the sensitive denylist SHALL never be yielded,
    regardless of whether user patterns would otherwise include them.
    """

    @settings(max_examples=100, deadline=None)
    @given(user_patterns=st.lists(st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
        min_size=1,
        max_size=10
    ), min_size=0, max_size=5))
    def test_sensitive_files_never_yielded(self, user_patterns: list[str]):
        """
        Property 3a: For any user-provided ignore patterns, files matching
        the sensitive denylist SHALL never be yielded by FileScanner.
        """
        from aci.core.file_scanner import FileScanner, SENSITIVE_DENYLIST

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create sensitive files that should always be excluded
            sensitive_files = [
                ".ssh/id_rsa",
                ".ssh/id_ed25519",
                ".gnupg/private-keys-v1.d/key.key",
                "config/.env",
                "config/.env.local",
                "certs/server.pem",
                "certs/server.key",
                "certs/ca.crt",
                "secrets/keystore.p12",
                ".netrc",
                ".npmrc",
                ".pypirc",
            ]

            # Create normal files that should be scanned
            normal_files = [
                "src/main.py",
                "src/utils.py",
                "tests/test_main.py",
            ]

            # Create all files
            for rel_path in sensitive_files + normal_files:
                file_path = tmpdir_path / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(f"# Content of {rel_path}\n", encoding="utf-8")

            # Create scanner with user-provided patterns (which should NOT override denylist)
            scanner = FileScanner(
                extensions={".py", ".pem", ".key", ".crt", ".p12"},
                ignore_patterns=user_patterns,  # User patterns should not affect denylist
            )

            # Scan the directory
            scanned_files = list(scanner.scan(tmpdir_path))
            scanned_names = {sf.path.name for sf in scanned_files}
            scanned_paths_str = {str(sf.path) for sf in scanned_files}

            # Property: No sensitive file should appear in results
            for rel_path in sensitive_files:
                file_path = (tmpdir_path / rel_path).resolve()
                filename = Path(rel_path).name

                # Check that the sensitive file is not in results
                assert file_path not in {sf.path for sf in scanned_files}, (
                    f"Sensitive file {rel_path} should be excluded by denylist "
                    f"but was found in scan results"
                )

    @settings(max_examples=100, deadline=None)
    @given(sensitive_pattern=st.sampled_from([
        ".ssh", ".gnupg", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
        ".env", ".netrc", ".npmrc", ".pypirc"
    ]))
    def test_exact_match_patterns_excluded(self, sensitive_pattern: str):
        """
        Property 3b: For any exact-match sensitive pattern, files with that
        exact name SHALL be excluded regardless of location in the directory tree.
        """
        from aci.core.file_scanner import FileScanner

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create the sensitive file/directory at various depths
            locations = [
                sensitive_pattern,  # Root level
                f"subdir/{sensitive_pattern}",  # One level deep
                f"deep/nested/path/{sensitive_pattern}",  # Multiple levels deep
            ]

            for loc in locations:
                file_path = tmpdir_path / loc
                file_path.parent.mkdir(parents=True, exist_ok=True)
                # Create as file (for patterns like .env, id_rsa)
                if not sensitive_pattern.startswith("."):
                    file_path.write_text("sensitive content", encoding="utf-8")
                else:
                    # For directory patterns like .ssh, .gnupg, create a file inside
                    file_path.mkdir(exist_ok=True)
                    (file_path / "test.py").write_text("# test", encoding="utf-8")

            # Also create a normal file to ensure scanner works
            normal_file = tmpdir_path / "normal.py"
            normal_file.write_text("# normal file", encoding="utf-8")

            # Create scanner with empty user patterns
            scanner = FileScanner(
                extensions={".py", ""},  # Include extensionless files
                ignore_patterns=[],
            )

            scanned_files = list(scanner.scan(tmpdir_path))
            scanned_names = {sf.path.name for sf in scanned_files}

            # Property: The sensitive pattern should not appear in any scanned file names
            # (for directory patterns, files inside should not be scanned)
            for sf in scanned_files:
                # Check that no part of the path contains the sensitive pattern
                path_parts = sf.path.parts
                assert sensitive_pattern not in path_parts, (
                    f"File {sf.path} contains sensitive pattern '{sensitive_pattern}' "
                    f"in its path but should have been excluded"
                )

    @settings(max_examples=100, deadline=None)
    @given(extension=st.sampled_from([".pem", ".key", ".p12", ".pfx", ".crt", ".keystore"]))
    def test_glob_pattern_extensions_excluded(self, extension: str):
        """
        Property 3c: For any glob-pattern sensitive extension (*.pem, *.key, etc.),
        files with that extension SHALL be excluded regardless of filename.
        """
        from aci.core.file_scanner import FileScanner

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create files with the sensitive extension and various names
            sensitive_files = [
                f"server{extension}",
                f"client{extension}",
                f"ca{extension}",
                f"subdir/nested{extension}",
            ]

            for rel_path in sensitive_files:
                file_path = tmpdir_path / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("sensitive content", encoding="utf-8")

            # Create a normal file
            normal_file = tmpdir_path / "normal.py"
            normal_file.write_text("# normal", encoding="utf-8")

            # Create scanner that would normally include these extensions
            scanner = FileScanner(
                extensions={".py", extension},  # Include the sensitive extension
                ignore_patterns=[],
            )

            scanned_files = list(scanner.scan(tmpdir_path))

            # Property: No file with the sensitive extension should be scanned
            for sf in scanned_files:
                assert sf.path.suffix != extension, (
                    f"File {sf.path} has sensitive extension '{extension}' "
                    f"but should have been excluded by denylist"
                )

    @settings(max_examples=100, deadline=None)
    @given(env_variant=st.sampled_from([
        ".env", ".env.local", ".env.production", ".env.development",
        ".env.test", ".env.staging"
    ]))
    def test_env_file_variants_excluded(self, env_variant: str):
        """
        Property 3d: For any .env file variant (.env, .env.local, .env.production, etc.),
        the file SHALL be excluded from scanning.
        """
        from aci.core.file_scanner import FileScanner

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create the env file at root and in subdirectories
            locations = [
                env_variant,
                f"config/{env_variant}",
                f"app/settings/{env_variant}",
            ]

            for loc in locations:
                file_path = tmpdir_path / loc
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text("SECRET_KEY=xxx", encoding="utf-8")

            # Create a normal file
            normal_file = tmpdir_path / "app.py"
            normal_file.write_text("# app", encoding="utf-8")

            # Create scanner with empty ignore patterns
            scanner = FileScanner(
                extensions={".py", ""},  # Include extensionless files
                ignore_patterns=[],
            )

            scanned_files = list(scanner.scan(tmpdir_path))

            # Property: No .env variant should be in results
            for sf in scanned_files:
                assert not sf.path.name.startswith(".env"), (
                    f"File {sf.path} is an env file variant but should have been excluded"
                )

    def test_denylist_cannot_be_overridden_by_empty_patterns(self):
        """
        Property 3e: Even with completely empty ignore patterns, the sensitive
        denylist SHALL still be enforced.
        """
        from aci.core.file_scanner import FileScanner

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create sensitive files
            sensitive_file = tmpdir_path / "id_rsa"
            sensitive_file.write_text("private key content", encoding="utf-8")

            env_file = tmpdir_path / ".env"
            env_file.write_text("SECRET=xxx", encoding="utf-8")

            # Create normal file
            normal_file = tmpdir_path / "main.py"
            normal_file.write_text("# main", encoding="utf-8")

            # Create scanner with explicitly empty patterns
            scanner = FileScanner(
                extensions={".py", ""},
                ignore_patterns=[],  # Empty - should not disable denylist
            )

            scanned_files = list(scanner.scan(tmpdir_path))
            scanned_names = {sf.path.name for sf in scanned_files}

            # Sensitive files should still be excluded
            assert "id_rsa" not in scanned_names
            assert ".env" not in scanned_names
            # Normal file should be included
            assert "main.py" in scanned_names


class TestErrorMessagePathInclusion:
    """
    **Feature: mcp-path-security, Property 4: Error Message Path Inclusion**
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    
    For any path validation error in MCP handlers, the error message
    SHALL include the original path string provided by the caller.
    """

    @settings(max_examples=100, deadline=None)
    @given(dirname=valid_directory_names())
    def test_nonexistent_path_error_includes_path(self, dirname: str):
        """
        Property 4a: When a path does not exist, the error message SHALL
        include the original path for debugging purposes.
        """
        from aci.mcp.handlers import _handle_index_codebase, _handle_update_index

        # Use a path that definitely doesn't exist
        nonexistent_path = f"/nonexistent_test_dir_{dirname}/subdir"

        # Test index_codebase
        result = _run_async(_handle_index_codebase({"path": nonexistent_path}))
        assert len(result) == 1
        error_text = result[0].text
        assert "Error" in error_text
        assert nonexistent_path in error_text
        assert "does not exist" in error_text

        # Test update_index
        result = _run_async(_handle_update_index({"path": nonexistent_path}))
        assert len(result) == 1
        error_text = result[0].text
        assert "Error" in error_text
        assert nonexistent_path in error_text
        assert "does not exist" in error_text

    @settings(max_examples=100, deadline=None)
    @given(filename=st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N'), min_codepoint=65, max_codepoint=122),
        min_size=1,
        max_size=20
    ))
    def test_file_path_error_includes_path(self, filename: str):
        """
        Property 4b: When a path is not a directory (is a file), the error
        message SHALL include the original path.
        """
        from aci.mcp.handlers import _handle_index_codebase, _handle_update_index

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file (not a directory)
            file_path = Path(tmpdir) / f"file_{filename}.txt"
            file_path.write_text("test content")
            file_path_str = str(file_path)

            # Test index_codebase
            result = _run_async(_handle_index_codebase({"path": file_path_str}))
            assert len(result) == 1
            error_text = result[0].text
            assert "Error" in error_text
            assert file_path_str in error_text
            assert "not a directory" in error_text

            # Test update_index
            result = _run_async(_handle_update_index({"path": file_path_str}))
            assert len(result) == 1
            error_text = result[0].text
            assert "Error" in error_text
            assert file_path_str in error_text
            assert "not a directory" in error_text

    @settings(max_examples=100, deadline=None)
    @given(dirname=valid_directory_names())
    def test_system_dir_error_includes_path(self, dirname: str):
        """
        Property 4c: When a path is a system directory, the error message
        SHALL include the original path and indicate forbidden.
        """
        from unittest.mock import patch
        from aci.mcp.handlers import _handle_index_codebase, _handle_update_index

        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / dirname
            test_path.mkdir(exist_ok=True)
            test_path_str = str(test_path)

            # Mock is_system_directory to return True
            with patch('aci.core.path_utils.is_system_directory', return_value=True):
                # Test index_codebase
                result = _run_async(_handle_index_codebase({"path": test_path_str}))
                assert len(result) == 1
                error_text = result[0].text
                assert "Error" in error_text
                assert test_path_str in error_text
                assert "forbidden" in error_text.lower()

                # Test update_index
                result = _run_async(_handle_update_index({"path": test_path_str}))
                assert len(result) == 1
                error_text = result[0].text
                assert "Error" in error_text
                assert test_path_str in error_text
                assert "forbidden" in error_text.lower()

    def test_error_format_consistency(self):
        """
        Property 4d: All error messages should follow the format:
        "Error: <message> (path: <original_path>)"
        """
        from aci.mcp.handlers import _handle_index_codebase

        test_cases = [
            "/nonexistent/path/xyz",  # Non-existent
        ]

        for path in test_cases:
            result = _run_async(_handle_index_codebase({"path": path}))
            assert len(result) == 1
            error_text = result[0].text
            # Should start with "Error:"
            assert error_text.startswith("Error:")
            # Should contain "(path: <path>)" at the end
            assert f"(path: {path})" in error_text


class TestSensitiveDenylistPatterns:
    """
    Unit tests for sensitive denylist patterns.
    
    **Validates: Requirements 2.4**
    
    Verifies that all required patterns are present in the SENSITIVE_DENYLIST
    constant and that pattern matching works correctly for each sensitive file type.
    """

    def test_required_ssh_patterns_in_denylist(self):
        """Verify SSH-related patterns are in the denylist."""
        from aci.core.file_scanner import SENSITIVE_DENYLIST

        ssh_patterns = {".ssh", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"}
        for pattern in ssh_patterns:
            assert pattern in SENSITIVE_DENYLIST, (
                f"SSH pattern '{pattern}' should be in SENSITIVE_DENYLIST"
            )

    def test_required_ssh_pub_patterns_in_denylist(self):
        """Verify SSH public key patterns are in the denylist."""
        from aci.core.file_scanner import SENSITIVE_DENYLIST

        ssh_pub_patterns = {"id_rsa.pub", "id_ed25519.pub", "id_ecdsa.pub", "id_dsa.pub"}
        for pattern in ssh_pub_patterns:
            assert pattern in SENSITIVE_DENYLIST, (
                f"SSH public key pattern '{pattern}' should be in SENSITIVE_DENYLIST"
            )

    def test_required_gnupg_pattern_in_denylist(self):
        """Verify GPG-related patterns are in the denylist."""
        from aci.core.file_scanner import SENSITIVE_DENYLIST

        assert ".gnupg" in SENSITIVE_DENYLIST, ".gnupg should be in SENSITIVE_DENYLIST"

    def test_required_certificate_patterns_in_denylist(self):
        """Verify certificate and key patterns are in the denylist."""
        from aci.core.file_scanner import SENSITIVE_DENYLIST

        cert_patterns = {"*.pem", "*.key", "*.p12", "*.pfx", "*.crt", "*.keystore"}
        for pattern in cert_patterns:
            assert pattern in SENSITIVE_DENYLIST, (
                f"Certificate pattern '{pattern}' should be in SENSITIVE_DENYLIST"
            )

    def test_required_env_patterns_in_denylist(self):
        """Verify environment file patterns are in the denylist."""
        from aci.core.file_scanner import SENSITIVE_DENYLIST

        env_patterns = {".env", ".env.*"}
        for pattern in env_patterns:
            assert pattern in SENSITIVE_DENYLIST, (
                f"Environment pattern '{pattern}' should be in SENSITIVE_DENYLIST"
            )

    def test_required_auth_config_patterns_in_denylist(self):
        """Verify authentication config patterns are in the denylist."""
        from aci.core.file_scanner import SENSITIVE_DENYLIST

        auth_patterns = {".netrc", ".npmrc", ".pypirc"}
        for pattern in auth_patterns:
            assert pattern in SENSITIVE_DENYLIST, (
                f"Auth config pattern '{pattern}' should be in SENSITIVE_DENYLIST"
            )

    def test_denylist_is_frozenset(self):
        """Verify SENSITIVE_DENYLIST is immutable (frozenset)."""
        from aci.core.file_scanner import SENSITIVE_DENYLIST

        assert isinstance(SENSITIVE_DENYLIST, frozenset), (
            "SENSITIVE_DENYLIST should be a frozenset to prevent modification"
        )

    def test_matches_sensitive_denylist_exact_match(self):
        """Test _matches_sensitive_denylist for exact filename matches."""
        from aci.core.file_scanner import FileScanner

        scanner = FileScanner()

        # Test exact matches
        exact_matches = [
            ".ssh", ".gnupg", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
            "id_rsa.pub", "id_ed25519.pub", ".env", ".netrc", ".npmrc", ".pypirc"
        ]

        for filename in exact_matches:
            path = Path(f"/some/path/{filename}")
            assert scanner._matches_sensitive_denylist(path), (
                f"File '{filename}' should match sensitive denylist"
            )

    def test_matches_sensitive_denylist_glob_patterns(self):
        """Test _matches_sensitive_denylist for glob pattern matches."""
        from aci.core.file_scanner import FileScanner

        scanner = FileScanner()

        # Test glob pattern matches
        glob_matches = [
            ("server.pem", "*.pem"),
            ("private.key", "*.key"),
            ("cert.p12", "*.p12"),
            ("keystore.pfx", "*.pfx"),
            ("ca.crt", "*.crt"),
            ("app.keystore", "*.keystore"),
            (".env.local", ".env.*"),
            (".env.production", ".env.*"),
            (".env.development", ".env.*"),
        ]

        for filename, pattern in glob_matches:
            path = Path(f"/some/path/{filename}")
            assert scanner._matches_sensitive_denylist(path), (
                f"File '{filename}' should match pattern '{pattern}' in sensitive denylist"
            )

    def test_matches_sensitive_denylist_non_sensitive_files(self):
        """Test _matches_sensitive_denylist returns False for normal files."""
        from aci.core.file_scanner import FileScanner

        scanner = FileScanner()

        # Test non-sensitive files
        non_sensitive = [
            "main.py",
            "config.yaml",
            "README.md",
            "package.json",
            "Dockerfile",
            "requirements.txt",
            "test_env.py",  # Contains "env" but not a .env file
            "ssh_utils.py",  # Contains "ssh" but not .ssh
            "key_manager.py",  # Contains "key" but not *.key
        ]

        for filename in non_sensitive:
            path = Path(f"/some/path/{filename}")
            assert not scanner._matches_sensitive_denylist(path), (
                f"File '{filename}' should NOT match sensitive denylist"
            )

    def test_all_required_patterns_present(self):
        """Comprehensive test that all required patterns from requirements are present."""
        from aci.core.file_scanner import SENSITIVE_DENYLIST

        # All patterns required by Requirements 2.4
        required_patterns = {
            # SSH and GPG
            ".ssh",
            ".gnupg",
            # SSH keys
            "id_rsa",
            "id_ed25519",
            # Additional SSH keys from task
            "id_rsa.pub",
            "id_ed25519.pub",
            "id_ecdsa",
            "id_dsa",
            # Certificates and keys
            "*.pem",
            "*.key",
            "*.p12",
            "*.pfx",
            "*.crt",
            "*.keystore",
            # Environment files
            ".env",
            ".env.*",
            # Other sensitive files
            ".netrc",
            ".npmrc",
            ".pypirc",
        }

        missing_patterns = required_patterns - SENSITIVE_DENYLIST
        assert not missing_patterns, (
            f"Missing required patterns in SENSITIVE_DENYLIST: {missing_patterns}"
        )
