"""
Unit tests for GitignoreManager error handling scenarios.

Tests graceful handling of:
- Invalid UTF-8 encoding
- Malformed patterns
- Permission errors
- Empty files

**Feature: gitignore-path-traversal-fix**
**Validates: Requirements 4.1, 4.2, 4.3, 4.4**
"""

import logging
import os
import stat
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from aci.core.gitignore_manager import GitignoreManager


class TestInvalidUTF8Handling:
    """
    Test handling of invalid UTF-8 encoding in .gitignore files.
    **Validates: Requirements 4.1**
    """

    def test_invalid_utf8_returns_zero_patterns(self):
        """GitignoreManager should return 0 patterns when file has invalid UTF-8."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gitignore_path = tmpdir_path / ".gitignore"

            # Write invalid UTF-8 bytes
            gitignore_path.write_bytes(b"valid_pattern\n\xff\xfe invalid bytes\n")

            manager = GitignoreManager(tmpdir_path)
            loaded = manager.load_gitignore(gitignore_path)

            assert loaded == 0, "Should return 0 patterns for invalid UTF-8 file"
            assert manager.pattern_count == 0

    def test_invalid_utf8_logs_warning(self, caplog):
        """GitignoreManager should log a warning for invalid UTF-8 files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gitignore_path = tmpdir_path / ".gitignore"

            # Write invalid UTF-8 bytes
            gitignore_path.write_bytes(b"\xff\xfe\x00\x01")

            manager = GitignoreManager(tmpdir_path)
            with caplog.at_level(logging.WARNING):
                manager.load_gitignore(gitignore_path)

            assert any(
                "Invalid UTF-8 encoding" in record.message
                for record in caplog.records
            ), "Should log warning about invalid UTF-8"

    def test_invalid_utf8_allows_continued_scanning(self):
        """After invalid UTF-8 error, manager should still work for other files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create invalid UTF-8 gitignore
            bad_gitignore = tmpdir_path / ".gitignore"
            bad_gitignore.write_bytes(b"\xff\xfe invalid")

            # Create valid gitignore in subdirectory
            subdir = tmpdir_path / "subdir"
            subdir.mkdir()
            good_gitignore = subdir / ".gitignore"
            good_gitignore.write_text("valid_pattern\n", encoding="utf-8")

            manager = GitignoreManager(tmpdir_path)

            # Load bad file first - should fail gracefully
            loaded_bad = manager.load_gitignore(bad_gitignore)
            assert loaded_bad == 0

            # Load good file - should succeed
            loaded_good = manager.load_gitignore(good_gitignore)
            assert loaded_good == 1


class TestMalformedPatternHandling:
    """
    Test handling of malformed patterns in .gitignore files.
    **Validates: Requirements 4.2**
    """

    def test_empty_pattern_after_special_chars_skipped(self):
        """Patterns that are empty after removing ! and / should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gitignore_path = tmpdir_path / ".gitignore"

            # Write patterns that become empty after stripping special chars
            gitignore_path.write_text("!/\n!\nvalid_pattern\n", encoding="utf-8")

            manager = GitignoreManager(tmpdir_path)
            loaded = manager.load_gitignore(gitignore_path)

            # Only valid_pattern should be loaded
            assert loaded == 1, "Should only load valid patterns"

    def test_unbalanced_brackets_skipped(self, caplog):
        """Patterns with unbalanced brackets should be skipped with warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gitignore_path = tmpdir_path / ".gitignore"

            gitignore_path.write_text(
                "[unbalanced\nvalid_pattern\n[balanced]\n", encoding="utf-8"
            )

            manager = GitignoreManager(tmpdir_path)
            with caplog.at_level(logging.WARNING):
                loaded = manager.load_gitignore(gitignore_path)

            # valid_pattern and [balanced] should be loaded
            assert loaded == 2, "Should load valid patterns and skip malformed"
            assert any(
                "Unbalanced brackets" in record.message for record in caplog.records
            ), "Should log warning about unbalanced brackets"

    def test_trailing_backslash_skipped(self, caplog):
        """Patterns ending with incomplete escape should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gitignore_path = tmpdir_path / ".gitignore"

            gitignore_path.write_text(
                "incomplete\\\nvalid_pattern\n", encoding="utf-8"
            )

            manager = GitignoreManager(tmpdir_path)
            with caplog.at_level(logging.WARNING):
                loaded = manager.load_gitignore(gitignore_path)

            assert loaded == 1, "Should only load valid pattern"
            assert any(
                "incomplete escape" in record.message for record in caplog.records
            ), "Should log warning about incomplete escape"

    def test_malformed_patterns_dont_crash(self):
        """Various malformed patterns should not crash the parser."""
        malformed_patterns = [
            "!/",  # Empty after stripping
            "!",  # Just negation
            "/",  # Just anchor
            "[",  # Unbalanced bracket
            "]",  # Unbalanced bracket
            "[[",  # Multiple unbalanced
            "pattern\\",  # Trailing backslash
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gitignore_path = tmpdir_path / ".gitignore"

            content = "\n".join(malformed_patterns) + "\nvalid_pattern\n"
            gitignore_path.write_text(content, encoding="utf-8")

            manager = GitignoreManager(tmpdir_path)
            # Should not raise any exceptions
            loaded = manager.load_gitignore(gitignore_path)

            # At least valid_pattern should be loaded
            assert loaded >= 1, "Should load at least the valid pattern"


class TestPermissionErrorHandling:
    """
    Test handling of permission errors when reading .gitignore files.
    **Validates: Requirements 4.3**
    """

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Permission tests behave differently on Windows",
    )
    def test_permission_denied_returns_zero_patterns(self):
        """GitignoreManager should return 0 patterns when file is unreadable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gitignore_path = tmpdir_path / ".gitignore"

            # Create file and remove read permissions
            gitignore_path.write_text("pattern\n", encoding="utf-8")
            os.chmod(gitignore_path, 0o000)

            try:
                manager = GitignoreManager(tmpdir_path)
                loaded = manager.load_gitignore(gitignore_path)

                assert loaded == 0, "Should return 0 for unreadable file"
            finally:
                # Restore permissions for cleanup
                os.chmod(gitignore_path, stat.S_IRUSR | stat.S_IWUSR)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Permission tests behave differently on Windows",
    )
    def test_permission_denied_logs_warning(self, caplog):
        """GitignoreManager should log warning for permission denied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gitignore_path = tmpdir_path / ".gitignore"

            gitignore_path.write_text("pattern\n", encoding="utf-8")
            os.chmod(gitignore_path, 0o000)

            try:
                manager = GitignoreManager(tmpdir_path)
                with caplog.at_level(logging.WARNING):
                    manager.load_gitignore(gitignore_path)

                assert any(
                    "Permission denied" in record.message
                    for record in caplog.records
                ), "Should log warning about permission denied"
            finally:
                os.chmod(gitignore_path, stat.S_IRUSR | stat.S_IWUSR)

    def test_permission_error_via_mock(self, caplog):
        """Test permission error handling using mock (works on all platforms)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gitignore_path = tmpdir_path / ".gitignore"
            gitignore_path.write_text("pattern\n", encoding="utf-8")

            manager = GitignoreManager(tmpdir_path)

            with patch.object(
                Path, "read_text", side_effect=PermissionError("Access denied")
            ):
                with caplog.at_level(logging.WARNING):
                    loaded = manager.load_gitignore(gitignore_path)

            assert loaded == 0
            assert any(
                "Permission denied" in record.message for record in caplog.records
            )


class TestEmptyFileHandling:
    """
    Test handling of empty .gitignore files.
    **Validates: Requirements 4.4**
    """

    def test_empty_file_returns_zero_patterns(self):
        """Empty .gitignore file should return 0 patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gitignore_path = tmpdir_path / ".gitignore"

            # Create empty file
            gitignore_path.write_text("", encoding="utf-8")

            manager = GitignoreManager(tmpdir_path)
            loaded = manager.load_gitignore(gitignore_path)

            assert loaded == 0, "Empty file should return 0 patterns"

    def test_whitespace_only_file_returns_zero_patterns(self):
        """File with only whitespace should return 0 patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gitignore_path = tmpdir_path / ".gitignore"

            # Create file with only whitespace
            gitignore_path.write_text("   \n\t\n  \n", encoding="utf-8")

            manager = GitignoreManager(tmpdir_path)
            loaded = manager.load_gitignore(gitignore_path)

            assert loaded == 0, "Whitespace-only file should return 0 patterns"

    def test_comments_only_file_returns_zero_patterns(self):
        """File with only comments should return 0 patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gitignore_path = tmpdir_path / ".gitignore"

            gitignore_path.write_text(
                "# This is a comment\n# Another comment\n", encoding="utf-8"
            )

            manager = GitignoreManager(tmpdir_path)
            loaded = manager.load_gitignore(gitignore_path)

            assert loaded == 0, "Comments-only file should return 0 patterns"

    def test_empty_file_allows_continued_operation(self):
        """After loading empty file, manager should still work normally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create empty gitignore
            empty_gitignore = tmpdir_path / ".gitignore"
            empty_gitignore.write_text("", encoding="utf-8")

            # Create valid gitignore in subdirectory
            subdir = tmpdir_path / "subdir"
            subdir.mkdir()
            valid_gitignore = subdir / ".gitignore"
            valid_gitignore.write_text("pattern\n", encoding="utf-8")

            manager = GitignoreManager(tmpdir_path)

            # Load empty file
            loaded_empty = manager.load_gitignore(empty_gitignore)
            assert loaded_empty == 0

            # Load valid file - should work
            loaded_valid = manager.load_gitignore(valid_gitignore)
            assert loaded_valid == 1


class TestFileNotFoundHandling:
    """Test handling of non-existent .gitignore files."""

    def test_nonexistent_file_returns_zero_patterns(self):
        """Non-existent .gitignore should return 0 patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gitignore_path = tmpdir_path / ".gitignore"

            # Don't create the file
            manager = GitignoreManager(tmpdir_path)
            loaded = manager.load_gitignore(gitignore_path)

            assert loaded == 0, "Non-existent file should return 0 patterns"

    def test_nonexistent_file_logs_debug(self, caplog):
        """Non-existent .gitignore should log at debug level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gitignore_path = tmpdir_path / ".gitignore"

            manager = GitignoreManager(tmpdir_path)
            with caplog.at_level(logging.DEBUG):
                manager.load_gitignore(gitignore_path)

            assert any(
                "not found" in record.message.lower() for record in caplog.records
            ), "Should log debug message about file not found"


class TestOSErrorHandling:
    """Test handling of general OS errors."""

    def test_os_error_returns_zero_patterns(self, caplog):
        """General OS errors should return 0 patterns and log warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            gitignore_path = tmpdir_path / ".gitignore"
            gitignore_path.write_text("pattern\n", encoding="utf-8")

            manager = GitignoreManager(tmpdir_path)

            with patch.object(
                Path, "read_text", side_effect=OSError("Disk error")
            ):
                with caplog.at_level(logging.WARNING):
                    loaded = manager.load_gitignore(gitignore_path)

            assert loaded == 0
            assert any(
                "Error reading" in record.message for record in caplog.records
            )


class TestVerboseSummary:
    """Test verbose summary logging functionality."""

    def test_get_loaded_gitignore_summary(self):
        """get_loaded_gitignore_summary should return correct counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create root gitignore with 2 patterns
            root_gitignore = tmpdir_path / ".gitignore"
            root_gitignore.write_text("pattern1\npattern2\n", encoding="utf-8")

            # Create subdir gitignore with 1 pattern
            subdir = tmpdir_path / "subdir"
            subdir.mkdir()
            subdir_gitignore = subdir / ".gitignore"
            subdir_gitignore.write_text("pattern3\n", encoding="utf-8")

            manager = GitignoreManager(tmpdir_path)
            manager.load_gitignore(root_gitignore)
            manager.load_gitignore(subdir_gitignore)

            summary = manager.get_loaded_gitignore_summary()

            assert len(summary) == 2, "Should have 2 gitignore files"
            assert summary[root_gitignore.resolve()] == 2
            assert summary[subdir_gitignore.resolve()] == 1

    def test_log_verbose_summary(self, caplog):
        """log_verbose_summary should log info about loaded files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            gitignore_path = tmpdir_path / ".gitignore"
            gitignore_path.write_text("pattern\n", encoding="utf-8")

            manager = GitignoreManager(tmpdir_path)
            manager.load_gitignore(gitignore_path)

            with caplog.at_level(logging.INFO):
                manager.log_verbose_summary()

            assert any(
                "1 pattern" in record.message for record in caplog.records
            ), "Should log pattern count"

    def test_log_verbose_summary_empty(self, caplog):
        """log_verbose_summary should handle no loaded files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            manager = GitignoreManager(tmpdir_path)

            with caplog.at_level(logging.INFO):
                manager.log_verbose_summary()

            assert any(
                "No .gitignore files loaded" in record.message
                for record in caplog.records
            )
