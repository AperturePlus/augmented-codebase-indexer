"""
Property-based tests for SymlinkValidator symlink security validation.

**Feature: gitignore-path-traversal-fix**
"""

import sys
import tempfile
from pathlib import Path

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from aci.core.symlink_validator import SymlinkValidator

# Skip all tests on Windows - symlinks require admin privileges
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Symlink tests require admin privileges on Windows"
)


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Valid directory/file name characters
name_chars = st.sampled_from(
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
)

# Simple name (like a filename or directory name)
simple_name = st.text(name_chars, min_size=1, max_size=15).filter(
    lambda s: s not in {".", ".."} and not s.startswith("-")
)

# Sensitive directory names from the default patterns
sensitive_names = st.sampled_from([
    ".ssh",
    ".gnupg",
    "id_rsa",
    "id_ed25519",
    ".env",
    ".netrc",
    ".npmrc",
    ".pypirc",
])


# =============================================================================
# Property Tests for Symlink Resolution (Property 9)
# =============================================================================


@given(target_name=simple_name)
@settings(max_examples=100, deadline=None)
def test_symlink_resolution_to_real_path(target_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 9: Symlink Resolution**
    **Validates: Requirements 3.1**

    For any symbolic link encountered during scanning, the SymlinkValidator
    SHALL resolve the link to its real path before processing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir).resolve()

        # Create a real file
        real_file = tmpdir_path / target_name
        real_file.write_text("content", encoding="utf-8")

        # Create a symlink to the real file
        symlink_path = tmpdir_path / f"link_to_{target_name}"
        symlink_path.symlink_to(real_file)

        # Create validator
        validator = SymlinkValidator(tmpdir_path)

        # Validate the symlink
        result = validator.is_safe_symlink(symlink_path)

        # Property: Symlink should be resolved and safe (within root, not sensitive)
        assert result.safe, f"Symlink to valid file should be safe: {result.reason}"
        assert result.target_path is not None, "Safe symlink should have target_path"
        assert result.target_path.resolve() == real_file.resolve(), (
            f"Target path should be resolved to real path: "
            f"expected {real_file.resolve()}, got {result.target_path.resolve()}"
        )


@given(target_name=simple_name)
@settings(max_examples=100, deadline=None)
def test_symlink_resolution_to_directory(target_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 9: Symlink Resolution**
    **Validates: Requirements 3.1**

    Tests that symlinks to directories are properly resolved.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir).resolve()

        # Create a real directory
        real_dir = tmpdir_path / target_name
        real_dir.mkdir()

        # Create a symlink to the directory
        symlink_path = tmpdir_path / f"link_to_{target_name}"
        symlink_path.symlink_to(real_dir)

        # Create validator
        validator = SymlinkValidator(tmpdir_path)

        # Validate the symlink
        result = validator.is_safe_symlink(symlink_path)

        # Property: Symlink to directory should be resolved
        assert result.safe, f"Symlink to valid directory should be safe: {result.reason}"
        assert result.target_path is not None
        assert result.target_path.resolve() == real_dir.resolve()


@given(target_name=simple_name)
@settings(max_examples=100, deadline=None)
def test_broken_symlink_detection(target_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 9: Symlink Resolution**
    **Validates: Requirements 3.1**

    Tests that broken symlinks (target doesn't exist) are detected.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir).resolve()

        # Create a symlink to a non-existent target
        non_existent = tmpdir_path / target_name
        symlink_path = tmpdir_path / f"broken_link_{target_name}"
        symlink_path.symlink_to(non_existent)

        # Create validator
        validator = SymlinkValidator(tmpdir_path)

        # Validate the symlink
        result = validator.is_safe_symlink(symlink_path)

        # Property: Broken symlink should be detected as unsafe
        assert not result.safe, "Broken symlink should be unsafe"
        assert result.target_path is None
        assert "target does not exist" in result.reason.lower() or "broken" in result.reason.lower()


# =============================================================================
# Property Tests for Symlink Boundary Enforcement (Property 10)
# =============================================================================


@given(subdir_name=simple_name, target_name=simple_name)
@settings(max_examples=100, deadline=None)
def test_symlink_boundary_enforcement_outside_root(subdir_name, target_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 10: Symlink Boundary Enforcement**
    **Validates: Requirements 3.2**

    For any symlink whose target resolves to a path outside the root directory,
    the SymlinkValidator SHALL skip the symlink and not traverse into it.
    """
    assume(subdir_name != target_name)

    with tempfile.TemporaryDirectory() as tmpdir1:
        with tempfile.TemporaryDirectory() as tmpdir2:
            root_path = Path(tmpdir1).resolve()
            outside_path = Path(tmpdir2).resolve()

            # Create a file outside the root
            outside_file = outside_path / target_name
            outside_file.write_text("outside content", encoding="utf-8")

            # Create a symlink inside root pointing outside
            symlink_path = root_path / f"escape_link_{target_name}"
            symlink_path.symlink_to(outside_file)

            # Create validator with root boundary
            validator = SymlinkValidator(root_path)

            # Validate the symlink
            result = validator.is_safe_symlink(symlink_path)

            # Property: Symlink escaping root should be blocked
            assert not result.safe, (
                f"Symlink pointing outside root should be unsafe: {symlink_path} -> {outside_file}"
            )
            assert result.target_path is None
            assert "outside" in result.reason.lower()


@given(target_name=simple_name)
@settings(max_examples=100, deadline=None)
def test_symlink_boundary_enforcement_parent_escape(target_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 10: Symlink Boundary Enforcement**
    **Validates: Requirements 3.2**

    Tests that symlinks using .. to escape root are blocked.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir).resolve()

        # Create a subdirectory as our "root"
        subdir = tmpdir_path / "subroot"
        subdir.mkdir()

        # Create a file in the parent (outside subroot)
        parent_file = tmpdir_path / target_name
        parent_file.write_text("parent content", encoding="utf-8")

        # Create a symlink in subroot pointing to parent via ..
        symlink_path = subdir / f"parent_link_{target_name}"
        symlink_path.symlink_to(Path("..") / target_name)

        # Create validator with subdir as root
        validator = SymlinkValidator(subdir)

        # Validate the symlink
        result = validator.is_safe_symlink(symlink_path)

        # Property: Symlink escaping via .. should be blocked
        assert not result.safe, (
            f"Symlink escaping via .. should be unsafe: {symlink_path}"
        )


@given(target_name=simple_name)
@settings(max_examples=100, deadline=None)
def test_symlink_within_root_is_safe(target_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 10: Symlink Boundary Enforcement**
    **Validates: Requirements 3.2**

    Tests that symlinks staying within root are allowed.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir).resolve()

        # Create subdirectories
        subdir1 = tmpdir_path / "subdir1"
        subdir2 = tmpdir_path / "subdir2"
        subdir1.mkdir()
        subdir2.mkdir()

        # Create a file in subdir1
        target_file = subdir1 / target_name
        target_file.write_text("content", encoding="utf-8")

        # Create a symlink in subdir2 pointing to file in subdir1
        symlink_path = subdir2 / f"link_{target_name}"
        symlink_path.symlink_to(target_file)

        # Create validator
        validator = SymlinkValidator(tmpdir_path)

        # Validate the symlink
        result = validator.is_safe_symlink(symlink_path)

        # Property: Symlink within root should be safe
        assert result.safe, f"Symlink within root should be safe: {result.reason}"
        assert result.target_path is not None


# =============================================================================
# Property Tests for Symlink Cycle Detection (Property 11)
# =============================================================================


@given(dir_name=simple_name)
@settings(max_examples=100, deadline=None)
def test_symlink_cycle_detection_direct(dir_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 11: Symlink Cycle Detection**
    **Validates: Requirements 3.3**

    For any symlink that creates a cycle back to an ancestor directory,
    the SymlinkValidator SHALL detect the cycle and skip the symlink.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir).resolve()

        # Create a directory
        subdir = tmpdir_path / dir_name
        subdir.mkdir()

        # Create a symlink inside subdir pointing back to root
        symlink_path = subdir / "cycle_link"
        symlink_path.symlink_to(tmpdir_path)

        # Create validator
        validator = SymlinkValidator(tmpdir_path)

        # Simulate visited set containing the root (as if we're traversing from root)
        visited = {tmpdir_path}

        # Validate the symlink
        result = validator.is_safe_symlink(symlink_path, visited=visited)

        # Property: Symlink creating cycle should be detected
        assert not result.safe, (
            f"Symlink creating cycle back to root should be unsafe: {symlink_path}"
        )
        assert "cycle" in result.reason.lower()


@given(dir_name=simple_name)
@settings(max_examples=100, deadline=None)
def test_symlink_cycle_detection_to_parent(dir_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 11: Symlink Cycle Detection**
    **Validates: Requirements 3.3**

    Tests that symlinks pointing to parent directories are detected as cycles.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir).resolve()

        # Create nested directories
        level1 = tmpdir_path / dir_name
        level2 = level1 / "level2"
        level1.mkdir()
        level2.mkdir()

        # Create a symlink in level2 pointing to level1
        symlink_path = level2 / "parent_link"
        symlink_path.symlink_to(level1)

        # Create validator
        validator = SymlinkValidator(tmpdir_path)

        # Simulate visited set containing ancestors
        visited = {tmpdir_path, level1}

        # Validate the symlink
        result = validator.is_safe_symlink(symlink_path, visited=visited)

        # Property: Symlink to parent should be detected as cycle
        assert not result.safe, (
            f"Symlink to parent directory should be detected as cycle: {symlink_path}"
        )


@given(dir_name=simple_name)
@settings(max_examples=100, deadline=None)
def test_symlink_no_cycle_without_visited(dir_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 11: Symlink Cycle Detection**
    **Validates: Requirements 3.3**

    Tests that cycle detection is skipped when no visited set is provided.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir).resolve()

        # Create a directory
        subdir = tmpdir_path / dir_name
        subdir.mkdir()

        # Create a symlink inside subdir pointing back to root
        symlink_path = subdir / "cycle_link"
        symlink_path.symlink_to(tmpdir_path)

        # Create validator
        validator = SymlinkValidator(tmpdir_path)

        # Validate WITHOUT visited set - cycle detection should be skipped
        result = validator.is_safe_symlink(symlink_path, visited=None)

        # Property: Without visited set, cycle detection is skipped
        # The symlink is still within root, so it should be safe
        assert result.safe, (
            f"Symlink should be safe when cycle detection is disabled: {result.reason}"
        )


# =============================================================================
# Property Tests for Sensitive Directory Protection (Property 12)
# =============================================================================


@given(sensitive_name=sensitive_names)
@settings(max_examples=100, deadline=None)
def test_symlink_sensitive_directory_protection(sensitive_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 12: Symlink Sensitive Directory Protection**
    **Validates: Requirements 3.4**

    For any symlink pointing to a sensitive directory (matching SENSITIVE_DENYLIST),
    the SymlinkValidator SHALL skip the symlink regardless of its location within the root.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir).resolve()

        # Create a "sensitive" directory within root
        sensitive_dir = tmpdir_path / sensitive_name
        sensitive_dir.mkdir()

        # Create a symlink pointing to the sensitive directory
        symlink_path = tmpdir_path / f"link_to_{sensitive_name.replace('.', '_')}"
        symlink_path.symlink_to(sensitive_dir)

        # Create validator
        validator = SymlinkValidator(tmpdir_path)

        # Validate the symlink
        result = validator.is_safe_symlink(symlink_path)

        # Property: Symlink to sensitive directory should be blocked
        assert not result.safe, (
            f"Symlink to sensitive directory '{sensitive_name}' should be unsafe"
        )
        assert "sensitive" in result.reason.lower()


@given(sensitive_name=sensitive_names, subdir_name=simple_name)
@settings(max_examples=100, deadline=None)
def test_symlink_sensitive_nested_protection(sensitive_name, subdir_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 12: Symlink Sensitive Directory Protection**
    **Validates: Requirements 3.4**

    Tests that symlinks to files inside sensitive directories are blocked.
    """
    assume(subdir_name != sensitive_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir).resolve()

        # Create a sensitive directory with a file inside
        sensitive_dir = tmpdir_path / sensitive_name
        sensitive_dir.mkdir()
        nested_file = sensitive_dir / "secret_file"
        nested_file.write_text("secret", encoding="utf-8")

        # Create a symlink pointing to the file inside sensitive directory
        symlink_path = tmpdir_path / "link_to_secret"
        symlink_path.symlink_to(nested_file)

        # Create validator
        validator = SymlinkValidator(tmpdir_path)

        # Validate the symlink
        result = validator.is_safe_symlink(symlink_path)

        # Property: Symlink to file inside sensitive directory should be blocked
        assert not result.safe, (
            f"Symlink to file inside sensitive directory '{sensitive_name}' should be unsafe"
        )


@given(target_name=simple_name)
@settings(max_examples=100, deadline=None)
def test_symlink_non_sensitive_allowed(target_name):
    """
    **Feature: gitignore-path-traversal-fix, Property 12: Symlink Sensitive Directory Protection**
    **Validates: Requirements 3.4**

    Tests that symlinks to non-sensitive directories are allowed.
    """
    # Ensure target_name is not in sensitive patterns
    assume(target_name not in {".ssh", ".gnupg", ".env", ".netrc", ".npmrc", ".pypirc"})
    assume(not target_name.startswith("id_"))
    assume(not target_name.endswith((".pem", ".key", ".p12", ".pfx", ".crt", ".keystore")))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir).resolve()

        # Create a non-sensitive directory
        normal_dir = tmpdir_path / target_name
        normal_dir.mkdir()

        # Create a symlink pointing to the normal directory
        symlink_path = tmpdir_path / f"link_to_{target_name}"
        symlink_path.symlink_to(normal_dir)

        # Create validator
        validator = SymlinkValidator(tmpdir_path)

        # Validate the symlink
        result = validator.is_safe_symlink(symlink_path)

        # Property: Symlink to non-sensitive directory should be safe
        assert result.safe, (
            f"Symlink to non-sensitive directory '{target_name}' should be safe: {result.reason}"
        )


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


def test_non_symlink_path_returns_unsafe():
    """
    Tests that passing a non-symlink path returns unsafe result.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir).resolve()

        # Create a regular file
        regular_file = tmpdir_path / "regular_file"
        regular_file.write_text("content", encoding="utf-8")

        # Create validator
        validator = SymlinkValidator(tmpdir_path)

        # Validate the regular file (not a symlink)
        result = validator.is_safe_symlink(regular_file)

        # Should return unsafe because it's not a symlink
        assert not result.safe
        assert "not a symlink" in result.reason.lower()


def test_custom_sensitive_patterns():
    """
    Tests that custom sensitive patterns can be provided.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir).resolve()

        # Create a directory that matches custom pattern
        custom_sensitive = tmpdir_path / "my_secrets"
        custom_sensitive.mkdir()

        # Create a symlink to it
        symlink_path = tmpdir_path / "link_to_secrets"
        symlink_path.symlink_to(custom_sensitive)

        # Create validator with custom sensitive patterns
        custom_patterns = frozenset(["my_secrets", "*.secret"])
        validator = SymlinkValidator(tmpdir_path, sensitive_patterns=custom_patterns)

        # Validate the symlink
        result = validator.is_safe_symlink(symlink_path)

        # Should be blocked by custom pattern
        assert not result.safe
        assert "sensitive" in result.reason.lower()
