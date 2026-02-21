"""
Path validation utilities for ACI.

Provides centralized path validation, system directory detection,
directory creation utilities, and collection name generation
used across CLI, REPL, and HTTP layers.
"""

import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath


@dataclass
class PathValidationResult:
    """Result of path validation.

    Attributes:
        valid: True if the path is valid for the requested operation.
        error_message: Human-readable error message if validation failed.
    """
    valid: bool
    error_message: str | None = None


# POSIX system directories that should not be indexed
POSIX_SYSTEM_DIRS = frozenset([
    "/etc",
    "/var",
    "/usr",
    "/bin",
    "/sbin",
    "/proc",
    "/sys",
    "/dev",
    "/root",
    "/boot",
    "/lib",
    "/lib64",
])

# Windows system directory names (case-insensitive)
WINDOWS_SYSTEM_DIRS = frozenset([
    "windows",
    "program files",
    "program files (x86)",
    "programdata",
    "system32",
    "syswow64",
])


def is_system_directory(path: Path) -> bool:
    """
    Check if a path is a protected system directory.

    Platform-aware: checks POSIX paths on Unix-like systems,
    Windows paths on Windows.

    Args:
        path: Path to check (will be resolved to absolute).

    Returns:
        True if the path is under a system directory, False otherwise.
    """
    try:
        resolved = path.resolve()
        path_str = str(resolved)

        if sys.platform == "win32":
            return _is_windows_system_directory(resolved, path_str)
        else:
            return _is_posix_system_directory(path_str)
    except (OSError, ValueError):
        # If we can't resolve the path, err on the side of caution
        return False


def _is_posix_system_directory(path_str: str) -> bool:
    """Check if path is under a POSIX system directory."""
    for sys_dir in POSIX_SYSTEM_DIRS:
        if path_str == sys_dir or path_str.startswith(sys_dir + "/"):
            return True
    return False


def _is_windows_system_directory(resolved: Path, path_str: str) -> bool:
    """Check if path is under a Windows system directory."""
    # Parse as Windows path explicitly so POSIX hosts correctly split
    # strings like "C:\\Windows" into drive + folders.
    windows_path = PureWindowsPath(path_str)
    win_parts = [part.lower() for part in windows_path.parts if part not in ("\\", "/")]

    # Strip drive prefix (e.g., "c:") if present.
    if win_parts and re.fullmatch(r"[a-z]:", win_parts[0]):
        win_parts = win_parts[1:]

    if win_parts and win_parts[0] in WINDOWS_SYSTEM_DIRS:
        return True

    # Fallback for already-normalized native Path objects.
    resolved_parts = [part.lower() for part in resolved.parts]
    if len(resolved_parts) >= 2 and resolved_parts[1] in WINDOWS_SYSTEM_DIRS:
        return True

    return any(part in WINDOWS_SYSTEM_DIRS for part in resolved_parts)


def validate_indexable_path(path: str | Path) -> PathValidationResult:
    """
    Validate that a path is suitable for indexing.

    Performs the following checks:
    1. Path exists
    2. Path is a directory
    3. Path is not a system directory

    Args:
        path: Path to validate (string or Path object).

    Returns:
        PathValidationResult with valid=True if all checks pass,
        or valid=False with an appropriate error message.
    """
    try:
        p = Path(path) if isinstance(path, str) else path

        # Check existence
        if not p.exists():
            return PathValidationResult(
                valid=False,
                error_message=f"Path '{path}' does not exist"
            )

        # Check if directory
        if not p.is_dir():
            return PathValidationResult(
                valid=False,
                error_message=f"Path '{path}' is not a directory"
            )

        # Check if system directory
        if is_system_directory(p):
            return PathValidationResult(
                valid=False,
                error_message="Indexing system directories is forbidden"
            )

        return PathValidationResult(valid=True)

    except (OSError, ValueError) as e:
        return PathValidationResult(
            valid=False,
            error_message=f"Invalid path '{path}': {e}"
        )


def ensure_directory_exists(path: Path) -> bool:
    """
    Ensure a directory exists, creating it if necessary.

    Creates the directory and all parent directories if they don't exist.

    Args:
        path: Path to the directory to ensure exists.

    Returns:
        True if the directory exists or was created successfully,
        False if creation failed (e.g., permission error).
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except (OSError, PermissionError):
        return False


def generate_collection_name(root_path: Path | str, prefix: str = "aci") -> str:
    """
    Generate a unique Qdrant collection name for a repository.

    Creates a deterministic collection name based on the absolute path,
    ensuring each repository has its own isolated collection.

    The format is: {prefix}_{sanitized_name}_{hash}
    - prefix: configurable prefix (default "aci")
    - sanitized_name: last directory component, sanitized for Qdrant
    - hash: first 8 chars of SHA-256 hash of the full path

    Args:
        root_path: Root path of the repository.
        prefix: Prefix for the collection name.

    Returns:
        A valid Qdrant collection name (alphanumeric, underscores, max 64 chars).

    Example:
        >>> generate_collection_name("/home/user/my-project")
        'aci_my_project_a1b2c3d4'
    """
    path = Path(root_path).resolve()
    path_str = str(path)

    # Generate hash of full path for uniqueness
    path_hash = hashlib.sha256(path_str.encode("utf-8")).hexdigest()[:8]

    # Get the last directory component as a readable name
    dir_name = path.name or "root"

    # Sanitize: replace non-alphanumeric with underscore, lowercase
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", dir_name).lower()
    # Remove consecutive underscores and trim
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    # Limit length to leave room for prefix and hash
    max_name_len = 64 - len(prefix) - 1 - 8 - 1  # prefix_name_hash
    sanitized = sanitized[:max_name_len]

    return f"{prefix}_{sanitized}_{path_hash}"


def get_collection_name_for_path(root_path: Path | str) -> str:
    """
    Get the collection name for a repository path.

    Convenience wrapper around generate_collection_name with default prefix.

    Args:
        root_path: Root path of the repository.

    Returns:
        Qdrant collection name for this repository.
    """
    return generate_collection_name(root_path)
