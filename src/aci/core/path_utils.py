"""
Path validation utilities for ACI.

Provides centralized path validation, system directory detection,
and directory creation utilities used across CLI, REPL, and HTTP layers.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PathValidationResult:
    """Result of path validation.
    
    Attributes:
        valid: True if the path is valid for the requested operation.
        error_message: Human-readable error message if validation failed.
    """
    valid: bool
    error_message: Optional[str] = None


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
    # Get path parts for case-insensitive comparison
    parts_lower = [p.lower() for p in resolved.parts]
    
    # Check if any part matches a known system directory name
    for part in parts_lower:
        if part in WINDOWS_SYSTEM_DIRS:
            return True
    
    # Also check the drive root system directories
    # e.g., C:\Windows, D:\Windows (though rare)
    if len(resolved.parts) >= 2:
        # parts[0] is drive like 'C:\', parts[1] is first folder
        first_folder = resolved.parts[1].lower()
        if first_folder in WINDOWS_SYSTEM_DIRS:
            return True
    
    return False


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
