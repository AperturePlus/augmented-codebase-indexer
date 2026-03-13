"""
Path validation utilities for ACI.

Provides centralized path validation, runtime path resolution,
system directory detection, directory creation utilities, and
collection name generation used across CLI, REPL, HTTP, and MCP layers.
"""

import hashlib
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


@dataclass
class PathValidationResult:
    """Result of path validation.

    Attributes:
        valid: True if the path is valid for the requested operation.
        error_message: Human-readable error message if validation failed.
    """
    valid: bool
    error_message: str | None = None


@dataclass(frozen=True)
class RuntimePathMapping:
    """Maps a host-side path prefix to a runtime-accessible path prefix."""

    source_prefix: str
    target_prefix: Path


@dataclass
class RuntimePathResolutionResult:
    """Result of resolving a user-supplied path inside the current runtime."""

    valid: bool
    original_path: str
    resolved_path: Path | None = None
    mapped: bool = False
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

_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[a-zA-Z]:[\\/]")


@dataclass(frozen=True)
class _ComparablePath:
    style: str
    absolute: bool
    normalized_parts: tuple[str, ...]
    raw_parts: tuple[str, ...]


def _looks_like_windows_path(path_str: str) -> bool:
    """Return True when a string looks like a Windows absolute path."""
    return bool(_WINDOWS_ABSOLUTE_PATH_RE.match(path_str)) or path_str.startswith("\\\\")


def _split_windows_parts(path_str: str) -> _ComparablePath:
    """Split a Windows path into comparable parts."""
    pure_path = PureWindowsPath(path_str)
    raw_parts: list[str] = []

    if pure_path.drive:
        raw_parts.append(pure_path.drive)

    anchor = pure_path.anchor.rstrip("\\/")
    for part in pure_path.parts:
        cleaned = part.rstrip("\\/")
        if not cleaned or cleaned == anchor or cleaned == pure_path.drive:
            continue
        raw_parts.append(cleaned)

    return _ComparablePath(
        style="windows",
        absolute=pure_path.is_absolute(),
        normalized_parts=tuple(part.lower() for part in raw_parts),
        raw_parts=tuple(raw_parts),
    )


def _split_posix_parts(path_str: str) -> _ComparablePath:
    """Split a POSIX path into comparable parts."""
    pure_path = PurePosixPath(path_str.replace("\\", "/"))
    parts = [part for part in pure_path.parts if part not in ("", ".")]
    if pure_path.is_absolute():
        raw_parts = tuple(["/"] + [part for part in parts if part != "/"])
    else:
        raw_parts = tuple(part for part in parts if part != "/")

    return _ComparablePath(
        style="posix",
        absolute=pure_path.is_absolute(),
        normalized_parts=raw_parts,
        raw_parts=raw_parts,
    )


def _to_comparable_path(path_str: str) -> _ComparablePath:
    """Convert a raw path string to comparable parts."""
    if _looks_like_windows_path(path_str):
        return _split_windows_parts(path_str)
    return _split_posix_parts(path_str)


def parse_runtime_path_mappings(raw_value: str | None) -> list[RuntimePathMapping]:
    """Parse semicolon-separated runtime path mappings.

    Expected format:
        source_prefix=target_prefix;source_prefix=target_prefix

    Examples:
        D:\\=/host/d;/Users/alice=/host/users/alice
        /=/hostfs
    """
    if raw_value is None or not raw_value.strip():
        return []

    mappings: list[RuntimePathMapping] = []
    for item in raw_value.split(";"):
        pair = item.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                "Invalid path mapping entry. Expected 'source=target' pairs separated by ';'."
            )

        source_prefix, target_prefix = pair.split("=", 1)
        source_prefix = source_prefix.strip()
        target_prefix = target_prefix.strip()
        if not source_prefix or not target_prefix:
            raise ValueError("Path mapping source and target must be non-empty.")

        mappings.append(
            RuntimePathMapping(
                source_prefix=source_prefix,
                target_prefix=Path(target_prefix),
            )
        )

    return mappings


def _apply_runtime_path_mapping(
    path_str: str,
    path_mappings: Sequence[RuntimePathMapping],
) -> Path | None:
    """Apply the first matching runtime path mapping."""
    input_path = _to_comparable_path(path_str)

    for mapping in path_mappings:
        source = _to_comparable_path(mapping.source_prefix)
        if input_path.style != source.style or input_path.absolute != source.absolute:
            continue
        if len(input_path.normalized_parts) < len(source.normalized_parts):
            continue
        if input_path.normalized_parts[: len(source.normalized_parts)] != source.normalized_parts:
            continue

        target_path = mapping.target_prefix
        remainder = input_path.raw_parts[len(source.raw_parts) :]
        for part in remainder:
            target_path = target_path / part
        return target_path

    return None


def _resolve_mapped_runtime_path(
    original_path: str,
    path_mappings: Sequence[RuntimePathMapping],
) -> RuntimePathResolutionResult | None:
    """Resolve a path via configured runtime mappings when available."""
    mapped_path = _apply_runtime_path_mapping(original_path, path_mappings)
    if mapped_path is None:
        return None

    if mapped_path.exists():
        return RuntimePathResolutionResult(
            valid=True,
            original_path=original_path,
            resolved_path=mapped_path.resolve(),
            mapped=True,
        )

    return RuntimePathResolutionResult(
        valid=False,
        original_path=original_path,
        error_message=(
            f"Path '{original_path}' is not accessible inside this runtime. "
            f"Mapped to '{mapped_path}', but that path does not exist. "
            "Check the container bind mount and ACI_MCP_PATH_MAPPINGS."
        ),
    )


def resolve_runtime_path(
    path: str | Path,
    workspace_root: str | Path | None = None,
    path_mappings: Sequence[RuntimePathMapping] | None = None,
) -> RuntimePathResolutionResult:
    """Resolve a user-supplied path within the current runtime environment."""
    original_path = str(path)
    path_str = original_path.strip()
    mappings = path_mappings or ()

    if not path_str:
        return RuntimePathResolutionResult(
            valid=False,
            original_path=original_path,
            error_message="Path cannot be empty",
        )

    is_windows_absolute = _looks_like_windows_path(path_str)
    is_posix_absolute = path_str.startswith("/")
    is_absolute = is_windows_absolute or is_posix_absolute

    if not is_absolute:
        base_dir = Path(workspace_root) if workspace_root is not None else Path.cwd()
        return RuntimePathResolutionResult(
            valid=True,
            original_path=original_path,
            resolved_path=(base_dir / path_str).resolve(),
            mapped=workspace_root is not None,
        )

    mapped_resolution = _resolve_mapped_runtime_path(original_path, mappings)
    if mapped_resolution is not None:
        return mapped_resolution

    if is_windows_absolute and sys.platform == "win32":
        return RuntimePathResolutionResult(
            valid=True,
            original_path=original_path,
            resolved_path=Path(path_str),
        )

    if is_posix_absolute and sys.platform != "win32":
        return RuntimePathResolutionResult(
            valid=True,
            original_path=original_path,
            resolved_path=Path(path_str),
        )

    return RuntimePathResolutionResult(
        valid=False,
        original_path=original_path,
        error_message=(
            f"Path '{original_path}' is not accessible inside this runtime. "
            "Configure ACI_MCP_PATH_MAPPINGS or mount the host path into the container."
        ),
    )


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

    # Strip Windows drive/root prefixes (e.g., "c:\\" -> "c:") so the
    # first semantic directory component can be matched reliably.
    if win_parts:
        drive = win_parts[0].rstrip("\\/")
        if re.fullmatch(r"[a-z]:", drive):
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
