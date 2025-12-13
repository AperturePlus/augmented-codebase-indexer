"""
Centralized repository path resolution for ACI.

Provides shared logic for validating repository paths and resolving
collection names across CLI, HTTP, and MCP interfaces.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aci.core.path_utils import get_collection_name_for_path
from aci.infrastructure.metadata_store import IndexMetadataStore


@dataclass
class RepositoryResolution:
    """
    Result of repository path resolution.

    Attributes:
        valid: True if the path is valid and indexed.
        collection_name: The collection name for the repository (if valid).
        indexed_root: The root path of the indexed repository (may differ
            from requested path if searching a subdirectory).
        error_message: Human-readable error message if resolution failed.
    """

    valid: bool
    collection_name: Optional[str] = None
    indexed_root: Optional[str] = None
    error_message: Optional[str] = None


def resolve_repository(
    path: Path | str,
    metadata_store: IndexMetadataStore,
) -> RepositoryResolution:
    """
    Validate a repository path and resolve its collection name.

    Performs the following checks:
    1. Path existence validation
    2. Directory validation
    3. Index existence check (including parent directories)
    4. Collection name retrieval or generation for legacy indexes

    Args:
        path: Path to validate and resolve.
        metadata_store: Metadata store for checking index status.

    Returns:
        RepositoryResolution with valid=True and collection_name if successful,
        or valid=False with an appropriate error_message.
    """
    # Convert to Path if string
    p = Path(path) if isinstance(path, str) else path

    # Check existence
    if not p.exists():
        return RepositoryResolution(
            valid=False,
            error_message=f"Path does not exist: {path}",
        )

    # Check if directory
    if not p.is_dir():
        return RepositoryResolution(
            valid=False,
            error_message=f"Path is not a directory: {path}",
        )

    # Resolve to absolute path
    path_abs = str(p.resolve())

    # Check if path (or a parent) is indexed
    index_info = metadata_store.find_parent_index(path_abs)
    if index_info is None:
        return RepositoryResolution(
            valid=False,
            error_message=f"Path has not been indexed: {path}. Run indexing first.",
        )

    # Get the indexed root path
    indexed_root = index_info.get("root_path", path_abs)

    # Get collection name, generating if needed for legacy indexes
    collection_name = index_info.get("collection_name")
    if not collection_name:
        # Legacy index without collection name - generate and register
        collection_name = get_collection_name_for_path(indexed_root)
        metadata_store.register_repository(indexed_root, collection_name)

    return RepositoryResolution(
        valid=True,
        collection_name=collection_name,
        indexed_root=indexed_root,
    )
