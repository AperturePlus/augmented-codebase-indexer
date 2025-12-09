"""
REPL context management module.

Manages the current codebase context for REPL sessions, allowing users
to set a working codebase for search operations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aci.infrastructure.metadata_store import IndexMetadataStore


@dataclass
class REPLContext:
    """
    Manages REPL session state including current codebase.
    
    The context tracks which codebase is currently selected for search
    operations. If no codebase is explicitly set, it defaults to the
    current working directory.
    
    Attributes:
        _current_codebase: The explicitly set codebase path, or None if not set.
        _metadata_store: Optional metadata store for checking indexed paths.
    """
    
    _current_codebase: Optional[Path] = field(default=None, repr=False)
    _metadata_store: Optional["IndexMetadataStore"] = field(default=None, repr=False)
    
    def set_codebase(self, path: Path) -> None:
        """
        Set the current working codebase.
        
        Args:
            path: The path to set as the current codebase.
        """
        self._current_codebase = path
    
    def get_codebase(self) -> Path:
        """
        Get the current working codebase.
        
        Returns the explicitly set codebase if one has been set,
        otherwise returns the current working directory.
        
        Returns:
            The current codebase path.
        """
        if self._current_codebase is not None:
            return self._current_codebase
        return Path.cwd()
    
    def clear_codebase(self) -> None:
        """
        Clear the current codebase selection.
        
        After clearing, get_codebase() will return the current working directory.
        """
        self._current_codebase = None
    
    def has_explicit_codebase(self) -> bool:
        """
        Check if a codebase has been explicitly set.
        
        Returns:
            True if set_codebase() has been called and not cleared,
            False otherwise.
        """
        return self._current_codebase is not None

    def set_metadata_store(self, store: "IndexMetadataStore") -> None:
        """
        Set the metadata store for checking indexed paths.
        
        Args:
            store: The metadata store instance.
        """
        self._metadata_store = store

    def is_path_indexed(self, path: Path) -> bool:
        """
        Check if a path has been indexed.
        
        Args:
            path: The path to check.
            
        Returns:
            True if the path is indexed, False otherwise.
        """
        if self._metadata_store is None:
            return False
        
        resolved = str(path.resolve())
        info = self._metadata_store.get_index_info(resolved)
        return info is not None
