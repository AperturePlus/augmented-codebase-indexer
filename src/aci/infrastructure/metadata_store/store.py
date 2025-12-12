"""
Index Metadata Store implementation.

SQLite-based storage for index metadata, file tracking, and statistics.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from .models import IndexedFileInfo, MetadataStoreError, PendingBatch
from .queries import MetadataQueryExecutor, _now_with_tz
from .schema import initialize_schema, migrate_schema

logger = logging.getLogger(__name__)


class IndexMetadataStore:
    """
    SQLite-based index metadata storage.

    Tracks indexed files, their hashes, and statistics for
    incremental update support.
    """

    def __init__(self, db_path: Path | str):
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._query: Optional[MetadataQueryExecutor] = None
        self._initialized = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
            self._query = MetadataQueryExecutor(self._conn)
        return self._conn

    def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return
        conn = self._get_connection()
        try:
            initialize_schema(conn)
            migrate_schema(conn)
            self._initialized = True
            logger.info(f"Initialized metadata store: {self._db_path}")
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to initialize schema: {e}") from e

    def _ensure_query(self) -> MetadataQueryExecutor:
        """Ensure query executor is available."""
        self.initialize()
        assert self._query is not None
        return self._query

    # ─────────────────────────────────────────────────────────────────
    # File Operations
    # ─────────────────────────────────────────────────────────────────

    def get_file_info(self, file_path: str) -> Optional[IndexedFileInfo]:
        """Get information about an indexed file."""
        try:
            row = self._ensure_query().get_file(file_path)
            if row is None:
                return None
            return IndexedFileInfo(
                file_path=row["file_path"],
                content_hash=row["content_hash"],
                language=row["language"],
                line_count=row["line_count"],
                chunk_count=row["chunk_count"],
                indexed_at=datetime.fromisoformat(row["indexed_at"]),
                modified_time=row["modified_time"],
            )
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get file info: {e}") from e

    def upsert_file(self, info: IndexedFileInfo) -> None:
        """Insert or update file index information."""
        try:
            self._ensure_query().upsert_file(
                info.file_path, info.content_hash, info.language,
                info.line_count, info.chunk_count,
                info.indexed_at.isoformat(), info.modified_time,
            )
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to upsert file: {e}") from e

    def upsert_files_batch(self, files: List[IndexedFileInfo]) -> None:
        """Batch insert or update file index information."""
        try:
            self._ensure_query().upsert_files_batch([
                (f.file_path, f.content_hash, f.language, f.line_count,
                 f.chunk_count, f.indexed_at.isoformat(), f.modified_time)
                for f in files
            ])
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to batch upsert files: {e}") from e

    def delete_file(self, file_path: str) -> bool:
        """Delete file index information. Returns True if deleted."""
        try:
            return self._ensure_query().delete_file(file_path) > 0
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to delete file: {e}") from e

    def delete_files_batch(self, file_paths: List[str]) -> int:
        """Batch delete file index information. Returns count deleted."""
        try:
            return self._ensure_query().delete_files_batch(file_paths)
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to batch delete files: {e}") from e

    def get_files_older_than(self, days: int) -> List[str]:
        """Get files indexed more than N days ago."""
        try:
            cutoff = _now_with_tz() - timedelta(days=days)
            return self._ensure_query().get_files_older_than(cutoff.isoformat())
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to query old files: {e}") from e

    def get_all_file_hashes(self) -> Dict[str, str]:
        """Get all file paths and their content hashes."""
        try:
            return self._ensure_query().get_all_file_hashes()
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get file hashes: {e}") from e

    def get_all_files(self) -> List[IndexedFileInfo]:
        """Get all indexed files."""
        try:
            return [
                IndexedFileInfo(
                    file_path=row["file_path"],
                    content_hash=row["content_hash"],
                    language=row["language"],
                    line_count=row["line_count"],
                    chunk_count=row["chunk_count"],
                    indexed_at=datetime.fromisoformat(row["indexed_at"]),
                    modified_time=row["modified_time"],
                )
                for row in self._ensure_query().get_all_files()
            ]
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get all files: {e}") from e

    def get_stale_files(self, limit: Optional[int] = None) -> List[tuple[str, float]]:
        """Get files where modified_time exceeds indexed_at (stale files)."""
        try:
            raw_data = self._ensure_query().get_stale_files_raw()
            stale_files = []
            for file_path, modified_time, indexed_at_str in raw_data:
                indexed_at_ts = datetime.fromisoformat(indexed_at_str).timestamp()
                staleness = modified_time - indexed_at_ts
                if staleness > 0:
                    stale_files.append((file_path, staleness))
            stale_files.sort(key=lambda x: x[1], reverse=True)
            return stale_files[:limit] if limit else stale_files
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get stale files: {e}") from e

    # ─────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get index statistics."""
        try:
            query = self._ensure_query()
            row = query.get_aggregate_stats()
            languages = query.get_language_breakdown()
            return {
                "total_files": row["total_files"],
                "total_chunks": row["total_chunks"],
                "total_lines": row["total_lines"],
                "languages": languages,
            }
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get stats: {e}") from e

    # ─────────────────────────────────────────────────────────────────
    # Index Info / Repository Operations
    # ─────────────────────────────────────────────────────────────────

    def set_index_info(
        self, index_id: str, root_path: str, collection_name: Optional[str] = None
    ) -> None:
        """Set or update index metadata."""
        try:
            self._ensure_query().upsert_index_info(index_id, root_path, collection_name)
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to set index info: {e}") from e

    def register_repository(self, root_path: str, collection_name: Optional[str] = None) -> None:
        """Register or update a repository root path."""
        self.set_index_info(index_id=root_path, root_path=root_path, collection_name=collection_name)

    def get_repositories(self) -> List[Dict]:
        """Get all registered repositories."""
        try:
            return [
                {
                    "root_path": row["root_path"],
                    "collection_name": row["collection_name"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in self._ensure_query().get_all_repositories()
            ]
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get repositories: {e}") from e

    def get_index_info(self, index_id: str) -> Optional[Dict]:
        """Get index metadata."""
        try:
            row = self._ensure_query().get_index_info(index_id)
            if row is None:
                return None
            return {
                "index_id": row["index_id"],
                "root_path": row["root_path"],
                "collection_name": row["collection_name"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get index info: {e}") from e

    def get_collection_name(self, root_path: str) -> Optional[str]:
        """Get the collection name for a repository."""
        info = self.get_index_info(root_path)
        return info.get("collection_name") if info else None

    def find_parent_index(self, path: str) -> Optional[Dict]:
        """Find an indexed repository that contains the given path.
        
        If path itself is indexed, returns that. Otherwise walks up the
        directory tree to find a parent that was indexed.
        
        Returns:
            Index info dict if found, None otherwise.
        """
        from pathlib import Path
        
        # First check exact match
        info = self.get_index_info(path)
        if info is not None:
            return info
        
        # Walk up parent directories
        current = Path(path).resolve()
        for parent in current.parents:
            parent_str = str(parent)
            info = self.get_index_info(parent_str)
            if info is not None:
                return info
        
        return None

    # ─────────────────────────────────────────────────────────────────
    # Pending Batch Operations
    # ─────────────────────────────────────────────────────────────────

    def create_pending_batch(
        self, batch_id: str, file_paths: List[str], chunk_ids: List[str]
    ) -> None:
        """Create a pending batch marker before writing to stores."""
        try:
            self._ensure_query().create_pending_batch(
                batch_id, json.dumps(file_paths), json.dumps(chunk_ids)
            )
            logger.debug(f"Created pending batch: {batch_id}")
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to create pending batch: {e}") from e

    def complete_pending_batch(self, batch_id: str) -> bool:
        """Mark a batch as complete. Returns True if found and deleted."""
        try:
            deleted = self._ensure_query().delete_pending_batch(batch_id) > 0
            if deleted:
                logger.debug(f"Completed pending batch: {batch_id}")
            return deleted
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to complete pending batch: {e}") from e

    def get_pending_batches(self) -> List[PendingBatch]:
        """Get all pending batches."""
        try:
            return [
                PendingBatch(
                    batch_id=row["batch_id"],
                    file_paths=json.loads(row["file_paths"]),
                    chunk_ids=json.loads(row["chunk_ids"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in self._ensure_query().get_all_pending_batches()
            ]
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get pending batches: {e}") from e

    def rollback_pending_batch(self, batch_id: str) -> bool:
        """Rollback a pending batch. Returns True if found and rolled back."""
        try:
            query = self._ensure_query()
            row = query.get_pending_batch(batch_id)
            if row is None:
                return False
            file_paths = json.loads(row["file_paths"])
            query.delete_files_in_list(file_paths)
            query.delete_pending_batch(batch_id)
            logger.info(f"Rolled back pending batch {batch_id}: removed {len(file_paths)} file entries")
            return True
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to rollback pending batch: {e}") from e

    # ─────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────

    def clear_all(self) -> None:
        """Clear all data from the store."""
        try:
            self._ensure_query().clear_all()
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to clear data: {e}") from e

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._query = None
            self._initialized = False


def create_metadata_store(db_path: Path | str) -> IndexMetadataStore:
    """Factory function to create a metadata store."""
    return IndexMetadataStore(db_path)
