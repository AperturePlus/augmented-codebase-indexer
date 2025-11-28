"""
Index Metadata Store for Project ACI.

SQLite-based storage for index metadata, file tracking, and statistics.
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MetadataStoreError(Exception):
    """Base exception for metadata store errors."""

    pass


@dataclass
class IndexedFileInfo:
    """Information about an indexed file."""

    file_path: str
    content_hash: str
    language: str
    line_count: int
    chunk_count: int
    indexed_at: datetime
    modified_time: float


class IndexMetadataStore:
    """
    SQLite-based index metadata storage.

    Tracks indexed files, their hashes, and statistics for
    incremental update support.
    """

    # SQL schema for the database
    _SCHEMA = """
    -- Index metadata
    CREATE TABLE IF NOT EXISTS index_info (
        index_id TEXT PRIMARY KEY,
        root_path TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Indexed files tracking
    CREATE TABLE IF NOT EXISTS indexed_files (
        file_path TEXT PRIMARY KEY,
        content_hash TEXT NOT NULL,
        language TEXT NOT NULL,
        line_count INTEGER NOT NULL,
        chunk_count INTEGER NOT NULL,
        indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        modified_time REAL NOT NULL
    );

    -- Statistics cache
    CREATE TABLE IF NOT EXISTS index_stats (
        stat_key TEXT PRIMARY KEY,
        stat_value TEXT NOT NULL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Indexes for efficient queries
    CREATE INDEX IF NOT EXISTS idx_files_indexed_at 
        ON indexed_files(indexed_at);
    CREATE INDEX IF NOT EXISTS idx_files_modified 
        ON indexed_files(modified_time);
    CREATE INDEX IF NOT EXISTS idx_files_language 
        ON indexed_files(language);
    """

    def __init__(self, db_path: Path | str):
        """
        Initialize the metadata store.

        Args:
            db_path: Path to the SQLite database file
        """
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._initialized = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            # Ensure parent directory exists
            self._db_path.parent.mkdir(parents=True, exist_ok=True)

            # Don't use PARSE_DECLTYPES to avoid timestamp conversion issues
            # We handle datetime conversion manually
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row

        return self._conn

    def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return

        conn = self._get_connection()
        try:
            conn.executescript(self._SCHEMA)
            conn.commit()
            self._initialized = True
            logger.info(f"Initialized metadata store: {self._db_path}")
        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to initialize schema: {e}") from e

    def get_file_info(self, file_path: str) -> Optional[IndexedFileInfo]:
        """Get information about an indexed file."""
        self.initialize()
        conn = self._get_connection()

        try:
            cursor = conn.execute(
                """
                SELECT file_path, content_hash, language, line_count, 
                       chunk_count, indexed_at, modified_time
                FROM indexed_files
                WHERE file_path = ?
                """,
                (file_path,),
            )
            row = cursor.fetchone()

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
        self.initialize()
        conn = self._get_connection()

        try:
            conn.execute(
                """
                INSERT INTO indexed_files 
                    (file_path, content_hash, language, line_count, 
                     chunk_count, indexed_at, modified_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    language = excluded.language,
                    line_count = excluded.line_count,
                    chunk_count = excluded.chunk_count,
                    indexed_at = excluded.indexed_at,
                    modified_time = excluded.modified_time
                """,
                (
                    info.file_path,
                    info.content_hash,
                    info.language,
                    info.line_count,
                    info.chunk_count,
                    info.indexed_at.isoformat(),
                    info.modified_time,
                ),
            )
            conn.commit()

        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to upsert file: {e}") from e

    def upsert_files_batch(self, files: List[IndexedFileInfo]) -> None:
        """Batch insert or update file index information."""
        self.initialize()
        conn = self._get_connection()

        try:
            conn.executemany(
                """
                INSERT INTO indexed_files 
                    (file_path, content_hash, language, line_count, 
                     chunk_count, indexed_at, modified_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    language = excluded.language,
                    line_count = excluded.line_count,
                    chunk_count = excluded.chunk_count,
                    indexed_at = excluded.indexed_at,
                    modified_time = excluded.modified_time
                """,
                [
                    (
                        f.file_path,
                        f.content_hash,
                        f.language,
                        f.line_count,
                        f.chunk_count,
                        f.indexed_at.isoformat(),
                        f.modified_time,
                    )
                    for f in files
                ],
            )
            conn.commit()

        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to batch upsert files: {e}") from e

    def delete_file(self, file_path: str) -> bool:
        """
        Delete file index information.

        Returns:
            True if file was deleted, False if not found
        """
        self.initialize()
        conn = self._get_connection()

        try:
            cursor = conn.execute(
                "DELETE FROM indexed_files WHERE file_path = ?",
                (file_path,),
            )
            conn.commit()
            return cursor.rowcount > 0

        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to delete file: {e}") from e

    def delete_files_batch(self, file_paths: List[str]) -> int:
        """
        Batch delete file index information.

        Returns:
            Number of files deleted
        """
        self.initialize()
        conn = self._get_connection()

        try:
            cursor = conn.executemany(
                "DELETE FROM indexed_files WHERE file_path = ?",
                [(fp,) for fp in file_paths],
            )
            conn.commit()
            return cursor.rowcount

        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to batch delete files: {e}") from e

    def get_files_older_than(self, days: int) -> List[str]:
        """
        Get files indexed more than N days ago.

        Args:
            days: Number of days threshold

        Returns:
            List of file paths
        """
        self.initialize()
        conn = self._get_connection()

        try:
            cutoff = datetime.now() - timedelta(days=days)
            cursor = conn.execute(
                """
                SELECT file_path FROM indexed_files
                WHERE indexed_at < ?
                """,
                (cutoff.isoformat(),),
            )
            return [row["file_path"] for row in cursor.fetchall()]

        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to query old files: {e}") from e

    def get_all_file_hashes(self) -> Dict[str, str]:
        """
        Get all file paths and their content hashes.

        Returns:
            Dict mapping file_path to content_hash
        """
        self.initialize()
        conn = self._get_connection()

        try:
            cursor = conn.execute("SELECT file_path, content_hash FROM indexed_files")
            return {row["file_path"]: row["content_hash"] for row in cursor.fetchall()}

        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get file hashes: {e}") from e

    def get_all_files(self) -> List[IndexedFileInfo]:
        """Get all indexed files."""
        self.initialize()
        conn = self._get_connection()

        try:
            cursor = conn.execute(
                """
                SELECT file_path, content_hash, language, line_count, 
                       chunk_count, indexed_at, modified_time
                FROM indexed_files
                """
            )
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
                for row in cursor.fetchall()
            ]

        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get all files: {e}") from e

    def get_stats(self) -> Dict:
        """
        Get index statistics.

        Returns:
            Dict with total_files, total_chunks, total_lines, languages
        """
        self.initialize()
        conn = self._get_connection()

        try:
            # Get aggregate stats
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total_files,
                    COALESCE(SUM(chunk_count), 0) as total_chunks,
                    COALESCE(SUM(line_count), 0) as total_lines
                FROM indexed_files
                """
            )
            row = cursor.fetchone()

            # Get language breakdown
            cursor = conn.execute(
                """
                SELECT language, COUNT(*) as count
                FROM indexed_files
                GROUP BY language
                """
            )
            languages = {r["language"]: r["count"] for r in cursor.fetchall()}

            return {
                "total_files": row["total_files"],
                "total_chunks": row["total_chunks"],
                "total_lines": row["total_lines"],
                "languages": languages,
            }

        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get stats: {e}") from e

    def set_index_info(self, index_id: str, root_path: str) -> None:
        """Set or update index metadata."""
        self.initialize()
        conn = self._get_connection()

        try:
            conn.execute(
                """
                INSERT INTO index_info (index_id, root_path, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(index_id) DO UPDATE SET
                    root_path = excluded.root_path,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (index_id, root_path),
            )
            conn.commit()

        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to set index info: {e}") from e

    def register_repository(self, root_path: str) -> None:
        """
        Register or update a repository root path.
        
        Uses the root_path itself as the unique index_id.
        """
        self.set_index_info(index_id=root_path, root_path=root_path)

    def get_repositories(self) -> List[Dict]:
        """
        Get all registered repositories.
        
        Returns:
            List of dicts with keys: root_path, created_at, updated_at
        """
        self.initialize()
        conn = self._get_connection()

        try:
            cursor = conn.execute(
                """
                SELECT root_path, created_at, updated_at
                FROM index_info
                ORDER BY updated_at DESC
                """
            )
            return [
                {
                    "root_path": row["root_path"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in cursor.fetchall()
            ]

        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get repositories: {e}") from e

    def get_index_info(self, index_id: str) -> Optional[Dict]:
        """Get index metadata."""
        self.initialize()
        conn = self._get_connection()

        try:
            cursor = conn.execute(
                """
                SELECT index_id, root_path, created_at, updated_at
                FROM index_info
                WHERE index_id = ?
                """,
                (index_id,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return {
                "index_id": row["index_id"],
                "root_path": row["root_path"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to get index info: {e}") from e

    def clear_all(self) -> None:
        """Clear all data from the store."""
        self.initialize()
        conn = self._get_connection()

        try:
            conn.execute("DELETE FROM indexed_files")
            conn.execute("DELETE FROM index_info")
            conn.execute("DELETE FROM index_stats")
            conn.commit()

        except sqlite3.Error as e:
            raise MetadataStoreError(f"Failed to clear data: {e}") from e

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            self._initialized = False


def create_metadata_store(db_path: Path | str) -> IndexMetadataStore:
    """
    Factory function to create a metadata store.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        Configured IndexMetadataStore instance
    """
    return IndexMetadataStore(db_path)
