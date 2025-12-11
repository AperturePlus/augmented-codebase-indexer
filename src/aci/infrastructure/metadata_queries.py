"""
Low-level SQL query executor for metadata store.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


def _now_with_tz() -> datetime:
    """Get current datetime with local timezone."""
    try:
        return datetime.now().astimezone()
    except Exception:
        from datetime import timezone as tz
        return datetime.now(tz(timedelta(hours=8)))


def _now_str() -> str:
    """Get current datetime as ISO string for SQLite."""
    return _now_with_tz().strftime("%Y-%m-%d %H:%M:%S")


class MetadataQueryExecutor:
    """Executes SQL queries for metadata store."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    # ─────────────────────────────────────────────────────────────────
    # Indexed Files Operations
    # ─────────────────────────────────────────────────────────────────

    def get_file(self, file_path: str) -> Optional[sqlite3.Row]:
        """Get a single indexed file by path."""
        cursor = self._conn.execute(
            """
            SELECT file_path, content_hash, language, line_count, 
                   chunk_count, indexed_at, modified_time
            FROM indexed_files WHERE file_path = ?
            """,
            (file_path,),
        )
        return cursor.fetchone()

    def upsert_file(
        self,
        file_path: str,
        content_hash: str,
        language: str,
        line_count: int,
        chunk_count: int,
        indexed_at: str,
        modified_time: float,
    ) -> None:
        """Insert or update a single file."""
        self._conn.execute(
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
            (file_path, content_hash, language, line_count, chunk_count, indexed_at, modified_time),
        )
        self._conn.commit()

    def upsert_files_batch(self, files: List[Tuple[str, str, str, int, int, str, float]]) -> None:
        """Batch insert or update files."""
        self._conn.executemany(
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
            files,
        )
        self._conn.commit()

    def delete_file(self, file_path: str) -> int:
        """Delete a file, returns rowcount."""
        cursor = self._conn.execute(
            "DELETE FROM indexed_files WHERE file_path = ?", (file_path,)
        )
        self._conn.commit()
        return cursor.rowcount

    def delete_files_batch(self, file_paths: List[str]) -> int:
        """Batch delete files, returns rowcount."""
        cursor = self._conn.executemany(
            "DELETE FROM indexed_files WHERE file_path = ?",
            [(fp,) for fp in file_paths],
        )
        self._conn.commit()
        return cursor.rowcount

    def get_all_files(self) -> List[sqlite3.Row]:
        """Get all indexed files."""
        cursor = self._conn.execute(
            """
            SELECT file_path, content_hash, language, line_count, 
                   chunk_count, indexed_at, modified_time
            FROM indexed_files
            """
        )
        return cursor.fetchall()

    def get_all_file_hashes(self) -> Dict[str, str]:
        """Get all file paths and their content hashes."""
        cursor = self._conn.execute("SELECT file_path, content_hash FROM indexed_files")
        return {row["file_path"]: row["content_hash"] for row in cursor.fetchall()}

    def get_files_older_than(self, cutoff_iso: str) -> List[str]:
        """Get file paths indexed before cutoff date."""
        cursor = self._conn.execute(
            "SELECT file_path FROM indexed_files WHERE indexed_at < ?",
            (cutoff_iso,),
        )
        return [row["file_path"] for row in cursor.fetchall()]

    def get_stale_files_raw(self) -> List[Tuple[str, float, str]]:
        """Get raw data for stale file calculation."""
        cursor = self._conn.execute(
            "SELECT file_path, modified_time, indexed_at FROM indexed_files"
        )
        return [(row["file_path"], row["modified_time"], row["indexed_at"]) for row in cursor.fetchall()]

    # ─────────────────────────────────────────────────────────────────
    # Statistics Operations
    # ─────────────────────────────────────────────────────────────────

    def get_aggregate_stats(self) -> sqlite3.Row:
        """Get aggregate statistics."""
        cursor = self._conn.execute(
            """
            SELECT 
                COUNT(*) as total_files,
                COALESCE(SUM(chunk_count), 0) as total_chunks,
                COALESCE(SUM(line_count), 0) as total_lines
            FROM indexed_files
            """
        )
        return cursor.fetchone()

    def get_language_breakdown(self) -> Dict[str, int]:
        """Get file count by language."""
        cursor = self._conn.execute(
            "SELECT language, COUNT(*) as count FROM indexed_files GROUP BY language"
        )
        return {r["language"]: r["count"] for r in cursor.fetchall()}

    # ─────────────────────────────────────────────────────────────────
    # Index Info Operations
    # ─────────────────────────────────────────────────────────────────

    def upsert_index_info(
        self, index_id: str, root_path: str, collection_name: Optional[str]
    ) -> None:
        """Insert or update index info."""
        now = _now_str()
        self._conn.execute(
            """
            INSERT INTO index_info (index_id, root_path, collection_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(index_id) DO UPDATE SET
                root_path = excluded.root_path,
                collection_name = COALESCE(excluded.collection_name, collection_name),
                updated_at = ?
            """,
            (index_id, root_path, collection_name, now, now, now),
        )
        self._conn.commit()

    def get_index_info(self, index_id: str) -> Optional[sqlite3.Row]:
        """Get index info by ID."""
        cursor = self._conn.execute(
            """
            SELECT index_id, root_path, collection_name, created_at, updated_at
            FROM index_info WHERE index_id = ?
            """,
            (index_id,),
        )
        return cursor.fetchone()

    def get_all_repositories(self) -> List[sqlite3.Row]:
        """Get all registered repositories."""
        cursor = self._conn.execute(
            """
            SELECT root_path, collection_name, created_at, updated_at
            FROM index_info ORDER BY updated_at DESC
            """
        )
        return cursor.fetchall()

    # ─────────────────────────────────────────────────────────────────
    # Pending Batches Operations
    # ─────────────────────────────────────────────────────────────────

    def create_pending_batch(
        self, batch_id: str, file_paths_json: str, chunk_ids_json: str
    ) -> None:
        """Create a pending batch marker."""
        self._conn.execute(
            """
            INSERT INTO pending_batches (batch_id, file_paths, chunk_ids, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (batch_id, file_paths_json, chunk_ids_json, _now_str()),
        )
        self._conn.commit()

    def delete_pending_batch(self, batch_id: str) -> int:
        """Delete a pending batch, returns rowcount."""
        cursor = self._conn.execute(
            "DELETE FROM pending_batches WHERE batch_id = ?", (batch_id,)
        )
        self._conn.commit()
        return cursor.rowcount

    def get_pending_batch(self, batch_id: str) -> Optional[sqlite3.Row]:
        """Get a pending batch by ID."""
        cursor = self._conn.execute(
            "SELECT file_paths FROM pending_batches WHERE batch_id = ?", (batch_id,)
        )
        return cursor.fetchone()

    def get_all_pending_batches(self) -> List[sqlite3.Row]:
        """Get all pending batches."""
        cursor = self._conn.execute(
            """
            SELECT batch_id, file_paths, chunk_ids, created_at
            FROM pending_batches ORDER BY created_at ASC
            """
        )
        return cursor.fetchall()

    def delete_files_in_list(self, file_paths: List[str]) -> None:
        """Delete files by path list."""
        if file_paths:
            placeholders = ",".join("?" * len(file_paths))
            self._conn.execute(
                f"DELETE FROM indexed_files WHERE file_path IN ({placeholders})",
                file_paths,
            )
            self._conn.commit()

    # ─────────────────────────────────────────────────────────────────
    # Clear Operations
    # ─────────────────────────────────────────────────────────────────

    def clear_all(self) -> None:
        """Clear all data from all tables."""
        self._conn.execute("DELETE FROM indexed_files")
        self._conn.execute("DELETE FROM index_info")
        self._conn.execute("DELETE FROM index_stats")
        self._conn.execute("DELETE FROM pending_batches")
        self._conn.commit()
