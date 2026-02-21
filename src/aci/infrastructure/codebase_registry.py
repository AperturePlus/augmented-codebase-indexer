"""
Global codebase registry for ACI.

This module provides a small SQLite registry stored in the user's home
directory (by default `~/.aci/registry.db`) to track which codebases have
been indexed and where their local `.aci` state lives.

The registry is intentionally lightweight and best-effort: failures to
read/write the registry should not break core indexing/search behavior.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _now_str() -> str:
    """Return current local time as an ISO-like string compatible with SQLite."""
    try:
        return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def is_global_registry_enabled() -> bool:
    """
    Check whether the global registry is enabled.

    Disabled when:
    - ACI_DISABLE_GLOBAL_REGISTRY is truthy
    - Running under pytest (PYTEST_CURRENT_TEST is set) unless explicitly enabled
      via ACI_ENABLE_GLOBAL_REGISTRY.
    """
    if _is_truthy(os.environ.get("ACI_DISABLE_GLOBAL_REGISTRY")):
        return False

    if os.environ.get("PYTEST_CURRENT_TEST") and not _is_truthy(
        os.environ.get("ACI_ENABLE_GLOBAL_REGISTRY")
    ):
        return False

    return True


def get_default_registry_db_path() -> Path:
    """
    Get the default registry database path.

    Override the location with:
    - ACI_REGISTRY_PATH: full path to registry.db
    - ACI_GLOBAL_ACI_DIR: directory for global ACI data (registry.db will be placed inside)
    """
    override = os.environ.get("ACI_REGISTRY_PATH")
    if override:
        return Path(override).expanduser()

    global_dir = os.environ.get("ACI_GLOBAL_ACI_DIR")
    if global_dir:
        return Path(global_dir).expanduser() / "registry.db"

    return Path.home() / ".aci" / "registry.db"


def _normalize_path(path: Path | str) -> str:
    """
    Normalize a path for stable storage and prefix matching.

    Uses absolute resolved path with forward slashes. Falls back to string
    representation if resolution fails (e.g., special SQLite ':memory:').
    """
    try:
        p = Path(path) if isinstance(path, str) else path
        if str(p) == ":memory:":
            return ":memory:"
        return p.resolve().as_posix()
    except Exception:
        return str(path)


@dataclass(frozen=True)
class CodebaseRecord:
    root_path: str
    metadata_db_path: str
    collection_name: str
    created_at: str
    updated_at: str
    last_indexed_at: str | None = None


class CodebaseRegistryError(RuntimeError):
    pass


class CodebaseRegistryStore:
    """
    SQLite-backed registry of indexed codebases.

    This registry is independent from per-project `.aci/index.db` metadata.
    It exists to provide a global "catalog" of indexed repositories.
    """

    def __init__(self, db_path: Path | str | None = None):
        self._db_path = Path(db_path) if db_path is not None else get_default_registry_db_path()
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            try:
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                raise CodebaseRegistryError(f"Failed to create registry directory: {self._db_path.parent}") from None

            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            try:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA busy_timeout=5000;")
            except sqlite3.Error:
                # Best-effort; continue without custom pragmas.
                pass
        return self._conn

    def initialize(self) -> None:
        if self._initialized:
            return

        conn = self._get_connection()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS codebases (
                    root_path TEXT PRIMARY KEY,
                    metadata_db_path TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    last_indexed_at TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_codebases_updated_at ON codebases(updated_at);
                """
            )
            conn.commit()
            self._initialized = True
        except sqlite3.Error as e:
            raise CodebaseRegistryError(f"Failed to initialize registry schema: {e}") from e

    def upsert_codebase(
        self,
        root_path: Path | str,
        *,
        metadata_db_path: Path | str,
        collection_name: str,
        last_indexed_at: str | None = None,
    ) -> None:
        self.initialize()

        root_norm = _normalize_path(root_path)
        db_norm = _normalize_path(metadata_db_path)
        now = _now_str()
        try:
            self._get_connection().execute(
                """
                INSERT INTO codebases (
                    root_path, metadata_db_path, collection_name, created_at, updated_at, last_indexed_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(root_path) DO UPDATE SET
                    metadata_db_path = excluded.metadata_db_path,
                    collection_name = excluded.collection_name,
                    updated_at = ?,
                    last_indexed_at = excluded.last_indexed_at
                """,
                (root_norm, db_norm, collection_name, now, now, last_indexed_at, now),
            )
            self._get_connection().commit()
        except sqlite3.Error as e:
            raise CodebaseRegistryError(f"Failed to upsert codebase: {e}") from e

    def delete_codebase(self, root_path: Path | str) -> bool:
        self.initialize()
        root_norm = _normalize_path(root_path)
        try:
            cur = self._get_connection().execute(
                "DELETE FROM codebases WHERE root_path = ?",
                (root_norm,),
            )
            self._get_connection().commit()
            return cur.rowcount > 0
        except sqlite3.Error as e:
            raise CodebaseRegistryError(f"Failed to delete codebase: {e}") from e

    def list_codebases(self) -> list[CodebaseRecord]:
        self.initialize()
        try:
            rows = self._get_connection().execute(
                """
                SELECT root_path, metadata_db_path, collection_name, created_at, updated_at, last_indexed_at
                FROM codebases
                ORDER BY updated_at DESC
                """
            ).fetchall()
            return [
                CodebaseRecord(
                    root_path=row["root_path"],
                    metadata_db_path=row["metadata_db_path"],
                    collection_name=row["collection_name"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    last_indexed_at=row["last_indexed_at"],
                )
                for row in rows
            ]
        except sqlite3.Error as e:
            raise CodebaseRegistryError(f"Failed to list codebases: {e}") from e

    def find_codebase_for_path(self, path: Path | str) -> CodebaseRecord | None:
        """
        Find the best matching indexed codebase for a given path.

        Returns the record whose root_path is the longest ancestor prefix of the
        provided path.
        """
        path_norm = _normalize_path(path)
        candidates = []
        for rec in self.list_codebases():
            root = rec.root_path
            if path_norm == root or path_norm.startswith(root.rstrip("/") + "/"):
                candidates.append(rec)
        if not candidates:
            return None
        return max(candidates, key=lambda r: len(r.root_path))

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
                self._initialized = False


def best_effort_update_registry(
    *,
    root_path: Path | str,
    metadata_db_path: Path | str,
    collection_name: str,
) -> None:
    """
    Best-effort helper to update the global registry.

    Any error is swallowed to avoid breaking main flows.
    """
    if not is_global_registry_enabled():
        return

    store = CodebaseRegistryStore()
    try:
        store.upsert_codebase(
            root_path,
            metadata_db_path=metadata_db_path,
            collection_name=collection_name,
            last_indexed_at=_now_str(),
        )
    except Exception as exc:
        logger.debug("Global registry update failed: %s", exc)
    finally:
        store.close()


def best_effort_remove_from_registry(*, root_path: Path | str) -> None:
    """
    Best-effort helper to remove a codebase from the global registry.
    """
    if not is_global_registry_enabled():
        return

    store = CodebaseRegistryStore()
    try:
        store.delete_codebase(root_path)
    except Exception as exc:
        logger.debug("Global registry delete failed: %s", exc)
    finally:
        store.close()

