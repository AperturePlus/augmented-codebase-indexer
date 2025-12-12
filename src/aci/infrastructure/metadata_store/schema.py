"""
Metadata store schema definitions and migrations.
"""

import logging
import sqlite3

logger = logging.getLogger(__name__)

SCHEMA = """
-- Index metadata
CREATE TABLE IF NOT EXISTS index_info (
    index_id TEXT PRIMARY KEY,
    root_path TEXT NOT NULL,
    collection_name TEXT,
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

-- Pending batches for transactional consistency
CREATE TABLE IF NOT EXISTS pending_batches (
    batch_id TEXT PRIMARY KEY,
    file_paths TEXT NOT NULL,
    chunk_ids TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_files_indexed_at 
    ON indexed_files(indexed_at);
CREATE INDEX IF NOT EXISTS idx_files_modified 
    ON indexed_files(modified_time);
CREATE INDEX IF NOT EXISTS idx_files_language 
    ON indexed_files(language);
"""


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Initialize the database schema."""
    conn.executescript(SCHEMA)
    conn.commit()


def migrate_schema(conn: sqlite3.Connection) -> None:
    """Run schema migrations for existing databases."""
    try:
        cursor = conn.execute("PRAGMA table_info(index_info)")
        columns = {row[1] for row in cursor.fetchall()}

        if "collection_name" not in columns:
            logger.info("Migrating database: adding collection_name column to index_info")
            conn.execute("ALTER TABLE index_info ADD COLUMN collection_name TEXT")
            conn.commit()

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='pending_batches'"
        )
        if cursor.fetchone() is None:
            logger.info("Migrating database: creating pending_batches table")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_batches (
                    batch_id TEXT PRIMARY KEY,
                    file_paths TEXT NOT NULL,
                    chunk_ids TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    except sqlite3.Error as e:
        logger.warning(f"Schema migration warning: {e}")
