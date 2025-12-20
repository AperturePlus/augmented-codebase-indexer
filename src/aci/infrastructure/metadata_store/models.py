"""
Data models for metadata store.
"""

from dataclasses import dataclass
from datetime import datetime


class MetadataStoreError(Exception):
    """Base exception for metadata store errors."""
    pass


@dataclass
class PendingBatch:
    """Information about a pending batch operation."""
    batch_id: str
    file_paths: list[str]
    chunk_ids: list[str]
    created_at: datetime


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
