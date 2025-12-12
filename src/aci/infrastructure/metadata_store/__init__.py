"""
Metadata Store module for Project ACI.

SQLite-based storage for index metadata, file tracking, and statistics.
"""

from .models import IndexedFileInfo, MetadataStoreError, PendingBatch
from .queries import MetadataQueryExecutor, _now_with_tz
from .schema import initialize_schema, migrate_schema
from .store import IndexMetadataStore, create_metadata_store

__all__ = [
    # Main classes
    "IndexMetadataStore",
    "IndexedFileInfo",
    "PendingBatch",
    "MetadataStoreError",
    # Query executor
    "MetadataQueryExecutor",
    # Schema
    "initialize_schema",
    "migrate_schema",
    # Factory
    "create_metadata_store",
    # Utilities
    "_now_with_tz",
]
