"""
Core Layer - File scanning, AST parsing, chunking, and tokenization components.
"""

from aci.core.config import (
    ACIConfig,
    EmbeddingConfig,
    VectorStoreConfig,
    IndexingConfig,
    SearchConfig,
    LoggingConfig,
    load_config,
)
from aci.core.file_scanner import (
    ScannedFile,
    FileScannerInterface,
    FileScanner,
    LanguageRegistry,
    get_default_registry,
    EXTENSION_TO_LANGUAGE,
)

__all__ = [
    # Config
    "ACIConfig",
    "EmbeddingConfig",
    "VectorStoreConfig",
    "IndexingConfig",
    "SearchConfig",
    "LoggingConfig",
    "load_config",
    # FileScanner
    "ScannedFile",
    "FileScannerInterface",
    "FileScanner",
    "LanguageRegistry",
    "get_default_registry",
    "EXTENSION_TO_LANGUAGE",
]
