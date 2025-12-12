"""
Chunker module for Project ACI.

Provides code chunking capabilities with AST-based semantic chunking
and fixed-size fallback for unsupported languages.
"""

from .chunker import Chunker, create_chunker
from .import_extractors import (
    GoImportExtractor,
    ImportExtractorRegistry,
    JavaScriptImportExtractor,
    NullImportExtractor,
    PythonImportExtractor,
    get_import_registry,
)
from .interfaces import ChunkerInterface, ImportExtractorInterface
from .models import ChunkingResult, CodeChunk
from .smart_splitter import SmartChunkSplitter

__all__ = [
    # Main classes
    "Chunker",
    "ChunkerInterface",
    "CodeChunk",
    "ChunkingResult",
    # Smart splitter
    "SmartChunkSplitter",
    # Import extractors
    "ImportExtractorInterface",
    "ImportExtractorRegistry",
    "PythonImportExtractor",
    "JavaScriptImportExtractor",
    "GoImportExtractor",
    "NullImportExtractor",
    "get_import_registry",
    # Factory
    "create_chunker",
]
