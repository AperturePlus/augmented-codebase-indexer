"""
Core Layer - File scanning, AST parsing, chunking, and tokenization components.
"""

from aci.core.ast_parser import (
    SUPPORTED_LANGUAGES,
    ASTNode,
    ASTParserInterface,
    TreeSitterParser,
    check_tree_sitter_setup,
)
from aci.core.chunker import (
    Chunker,
    ChunkerConfig,
    ChunkerInterface,
    CodeChunk,
    GoImportExtractor,
    ImportExtractorInterface,
    ImportExtractorRegistry,
    JavaScriptImportExtractor,
    PythonImportExtractor,
    create_chunker,
    get_import_registry,
)
from aci.core.config import (
    ACIConfig,
    EmbeddingConfig,
    IndexingConfig,
    LoggingConfig,
    SearchConfig,
    VectorStoreConfig,
    load_config,
)
from aci.core.debouncer import Debouncer
from aci.core.file_events import DebouncedBatch, FileEvent, FileEventType
from aci.core.file_scanner import (
    EXTENSION_TO_LANGUAGE,
    FileScanner,
    FileScannerInterface,
    LanguageRegistry,
    ScannedFile,
    get_default_registry,
)
from aci.core.tokenizer import (
    CharacterTokenizer,
    SimpleTokenizer,
    TiktokenTokenizer,
    TokenizerInterface,
    get_default_tokenizer,
)
from aci.core.watch_config import WatchConfig

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
    # AST Parser
    "ASTNode",
    "ASTParserInterface",
    "TreeSitterParser",
    "SUPPORTED_LANGUAGES",
    "check_tree_sitter_setup",
    # Tokenizer
    "TokenizerInterface",
    "TiktokenTokenizer",
    "CharacterTokenizer",
    "SimpleTokenizer",
    "get_default_tokenizer",
    # Chunker
    "CodeChunk",
    "ChunkerConfig",
    "ChunkerInterface",
    "Chunker",
    "create_chunker",
    # Import Extractors
    "ImportExtractorInterface",
    "ImportExtractorRegistry",
    "PythonImportExtractor",
    "JavaScriptImportExtractor",
    "GoImportExtractor",
    "get_import_registry",
    # Watch Config
    "WatchConfig",
    # File Events
    "FileEventType",
    "FileEvent",
    "DebouncedBatch",
    # Debouncer
    "Debouncer",
]
