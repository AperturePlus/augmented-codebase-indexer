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
from aci.core.ast_parser import (
    ASTNode,
    ASTParserInterface,
    TreeSitterParser,
    SUPPORTED_LANGUAGES,
    check_tree_sitter_setup,
)
from aci.core.tokenizer import (
    TokenizerInterface,
    TiktokenTokenizer,
    get_default_tokenizer,
)
from aci.core.chunker import (
    CodeChunk,
    ChunkerInterface,
    Chunker,
    create_chunker,
    ImportExtractorInterface,
    ImportExtractorRegistry,
    PythonImportExtractor,
    JavaScriptImportExtractor,
    GoImportExtractor,
    get_import_registry,
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
    # AST Parser
    "ASTNode",
    "ASTParserInterface",
    "TreeSitterParser",
    "SUPPORTED_LANGUAGES",
    "check_tree_sitter_setup",
    # Tokenizer
    "TokenizerInterface",
    "TiktokenTokenizer",
    "get_default_tokenizer",
    # Chunker
    "CodeChunk",
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
]
