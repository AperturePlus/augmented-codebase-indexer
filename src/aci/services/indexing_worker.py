"""
Indexing worker functions for parallel processing.

Contains worker initialization and file processing functions
that run in separate processes via ProcessPoolExecutor.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from aci.core.ast_parser import ASTParserInterface, TreeSitterParser
from aci.core.chunker import ChunkerInterface, create_chunker
from aci.core.file_scanner import ScannedFile

logger = logging.getLogger(__name__)


@dataclass
class ChunkerConfig:
    """Configuration for chunker in worker processes."""
    max_tokens: int = 8192
    fixed_chunk_lines: int = 50
    overlap_lines: int = 5


# Global variables for worker processes
_worker_parser: Optional[ASTParserInterface] = None
_worker_chunker: Optional[ChunkerInterface] = None
_worker_config: Optional[ChunkerConfig] = None


def init_worker(config: Optional[ChunkerConfig] = None) -> None:
    """
    Initialize worker process with parser and chunker.
    
    This runs once per worker process to avoid repeatedly creating
    TreeSitterParser instances (which load shared libraries) and
    Chunker instances.
    
    Args:
        config: Optional chunker configuration. If None, uses defaults.
    """
    global _worker_parser, _worker_chunker, _worker_config
    _worker_config = config or ChunkerConfig()
    _worker_parser = TreeSitterParser()
    _worker_chunker = create_chunker(
        max_tokens=_worker_config.max_tokens,
        fixed_chunk_lines=_worker_config.fixed_chunk_lines,
        overlap_lines=_worker_config.overlap_lines,
    )


def process_file_worker(
    file_path: str,
    content: str,
    language: str,
    content_hash: str,
    modified_time: float,
    size_bytes: int,
) -> Tuple[str, List[dict], str, int, str, Optional[str]]:
    """
    Worker function for parallel file processing.
    
    This function runs in a separate process to handle CPU-intensive
    AST parsing and chunking operations. It uses global parser/chunker
    instances initialized by init_worker.
    
    Args:
        file_path: Path to the file
        content: File content
        language: Programming language
        content_hash: SHA-256 hash of content
        modified_time: File modification timestamp
        size_bytes: File size in bytes
        
    Returns:
        Tuple of (file_path, chunks_data, language, line_count, content_hash, error)
        where chunks_data is a list of serializable chunk dictionaries
    """
    try:
        # Use global instances initialized per worker
        global _worker_parser, _worker_chunker, _worker_config
        
        # Fallback if not initialized (e.g., if run directly not via executor)
        if _worker_parser is None:
            _worker_parser = TreeSitterParser()
        if _worker_chunker is None:
            cfg = _worker_config or ChunkerConfig()
            _worker_chunker = create_chunker(
                max_tokens=cfg.max_tokens,
                fixed_chunk_lines=cfg.fixed_chunk_lines,
                overlap_lines=cfg.overlap_lines,
            )
            
        parser = _worker_parser
        chunker = _worker_chunker
        
        # Create a ScannedFile object
        scanned_file = ScannedFile(
            path=Path(file_path),
            content=content,
            language=language,
            size_bytes=size_bytes,
            modified_time=modified_time,
            content_hash=content_hash,
        )
        
        # Parse AST if language is supported
        ast_nodes = []
        if parser.supports_language(language):
            ast_nodes = parser.parse(content, language)
        
        # Chunk the file
        chunks = chunker.chunk(scanned_file, ast_nodes)
        
        # Convert chunks to serializable dictionaries
        chunks_data = [
            {
                "chunk_id": chunk.chunk_id,
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content": chunk.content,
                "language": chunk.language,
                "chunk_type": chunk.chunk_type,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        
        line_count = content.count('\n') + 1
        
        return (file_path, chunks_data, language, line_count, content_hash, None)
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return (file_path, [], language, 0, content_hash, str(e))
