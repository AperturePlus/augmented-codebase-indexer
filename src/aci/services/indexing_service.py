"""
Indexing Service for Project ACI.

Coordinates the indexing workflow: file scanning, parsing, chunking,
embedding generation, and vector storage.

Supports parallel file processing using ProcessPoolExecutor for CPU-intensive
operations (AST parsing, chunking) and async for IO-intensive operations
(embedding API calls, vector storage).
"""

import asyncio
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from aci.core.ast_parser import ASTParserInterface, TreeSitterParser
from aci.core.chunker import ChunkerInterface, CodeChunk, Chunker, create_chunker
from aci.core.file_scanner import FileScanner, FileScannerInterface, ScannedFile
from aci.infrastructure.embedding_client import EmbeddingClientInterface
from aci.infrastructure.metadata_store import IndexedFileInfo, IndexMetadataStore
from aci.infrastructure.vector_store import VectorStoreInterface

logger = logging.getLogger(__name__)


def _process_file_worker(
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
    AST parsing and chunking operations.
    
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
        # Create parser and chunker in the worker process
        parser = TreeSitterParser()
        chunker = create_chunker()
        
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


@dataclass
class IndexingResult:
    """Result of an indexing operation."""
    total_files: int = 0
    total_chunks: int = 0
    new_files: int = 0
    modified_files: int = 0
    deleted_files: int = 0
    failed_files: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class ProcessedFile:
    """Result of processing a single file."""
    file_path: str
    chunks: List[CodeChunk]
    language: str
    line_count: int
    content_hash: str
    error: Optional[str] = None


class IndexingService:
    """
    Service for indexing codebases.
    
    Coordinates file scanning, AST parsing, chunking, embedding generation,
    and vector storage with support for parallel processing and incremental updates.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClientInterface,
        vector_store: VectorStoreInterface,
        metadata_store: IndexMetadataStore,
        file_scanner: Optional[FileScannerInterface] = None,
        ast_parser: Optional[ASTParserInterface] = None,
        chunker: Optional[ChunkerInterface] = None,
        batch_size: int = 32,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """
        Initialize the indexing service.
        
        Args:
            embedding_client: Client for generating embeddings
            vector_store: Store for vectors and metadata
            metadata_store: SQLite store for index metadata
            file_scanner: Scanner for finding files (default: FileScanner)
            ast_parser: Parser for code structure (default: TreeSitterParser)
            chunker: Chunker for splitting code (default: Chunker)
            batch_size: Number of chunks to embed at once
            max_workers: Number of parallel workers for file processing
            progress_callback: Optional callback(current, total, message)
        """
        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._metadata_store = metadata_store
        self._file_scanner = file_scanner or FileScanner()
        self._ast_parser = ast_parser or TreeSitterParser()
        self._chunker = chunker or create_chunker()
        self._batch_size = batch_size
        self._max_workers = max_workers
        self._progress_callback = progress_callback

    def _report_progress(self, current: int, total: int, message: str) -> None:
        """Report progress if callback is set."""
        if self._progress_callback:
            self._progress_callback(current, total, message)
        logger.info(f"Progress: {current}/{total} - {message}")

    async def index_directory(self, root_path: Path) -> IndexingResult:
        """
        Index all files in a directory.
        
        Uses parallel processing for CPU-intensive operations (AST parsing,
        chunking) and async for IO-intensive operations (embedding, storage).
        
        Args:
            root_path: Root directory to index
            
        Returns:
            IndexingResult with statistics
        """
        start_time = time.time()
        result = IndexingResult()
        
        # Scan files
        self._report_progress(0, 0, "Scanning files...")
        files = list(self._file_scanner.scan(root_path))
        total_files = len(files)
        
        if total_files == 0:
            result.duration_seconds = time.time() - start_time
            return result
        
        # Process files in parallel and collect chunks
        self._report_progress(0, total_files, "Processing files...")
        all_chunks: List[CodeChunk] = []
        
        if self._max_workers > 1 and total_files > 1:
            # Use parallel processing for multiple files
            all_chunks, result = await self._process_files_parallel(
                files, result, total_files
            )
        else:
            # Sequential processing for single file or single worker
            all_chunks, result = await self._process_files_sequential(
                files, result, total_files
            )
        
        result.total_chunks = len(all_chunks)
        
        # Generate embeddings and store
        if all_chunks:
            await self._embed_and_store_chunks(all_chunks)
        
        result.duration_seconds = time.time() - start_time
        return result

    async def _process_files_parallel(
        self,
        files: List[ScannedFile],
        result: IndexingResult,
        total_files: int,
    ) -> Tuple[List[CodeChunk], IndexingResult]:
        """
        Process files in parallel using ThreadPoolExecutor.
        
        Note: We use ThreadPoolExecutor instead of ProcessPoolExecutor because
        the worker function needs access to tree-sitter parsers which may have
        issues with process serialization. ThreadPoolExecutor still provides
        parallelism benefits for I/O-bound operations.
        
        Args:
            files: List of scanned files to process
            result: IndexingResult to update
            total_files: Total number of files for progress reporting
            
        Returns:
            Tuple of (all_chunks, updated_result)
        """
        all_chunks: List[CodeChunk] = []
        loop = asyncio.get_event_loop()
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit all tasks
            futures = []
            for scanned_file in files:
                future = loop.run_in_executor(
                    executor,
                    _process_file_worker,
                    str(scanned_file.path),
                    scanned_file.content,
                    scanned_file.language,
                    scanned_file.content_hash,
                    scanned_file.modified_time,
                    scanned_file.size_bytes,
                )
                futures.append((future, scanned_file))
            
            # Collect results as they complete
            processed_count = 0
            for future, scanned_file in futures:
                try:
                    file_path, chunks_data, language, line_count, content_hash, error = await future
                    
                    if error:
                        result.failed_files.append(file_path)
                    else:
                        # Convert chunk dictionaries back to CodeChunk objects
                        chunks = [
                            CodeChunk(
                                chunk_id=cd["chunk_id"],
                                file_path=cd["file_path"],
                                start_line=cd["start_line"],
                                end_line=cd["end_line"],
                                content=cd["content"],
                                language=cd["language"],
                                chunk_type=cd["chunk_type"],
                                metadata=cd["metadata"],
                            )
                            for cd in chunks_data
                        ]
                        all_chunks.extend(chunks)
                        result.total_files += 1
                        
                        # Update metadata
                        self._metadata_store.upsert_file(IndexedFileInfo(
                            file_path=file_path,
                            content_hash=content_hash,
                            language=language,
                            line_count=line_count,
                            chunk_count=len(chunks),
                            indexed_at=datetime.now(),
                            modified_time=scanned_file.modified_time,
                        ))
                        
                except Exception as e:
                    logger.error(f"Failed to process {scanned_file.path}: {e}")
                    result.failed_files.append(str(scanned_file.path))
                
                processed_count += 1
                if processed_count % 10 == 0 or processed_count == total_files:
                    self._report_progress(
                        processed_count, total_files, 
                        f"Processed {processed_count} files"
                    )
        
        return all_chunks, result

    async def _process_files_sequential(
        self,
        files: List[ScannedFile],
        result: IndexingResult,
        total_files: int,
    ) -> Tuple[List[CodeChunk], IndexingResult]:
        """
        Process files sequentially (fallback for single worker).
        
        Args:
            files: List of scanned files to process
            result: IndexingResult to update
            total_files: Total number of files for progress reporting
            
        Returns:
            Tuple of (all_chunks, updated_result)
        """
        all_chunks: List[CodeChunk] = []
        
        for i, scanned_file in enumerate(files):
            try:
                processed = self._process_file(scanned_file)
                if processed.error:
                    result.failed_files.append(processed.file_path)
                else:
                    all_chunks.extend(processed.chunks)
                    result.total_files += 1
                    
                    # Update metadata
                    self._metadata_store.upsert_file(IndexedFileInfo(
                        file_path=processed.file_path,
                        content_hash=processed.content_hash,
                        language=processed.language,
                        line_count=processed.line_count,
                        chunk_count=len(processed.chunks),
                        indexed_at=datetime.now(),
                        modified_time=scanned_file.modified_time,
                    ))
            except Exception as e:
                logger.error(f"Failed to process {scanned_file.path}: {e}")
                result.failed_files.append(str(scanned_file.path))
            
            if (i + 1) % 10 == 0 or i == total_files - 1:
                self._report_progress(i + 1, total_files, f"Processed {i + 1} files")
        
        return all_chunks, result


    def _process_file(self, scanned_file: ScannedFile) -> ProcessedFile:
        """
        Process a single file: parse and chunk.
        
        Args:
            scanned_file: File to process
            
        Returns:
            ProcessedFile with chunks or error
        """
        try:
            # Parse AST
            ast_nodes = []
            if self._ast_parser.supports_language(scanned_file.language):
                ast_nodes = self._ast_parser.parse(
                    scanned_file.content, 
                    scanned_file.language
                )
            
            # Chunk the file
            chunks = self._chunker.chunk(scanned_file, ast_nodes)
            
            # Count lines
            line_count = scanned_file.content.count('\n') + 1
            
            return ProcessedFile(
                file_path=str(scanned_file.path),
                chunks=chunks,
                language=scanned_file.language,
                line_count=line_count,
                content_hash=scanned_file.content_hash,
            )
        except Exception as e:
            logger.error(f"Error processing {scanned_file.path}: {e}")
            return ProcessedFile(
                file_path=str(scanned_file.path),
                chunks=[],
                language=scanned_file.language,
                line_count=0,
                content_hash=scanned_file.content_hash,
                error=str(e),
            )

    async def _embed_and_store_chunks(self, chunks: List[CodeChunk]) -> None:
        """
        Generate embeddings and store chunks in batches.
        
        Args:
            chunks: List of chunks to embed and store
        """
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, self._batch_size):
            batch = chunks[i:i + self._batch_size]
            
            # Extract texts for embedding
            texts = [chunk.content for chunk in batch]
            
            # Generate embeddings
            embeddings = await self._embedding_client.embed_batch(texts)
            
            # Store in vector store
            for chunk, embedding in zip(batch, embeddings):
                payload = {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content": chunk.content,
                    "language": chunk.language,
                    "chunk_type": chunk.chunk_type,
                    **chunk.metadata,
                }
                await self._vector_store.upsert(chunk.chunk_id, embedding, payload)
            
            self._report_progress(
                min(i + self._batch_size, total_chunks),
                total_chunks,
                f"Embedded {min(i + self._batch_size, total_chunks)} chunks"
            )

    async def update_incremental(self, root_path: Path) -> IndexingResult:
        """
        Perform incremental update of the index.
        
        Detects new, modified, and deleted files using content hash comparison
        and updates the index accordingly:
        - New files: Index and add to vector store
        - Modified files: Remove old chunks, re-index with new content
        - Deleted files: Remove from vector store and metadata
        
        Args:
            root_path: Root directory to update
            
        Returns:
            IndexingResult with statistics including counts of new/modified/deleted files
        """
        start_time = time.time()
        result = IndexingResult()
        
        # Get current file hashes from metadata store
        self._report_progress(0, 0, "Loading existing index metadata...")
        existing_hashes = self._metadata_store.get_all_file_hashes()
        
        # Scan current files on disk
        self._report_progress(0, 0, "Scanning files...")
        current_files = {
            str(f.path): f for f in self._file_scanner.scan(root_path)
        }
        
        current_paths = set(current_files.keys())
        existing_paths = set(existing_hashes.keys())
        
        # Detect changes using hash comparison (Requirements 5.1, 5.2, 5.3, 5.4)
        new_paths = current_paths - existing_paths
        deleted_paths = existing_paths - current_paths
        common_paths = current_paths & existing_paths
        
        # Modified files are those with different content hashes
        modified_paths = {
            p for p in common_paths
            if current_files[p].content_hash != existing_hashes[p]
        }
        
        result.new_files = len(new_paths)
        result.modified_files = len(modified_paths)
        result.deleted_files = len(deleted_paths)
        
        total_changes = result.new_files + result.modified_files + result.deleted_files
        if total_changes == 0:
            logger.info("No changes detected, index is up to date")
            result.duration_seconds = time.time() - start_time
            return result
        
        logger.info(
            f"Detected changes: {result.new_files} new, "
            f"{result.modified_files} modified, {result.deleted_files} deleted"
        )
        
        # Handle deleted files - remove from vector store and metadata
        if deleted_paths:
            self._report_progress(0, len(deleted_paths), "Removing deleted files...")
            for i, path in enumerate(deleted_paths):
                await self._vector_store.delete_by_file(path)
                self._metadata_store.delete_file(path)
                if (i + 1) % 10 == 0 or i == len(deleted_paths) - 1:
                    self._report_progress(
                        i + 1, len(deleted_paths), 
                        f"Removed {i + 1} deleted files"
                    )
        
        # Handle modified files - delete old chunks first
        if modified_paths:
            self._report_progress(0, len(modified_paths), "Removing old chunks for modified files...")
            for i, path in enumerate(modified_paths):
                await self._vector_store.delete_by_file(path)
                if (i + 1) % 10 == 0 or i == len(modified_paths) - 1:
                    self._report_progress(
                        i + 1, len(modified_paths), 
                        f"Removed old chunks for {i + 1} modified files"
                    )
        
        # Process new and modified files
        files_to_process = [current_files[p] for p in (new_paths | modified_paths)]
        all_chunks: List[CodeChunk] = []
        
        if files_to_process:
            total_to_process = len(files_to_process)
            self._report_progress(0, total_to_process, "Processing new and modified files...")
            
            # Use parallel processing if multiple files and workers
            if self._max_workers > 1 and total_to_process > 1:
                all_chunks, result = await self._process_files_parallel(
                    files_to_process, result, total_to_process
                )
            else:
                all_chunks, result = await self._process_files_sequential(
                    files_to_process, result, total_to_process
                )
        
        result.total_chunks = len(all_chunks)
        
        # Embed and store new chunks
        if all_chunks:
            await self._embed_and_store_chunks(all_chunks)
        
        result.duration_seconds = time.time() - start_time
        logger.info(
            f"Incremental update completed in {result.duration_seconds:.2f}s: "
            f"{result.total_files} files, {result.total_chunks} chunks"
        )
        return result
