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
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from aci.core.ast_parser import ASTParserInterface, TreeSitterParser
from aci.core.chunker import Chunker, ChunkerInterface, CodeChunk, create_chunker
from aci.core.file_scanner import FileScanner, FileScannerInterface, ScannedFile
from aci.core.path_utils import get_collection_name_for_path
from aci.core.summary_artifact import ArtifactType, SummaryArtifact
from aci.infrastructure.embedding_client import EmbeddingClientInterface
from aci.infrastructure.metadata_store import IndexedFileInfo, IndexMetadataStore
from aci.infrastructure.vector_store import VectorStoreInterface
from aci.services.indexing_models import IndexingError, IndexingResult, ProcessedFile
from aci.services.indexing_worker import ChunkerConfig, init_worker, process_file_worker

logger = logging.getLogger(__name__)


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
        
        # Extract chunker config for parallel workers
        self._chunker_config = self._extract_chunker_config(self._chunker)

    def _extract_chunker_config(self, chunker: ChunkerInterface) -> ChunkerConfig:
        """Extract configuration from chunker for parallel workers."""
        if isinstance(chunker, Chunker):
            return ChunkerConfig(
                max_tokens=chunker._max_tokens,
                fixed_chunk_lines=chunker._fixed_chunk_lines,
                overlap_lines=chunker._overlap_lines,
            )
        # Default config for custom chunker implementations
        return ChunkerConfig()

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
        
        logger.debug(f"index_directory: starting for {root_path}")
        
        # Generate and set collection name for this repository
        abs_root = str(root_path.resolve())
        collection_name = get_collection_name_for_path(abs_root)
        logger.debug(f"index_directory: collection_name={collection_name}")
        
        # Switch vector store to use repository-specific collection
        if hasattr(self._vector_store, "set_collection"):
            self._vector_store.set_collection(collection_name)
        
        # Register repository with collection name
        self._metadata_store.register_repository(abs_root, collection_name)
        logger.debug("index_directory: repository registered")

        # Scan files
        self._report_progress(0, 0, "Scanning files...")
        logger.debug("index_directory: scanning files...")
        files = list(self._file_scanner.scan(root_path))
        total_files = len(files)
        logger.debug(f"index_directory: found {total_files} files")

        if total_files == 0:
            result.duration_seconds = time.time() - start_time
            return result

        # Process files in parallel and collect chunks and summaries
        self._report_progress(0, total_files, "Processing files...")
        all_chunks: List[CodeChunk] = []
        all_summaries: List[SummaryArtifact] = []

        if self._max_workers > 1 and total_files > 1:
            all_chunks, all_summaries, result = await self._process_files_parallel(
                files, result, total_files
            )
        else:
            all_chunks, all_summaries, result = await self._process_files_sequential(
                files, result, total_files
            )

        result.total_chunks = len(all_chunks)

        # Generate embeddings and store chunks and summaries
        if all_chunks or all_summaries:
            await self._embed_and_store_chunks(all_chunks, all_summaries)

        result.duration_seconds = time.time() - start_time
        return result

    async def _process_files_parallel(
        self,
        files: List[ScannedFile],
        result: IndexingResult,
        total_files: int,
    ) -> Tuple[List[CodeChunk], List[SummaryArtifact], IndexingResult]:
        """
        Process files in parallel using ProcessPoolExecutor.
        
        Uses ProcessPoolExecutor for true parallelism of CPU-bound tasks
        (parsing and chunking). Workers are initialized with shared
        parser/chunker instances to avoid overhead.
        
        Note: Parallel workers do not generate summaries. Summary generation
        requires SummaryGenerator which is not available in worker processes.
        For summary generation, use sequential processing (max_workers=1).
        """
        all_chunks: List[CodeChunk] = []
        all_summaries: List[SummaryArtifact] = []  # Empty for parallel processing
        loop = asyncio.get_running_loop()
        
        try:
            with ProcessPoolExecutor(
                max_workers=self._max_workers,
                initializer=init_worker,
                initargs=(self._chunker_config,),
            ) as executor:
                futures = []
                for scanned_file in files:
                    future = loop.run_in_executor(
                        executor,
                        process_file_worker,
                        str(scanned_file.path),
                        scanned_file.content,
                        scanned_file.language,
                        scanned_file.content_hash,
                        scanned_file.modified_time,
                        scanned_file.size_bytes,
                    )
                    futures.append((future, scanned_file))
                
                processed_count = 0
                for future, scanned_file in futures:
                    try:
                        file_path, chunks_data, language, line_count, content_hash, error = await future
                        
                        if error:
                            result.failed_files.append(file_path)
                        else:
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
                            
                            for chunk in chunks:
                                chunk.metadata["_pending_file_info"] = {
                                    "file_path": file_path,
                                    "content_hash": content_hash,
                                    "language": language,
                                    "line_count": line_count,
                                    "chunk_count": len(chunks),
                                    "modified_time": scanned_file.modified_time,
                                }
                            
                    except Exception as e:
                        logger.error(f"Failed to process {scanned_file.path}: {e}")
                        result.failed_files.append(str(scanned_file.path))
                    
                    processed_count += 1
                    if processed_count % 10 == 0 or processed_count == total_files:
                        self._report_progress(
                            processed_count, total_files, 
                            f"Processed {processed_count} files"
                        )
        except Exception as exc:
            logger.warning(
                "ProcessPoolExecutor failed (%s). Falling back to sequential processing.",
                exc,
            )
            return await self._process_files_sequential(files, result, total_files)

        return all_chunks, all_summaries, result

    async def _process_files_sequential(
        self,
        files: List[ScannedFile],
        result: IndexingResult,
        total_files: int,
    ) -> Tuple[List[CodeChunk], List[SummaryArtifact], IndexingResult]:
        """Process files sequentially (fallback for single worker)."""
        all_chunks: List[CodeChunk] = []
        all_summaries: List[SummaryArtifact] = []

        for i, scanned_file in enumerate(files):
            try:
                processed = self._process_file(scanned_file)
                if processed.error:
                    result.failed_files.append(processed.file_path)
                else:
                    all_chunks.extend(processed.chunks)
                    all_summaries.extend(processed.summaries)
                    result.total_files += 1

                    for chunk in processed.chunks:
                        chunk.metadata["_pending_file_info"] = {
                            "file_path": processed.file_path,
                            "content_hash": processed.content_hash,
                            "language": processed.language,
                            "line_count": processed.line_count,
                            "chunk_count": len(processed.chunks),
                            "modified_time": scanned_file.modified_time,
                        }
            except Exception as e:
                logger.error(f"Failed to process {scanned_file.path}: {e}")
                result.failed_files.append(str(scanned_file.path))

            if (i + 1) % 10 == 0 or i == total_files - 1:
                self._report_progress(i + 1, total_files, f"Processed {i + 1} files")

        return all_chunks, all_summaries, result

    def _process_file(self, scanned_file: ScannedFile) -> ProcessedFile:
        """Process a single file: parse and chunk."""
        try:
            ast_nodes = []
            if self._ast_parser.supports_language(scanned_file.language):
                ast_nodes = self._ast_parser.parse(scanned_file.content, scanned_file.language)

            chunking_result = self._chunker.chunk(scanned_file, ast_nodes)
            line_count = scanned_file.content.count("\n") + 1

            return ProcessedFile(
                file_path=str(scanned_file.path),
                chunks=chunking_result.chunks,
                summaries=chunking_result.summaries,
                language=scanned_file.language,
                line_count=line_count,
                content_hash=scanned_file.content_hash,
            )
        except Exception as e:
            logger.error(f"Error processing {scanned_file.path}: {e}")
            return ProcessedFile(
                file_path=str(scanned_file.path),
                chunks=[],
                summaries=[],
                language=scanned_file.language,
                line_count=0,
                content_hash=scanned_file.content_hash,
                error=str(e),
            )


    async def _embed_and_store_chunks(
        self,
        chunks: List[CodeChunk],
        summaries: Optional[List[SummaryArtifact]] = None,
    ) -> None:
        """
        Generate embeddings and store chunks and summaries in batches.
        
        Writes metadata immediately after each batch succeeds to maintain
        consistency between Qdrant and SQLite if a later batch fails.
        
        Args:
            chunks: List of code chunks to embed and store
            summaries: Optional list of summary artifacts to embed and store
            
        Raises:
            IndexingError: If embedding API returns fewer vectors than expected
        """
        summaries = summaries or []
        total_chunks = len(chunks)
        total_summaries = len(summaries)
        total_items = total_chunks + total_summaries
        
        # Track files already persisted to avoid duplicate metadata writes
        persisted_files: set[str] = set()

        # Process chunks first
        for i in range(0, total_chunks, self._batch_size):
            batch = chunks[i : i + self._batch_size]
            batch_index = i // self._batch_size

            texts = [chunk.content for chunk in batch]
            embeddings = await self._embedding_client.embed_batch(texts)

            if len(embeddings) != len(texts):
                raise IndexingError(
                    f"Embedding count mismatch in chunk batch {batch_index}: "
                    f"expected {len(texts)}, got {len(embeddings)}. "
                    f"This may indicate API rate limiting or content issues.",
                    batch_index=batch_index,
                    expected=len(texts),
                    actual=len(embeddings),
                )

            # Collect file info from this batch before writing vectors
            batch_file_infos: dict[str, dict] = {}
            for chunk, embedding in zip(batch, embeddings):
                pending_info = chunk.metadata.pop("_pending_file_info", None)
                
                payload = {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content": chunk.content,
                    "language": chunk.language,
                    "chunk_type": chunk.chunk_type,
                    "artifact_type": ArtifactType.CHUNK.value,
                    **chunk.metadata,
                }
                await self._vector_store.upsert(chunk.chunk_id, embedding, payload)
                
                file_path = pending_info["file_path"] if pending_info else None
                if file_path and file_path not in persisted_files:
                    batch_file_infos[file_path] = pending_info

            # Write metadata for this batch immediately after vectors succeed
            for file_path, info in batch_file_infos.items():
                self._metadata_store.upsert_file(
                    IndexedFileInfo(
                        file_path=info["file_path"],
                        content_hash=info["content_hash"],
                        language=info["language"],
                        line_count=info["line_count"],
                        chunk_count=info["chunk_count"],
                        indexed_at=datetime.now().astimezone(),
                        modified_time=info["modified_time"],
                    )
                )
                persisted_files.add(file_path)

            self._report_progress(
                min(i + self._batch_size, total_chunks),
                total_items,
                f"Embedded {min(i + self._batch_size, total_chunks)} chunks",
            )

        # Process summaries
        if summaries:
            for i in range(0, total_summaries, self._batch_size):
                batch = summaries[i : i + self._batch_size]
                batch_index = i // self._batch_size

                texts = [summary.content for summary in batch]
                embeddings = await self._embedding_client.embed_batch(texts)

                if len(embeddings) != len(texts):
                    raise IndexingError(
                        f"Embedding count mismatch in summary batch {batch_index}: "
                        f"expected {len(texts)}, got {len(embeddings)}. "
                        f"This may indicate API rate limiting or content issues.",
                        batch_index=batch_index,
                        expected=len(texts),
                        actual=len(embeddings),
                    )

                for summary, embedding in zip(batch, embeddings):
                    payload = {
                        "file_path": summary.file_path,
                        "start_line": summary.start_line,
                        "end_line": summary.end_line,
                        "content": summary.content,
                        "artifact_type": summary.artifact_type.value,
                        "name": summary.name,
                        **summary.metadata,
                    }
                    await self._vector_store.upsert(summary.artifact_id, embedding, payload)

                self._report_progress(
                    total_chunks + min(i + self._batch_size, total_summaries),
                    total_items,
                    f"Embedded {min(i + self._batch_size, total_summaries)} summaries",
                )

    async def update_incremental(self, root_path: Path) -> IndexingResult:
        """
        Perform incremental update of the index.

        Detects new, modified, and deleted files using content hash comparison
        and updates the index accordingly.
        """
        start_time = time.time()
        result = IndexingResult()
        
        abs_root = str(root_path.resolve())
        collection_name = get_collection_name_for_path(abs_root)
        
        if hasattr(self._vector_store, "set_collection"):
            self._vector_store.set_collection(collection_name)
        
        self._metadata_store.register_repository(abs_root, collection_name)

        self._report_progress(0, 0, "Loading existing index metadata...")
        existing_hashes = self._metadata_store.get_all_file_hashes()

        self._report_progress(0, 0, "Scanning files...")
        current_files = {str(f.path): f for f in self._file_scanner.scan(root_path)}

        current_paths = set(current_files.keys())
        existing_paths = set(existing_hashes.keys())

        new_paths = current_paths - existing_paths
        deleted_paths = existing_paths - current_paths
        common_paths = current_paths & existing_paths

        modified_paths = {
            p for p in common_paths if current_files[p].content_hash != existing_hashes[p]
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

        if deleted_paths:
            self._report_progress(0, len(deleted_paths), "Removing deleted files...")
            for i, path in enumerate(deleted_paths):
                await self._vector_store.delete_by_file(path)
                self._metadata_store.delete_file(path)
                if (i + 1) % 10 == 0 or i == len(deleted_paths) - 1:
                    self._report_progress(
                        i + 1, len(deleted_paths), f"Removed {i + 1} deleted files"
                    )

        if modified_paths:
            self._report_progress(
                0, len(modified_paths), "Removing old artifacts for modified files..."
            )
            for i, path in enumerate(modified_paths):
                # delete_by_file removes all artifacts (chunks and summaries) for the file
                await self._vector_store.delete_by_file(path)
                if (i + 1) % 10 == 0 or i == len(modified_paths) - 1:
                    self._report_progress(
                        i + 1, len(modified_paths), f"Removed old artifacts for {i + 1} modified files"
                    )

        files_to_process = [current_files[p] for p in (new_paths | modified_paths)]
        all_chunks: List[CodeChunk] = []
        all_summaries: List[SummaryArtifact] = []

        if files_to_process:
            total_to_process = len(files_to_process)
            self._report_progress(0, total_to_process, "Processing new and modified files...")

            if self._max_workers > 1 and total_to_process > 1:
                all_chunks, all_summaries, result = await self._process_files_parallel(
                    files_to_process, result, total_to_process
                )
            else:
                all_chunks, all_summaries, result = await self._process_files_sequential(
                    files_to_process, result, total_to_process
                )

        result.total_chunks = len(all_chunks)

        if all_chunks or all_summaries:
            await self._embed_and_store_chunks(all_chunks, all_summaries)

        result.duration_seconds = time.time() - start_time
        logger.info(
            f"Incremental update completed in {result.duration_seconds:.2f}s: "
            f"{result.total_files} files, {result.total_chunks} chunks, "
            f"{len(all_summaries)} summaries"
        )
        return result
