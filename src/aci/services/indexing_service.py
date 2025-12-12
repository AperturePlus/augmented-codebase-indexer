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
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from aci.core.ast_parser import ASTNode, ASTParserInterface, TreeSitterParser
from aci.core.chunker import Chunker, ChunkerInterface, CodeChunk, create_chunker
from aci.core.file_scanner import FileScanner, FileScannerInterface, ScannedFile
from aci.core.path_utils import get_collection_name_for_path
from aci.core.summary_artifact import ArtifactType, SummaryArtifact
from aci.infrastructure.embedding import EmbeddingClientInterface
from aci.infrastructure.metadata_store import IndexedFileInfo, IndexMetadataStore
from aci.infrastructure.vector_store import VectorStoreInterface
from aci.services.indexing_models import IndexingError, IndexingResult, ProcessedFile
from aci.services.indexing_worker import ChunkerConfig, init_worker, process_file_worker
from aci.services.metrics_collector import MetricsCollector

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
        metrics_collector: Optional[MetricsCollector] = None,
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
            metrics_collector: Optional collector for indexing metrics
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
        self._metrics_collector = metrics_collector
        
        # Extract chunker config for parallel workers
        self._chunker_config = self._extract_chunker_config(self._chunker)
        
        # Check for pending batches from previous runs
        self._check_pending_batches_on_startup()

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

    def _check_pending_batches_on_startup(self) -> None:
        """Check for pending batches from previous runs and log warning if any exist."""
        try:
            pending_batches = self._metadata_store.get_pending_batches()
            if pending_batches:
                batch_ids = [batch.batch_id for batch in pending_batches]
                logger.warning(
                    "Detected incomplete pending batches from previous runs",
                    extra={
                        "pending_batch_count": len(pending_batches),
                        "batch_ids": batch_ids,
                    },
                )
                logger.warning(
                    f"Found {len(pending_batches)} pending batch(es) that may indicate "
                    f"incomplete indexing operations. Use cleanup_pending_batches() to "
                    f"clean up these batches."
                )
        except Exception as e:
            logger.debug(f"Could not check pending batches on startup: {e}")

    def cleanup_pending_batches(self) -> int:
        """
        Clean up all pending batches from previous failed runs.
        
        This method should be called manually to recover from incomplete
        indexing operations. It rolls back each pending batch, removing
        any associated file metadata.
        
        Returns:
            Number of batches cleaned up
        """
        pending_batches = self._metadata_store.get_pending_batches()
        cleaned_count = 0
        
        for batch in pending_batches:
            logger.info(
                f"Cleaning up pending batch: {batch.batch_id}",
                extra={
                    "batch_id": batch.batch_id,
                    "file_count": len(batch.file_paths),
                    "chunk_count": len(batch.chunk_ids),
                    "created_at": batch.created_at.isoformat(),
                },
            )
            if self._metadata_store.rollback_pending_batch(batch.batch_id):
                cleaned_count += 1
                logger.info(f"Successfully cleaned up batch: {batch.batch_id}")
            else:
                logger.warning(f"Failed to clean up batch: {batch.batch_id}")
        
        if cleaned_count > 0:
            logger.info(
                f"Cleaned up {cleaned_count} pending batch(es)",
                extra={"cleaned_count": cleaned_count},
            )
        
        return cleaned_count

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
        
        # Log indexing summary with structured fields
        logger.info(
            "Indexing completed",
            extra={
                "total_chunks": result.total_chunks,
                "total_files": result.total_files,
                "duration_seconds": result.duration_seconds,
            },
        )
        
        # Record metrics if collector is available
        if self._metrics_collector:
            self._metrics_collector.record_indexing_complete(
                result.total_chunks, result.total_files, result.duration_seconds
            )
        
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
        
        After parallel chunk processing completes, summary generation is
        performed in the main process as a post-processing step. This is
        necessary because SummaryGenerator cannot be serialized across
        process boundaries.
        """
        all_chunks: List[CodeChunk] = []
        all_summaries: List[SummaryArtifact] = []
        # Track successfully processed files for post-processing summary generation
        successfully_processed_files: List[ScannedFile] = []
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
                            # Track successfully processed file for summary generation
                            successfully_processed_files.append(scanned_file)
                            
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

        # Post-processing: Generate summaries for successfully processed files
        # This runs in the main process since SummaryGenerator is not serializable
        all_summaries = self._generate_summaries_for_files(successfully_processed_files)

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

    def _generate_summaries_for_files(
        self, files: List[ScannedFile]
    ) -> List[SummaryArtifact]:
        """
        Generate summaries for a list of files in post-processing.
        
        This method is used after parallel chunk processing to generate
        summary artifacts. It re-parses AST nodes and invokes the chunker's
        summary generator for each file.
        
        Args:
            files: List of successfully processed ScannedFile objects
            
        Returns:
            List of SummaryArtifact objects generated from all files
        """
        all_summaries: List[SummaryArtifact] = []
        
        # Check if chunker has a summary generator
        if not hasattr(self._chunker, "_summary_generator"):
            return all_summaries
        
        summary_generator = getattr(self._chunker, "_summary_generator", None)
        if summary_generator is None:
            return all_summaries
        
        logger.debug(
            f"Generating summaries for {len(files)} files in post-processing"
        )
        
        for scanned_file in files:
            try:
                file_summaries = self._generate_summaries_for_single_file(
                    scanned_file, summary_generator
                )
                all_summaries.extend(file_summaries)
            except Exception as e:
                logger.warning(
                    f"Failed to generate summaries for {scanned_file.path}: {e}"
                )
                # Continue processing other files
        
        logger.debug(f"Generated {len(all_summaries)} summaries in post-processing")
        return all_summaries

    def _generate_summaries_for_single_file(
        self,
        scanned_file: ScannedFile,
        summary_generator: "SummaryGeneratorInterface",
    ) -> List[SummaryArtifact]:
        """
        Generate summaries for a single file.
        
        Re-parses AST nodes and generates function, class, and file summaries.
        
        Args:
            scanned_file: The file to generate summaries for
            summary_generator: The summary generator to use
            
        Returns:
            List of SummaryArtifact objects for this file
        """
        from aci.core.summary_generator import SummaryGeneratorInterface
        
        summaries: List[SummaryArtifact] = []
        file_path = str(scanned_file.path)
        
        # Parse AST nodes
        ast_nodes = []
        if self._ast_parser.supports_language(scanned_file.language):
            ast_nodes = self._ast_parser.parse(
                scanned_file.content, scanned_file.language
            )
        
        # Extract imports for file summary
        imports = self._extract_imports_for_summary(
            scanned_file.content, scanned_file.language
        )
        
        # Group methods by parent class for class summary generation
        class_methods: dict[str, List[ASTNode]] = {}
        for node in ast_nodes:
            if node.node_type == "method" and node.parent_name:
                if node.parent_name not in class_methods:
                    class_methods[node.parent_name] = []
                class_methods[node.parent_name].append(node)
        
        # Generate summaries for each node
        for node in ast_nodes:
            if node.node_type == "function":
                try:
                    summary = summary_generator.generate_function_summary(
                        node, file_path
                    )
                    summaries.append(summary)
                except Exception as e:
                    logger.warning(
                        f"Failed to generate function summary for {node.name}: {e}"
                    )
            elif node.node_type == "method":
                try:
                    summary = summary_generator.generate_function_summary(
                        node, file_path
                    )
                    summaries.append(summary)
                except Exception as e:
                    logger.warning(
                        f"Failed to generate method summary for {node.name}: {e}"
                    )
            elif node.node_type == "class":
                try:
                    methods = class_methods.get(node.name, [])
                    summary = summary_generator.generate_class_summary(
                        node, methods, file_path
                    )
                    summaries.append(summary)
                except Exception as e:
                    logger.warning(
                        f"Failed to generate class summary for {node.name}: {e}"
                    )
        
        # Generate file summary
        try:
            file_summary = summary_generator.generate_file_summary(
                file_path=file_path,
                language=scanned_file.language,
                imports=imports,
                nodes=ast_nodes,
            )
            summaries.append(file_summary)
        except Exception as e:
            logger.warning(f"Failed to generate file summary for {file_path}: {e}")
        
        return summaries

    def _extract_imports_for_summary(
        self, content: str, language: str
    ) -> List[str]:
        """
        Extract imports from file content for summary generation.
        
        Uses the chunker's import registry if available, otherwise returns
        an empty list.
        
        Args:
            content: File content
            language: Programming language
            
        Returns:
            List of import statements
        """
        if hasattr(self._chunker, "_import_registry"):
            import_registry = getattr(self._chunker, "_import_registry", None)
            if import_registry is not None:
                return import_registry.extract_imports(content, language)
        return []

    async def _embed_and_store_chunks(
        self,
        chunks: List[CodeChunk],
        summaries: Optional[List[SummaryArtifact]] = None,
    ) -> None:
        """
        Generate embeddings and store chunks and summaries in batches.
        
        Uses pending batch tracking to maintain consistency between Qdrant
        and SQLite. If a batch fails, the pending batch is rolled back.
        
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
            
            # Generate batch_id for pending batch tracking
            batch_id = f"batch_{uuid.uuid4().hex[:12]}"
            
            # Collect file paths and chunk IDs for this batch
            batch_file_paths = list({
                chunk.metadata.get("_pending_file_info", {}).get("file_path", chunk.file_path)
                for chunk in batch
            })
            batch_chunk_ids = [chunk.chunk_id for chunk in batch]
            
            # Create pending batch marker before any writes
            self._metadata_store.create_pending_batch(
                batch_id, batch_file_paths, batch_chunk_ids
            )
            
            try:
                texts = [chunk.content for chunk in batch]
                
                # Time embedding API call
                embed_start = time.time()
                embeddings = await self._embedding_client.embed_batch(texts)
                embed_latency_ms = (time.time() - embed_start) * 1000
                
                # Log embedding latency with structured fields
                logger.info(
                    "Embedding batch completed",
                    extra={
                        "latency_ms": embed_latency_ms,
                        "batch_index": batch_index,
                        "chunk_count": len(texts),
                    },
                )
                
                # Record metrics if collector is available
                if self._metrics_collector:
                    self._metrics_collector.record_embedding_latency(embed_latency_ms)

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
                
                # Time Qdrant upsert operations
                qdrant_start = time.time()
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
                
                qdrant_duration_ms = (time.time() - qdrant_start) * 1000
                
                # Log Qdrant operation with structured fields
                logger.info(
                    "Qdrant upsert completed",
                    extra={
                        "duration_ms": qdrant_duration_ms,
                        "chunk_count": len(batch),
                        "batch_index": batch_index,
                    },
                )
                
                # Record metrics if collector is available
                if self._metrics_collector:
                    self._metrics_collector.record_qdrant_duration(
                        qdrant_duration_ms, len(batch)
                    )

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
                
                # Batch completed successfully, clear pending marker
                self._metadata_store.complete_pending_batch(batch_id)
                
            except Exception as e:
                # Rollback pending batch on failure
                logger.error(f"Batch {batch_id} failed: {e}. Rolling back.")
                self._metadata_store.rollback_pending_batch(batch_id)
                raise

            self._report_progress(
                min(i + self._batch_size, total_chunks),
                total_items,
                f"Embedded {min(i + self._batch_size, total_chunks)} chunks",
            )

        # Process summaries (no pending batch tracking for summaries as they
        # don't have associated file metadata)
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
        
        # Log indexing summary with structured fields
        logger.info(
            "Indexing completed",
            extra={
                "total_chunks": result.total_chunks,
                "total_files": result.total_files,
                "duration_seconds": result.duration_seconds,
            },
        )
        
        # Record metrics if collector is available
        if self._metrics_collector:
            self._metrics_collector.record_indexing_complete(
                result.total_chunks, result.total_files, result.duration_seconds
            )
        
        logger.info(
            f"Incremental update completed in {result.duration_seconds:.2f}s: "
            f"{result.total_files} files, {result.total_chunks} chunks, "
            f"{len(all_summaries)} summaries"
        )
        return result
