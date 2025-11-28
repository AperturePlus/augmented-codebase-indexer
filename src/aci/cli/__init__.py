"""
CLI for Project ACI.

Provides command-line interface for indexing and searching codebases.
"""

import asyncio
import time
from pathlib import Path
from typing import Optional

import typer

from aci.core.chunker import create_chunker
from aci.core.config import load_config
from aci.core.file_scanner import FileScanner
from aci.core.qdrant_launcher import ensure_qdrant_running
from aci.infrastructure import (
    IndexMetadataStore,
    create_embedding_client,
    create_metadata_store,
    create_vector_store,
)
from aci.services import IndexingService, SearchService
from aci.services.reranker import (
    OpenAICompatibleReranker,
    SimpleReranker,
)

app = typer.Typer(
    name="aci",
    help="Augmented Codebase Indexer - Semantic code search",
    add_completion=False,
)


class ProgressReporter:
    """Progress reporter with ETA calculation."""

    def __init__(self, echo_func=typer.echo):
        self._start_time: Optional[float] = None
        self._echo = echo_func

    def __call__(self, current: int, total: int, message: str) -> None:
        """Report progress with ETA."""
        if self._start_time is None:
            self._start_time = time.time()

        if total > 0 and current > 0:
            elapsed = time.time() - self._start_time
            rate = current / elapsed if elapsed > 0 else 0
            remaining = total - current
            eta_seconds = remaining / rate if rate > 0 else 0

            if eta_seconds > 60:
                eta_str = f"ETA: {eta_seconds / 60:.1f}m"
            else:
                eta_str = f"ETA: {eta_seconds:.0f}s"

            self._echo(f"  [{current}/{total}] {message} ({eta_str})")
        else:
            self._echo(f"  [{current}/{total}] {message}")


def get_services():
    """Initialize services from .env with config-driven settings."""
    config = load_config()

    # Best-effort auto-start of Qdrant on the configured port
    ensure_qdrant_running(port=config.vector_store.port)

    embedding_client = create_embedding_client(
        api_url=config.embedding.api_url,
        api_key=config.embedding.api_key,
        model=config.embedding.model,
        batch_size=config.embedding.batch_size,
        max_retries=config.embedding.max_retries,
        timeout=config.embedding.timeout,
        dimension=config.embedding.dimension,
    )

    vector_store = create_vector_store(
        host=config.vector_store.host,
        port=config.vector_store.port,
        collection_name=config.vector_store.collection_name,
        vector_size=config.vector_store.vector_size,
    )

    # Use default metadata store path
    metadata_store = create_metadata_store(Path(".aci/index.db"))

    # Create file scanner with config-driven settings (Req 1.1, 7.1, 7.2)
    file_scanner = FileScanner(
        extensions=set(config.indexing.file_extensions),
        ignore_patterns=config.indexing.ignore_patterns,
    )

    # Create chunker with config-driven settings (Req 2.5)
    chunker = create_chunker(
        max_tokens=config.indexing.max_chunk_tokens,
        overlap_lines=config.indexing.chunk_overlap_lines,
    )

    # Reranker selection based on config (prefer API-based reranker)
    reranker = None
    if config.search.use_rerank:
        if config.search.rerank_api_url:
            reranker = OpenAICompatibleReranker(
                api_url=config.search.rerank_api_url,
                api_key=config.search.rerank_api_key,
                model=config.search.rerank_model,
                timeout=config.search.rerank_timeout,
                endpoint=config.search.rerank_endpoint,
            )
        else:
            reranker = SimpleReranker()

    return (
        config,
        embedding_client,
        vector_store,
        metadata_store,
        file_scanner,
        chunker,
        reranker,
    )


@app.command()
def index(
    path: Path = typer.Argument(..., help="Directory to index"),
    workers: Optional[int] = typer.Option(
        None, "--workers", "-w", help="Number of parallel workers"
    ),
):
    """Index a directory for semantic search."""
    typer.echo(f"Indexing {path}...")

    try:
        (
            cfg,
            embedding_client,
            vector_store,
            metadata_store,
            file_scanner,
            chunker,
            reranker,
        ) = get_services()

        # Use config workers if not overridden by CLI
        actual_workers = workers if workers is not None else cfg.indexing.max_workers

        indexing_service = IndexingService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            metadata_store=metadata_store,
            file_scanner=file_scanner,
            chunker=chunker,
            max_workers=actual_workers,
            batch_size=cfg.embedding.batch_size,
            progress_callback=ProgressReporter(),
        )

        result = asyncio.run(indexing_service.index_directory(path))

        typer.echo("\nIndexing complete:")
        typer.echo(f"  Files: {result.total_files}")
        typer.echo(f"  Chunks: {result.total_chunks}")
        typer.echo(f"  Duration: {result.duration_seconds:.2f}s")

        if result.failed_files:
            typer.echo(f"  Failed: {len(result.failed_files)}")
            for f in result.failed_files[:5]:
                typer.echo(f"    - {f}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Number of results"),
    file_filter: Optional[str] = typer.Option(
        None, "--filter", "-f", help="File path filter (glob)"
    ),
    rerank: Optional[bool] = typer.Option(
        None, "--rerank/--no-rerank", help="Enable/disable reranking"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full chunk content"),
    snippet_lines: int = typer.Option(
        3, "--snippet-lines", help="Number of lines to display when not verbose"
    ),
):
    """Search the indexed codebase."""
    try:
        (
            cfg,
            embedding_client,
            vector_store,
            metadata_store,
            file_scanner,
            chunker,
            config_reranker,
        ) = get_services()

        # Use config values if not overridden by CLI
        actual_limit = limit if limit is not None else cfg.search.default_limit
        use_rerank = rerank if rerank is not None else cfg.search.use_rerank

        reranker = config_reranker if use_rerank else None
        if use_rerank and reranker is None:
            typer.echo(
                "Warning: rerank requested but no reranker available; continuing without rerank",
                err=True,
            )
            use_rerank = False

        search_service = SearchService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            reranker=reranker,
            default_limit=actual_limit,
        )

        results = asyncio.run(
            search_service.search(
                query=query,
                limit=actual_limit,
                file_filter=file_filter,
                use_rerank=use_rerank and reranker is not None,
            )
        )

        if not results:
            typer.echo("No results found.")
            return

        typer.echo(f"Found {len(results)} results:\n")

        for i, result in enumerate(results, 1):
            typer.echo(
                f"{i}. {result.file_path}:{result.start_line}-{result.end_line} (score: {result.score:.3f})"
            )
            lines = result.content.split("\n")
            if verbose:
                for line in lines:
                    typer.echo(f"   {line}")
            else:
                for line in lines[:snippet_lines]:
                    typer.echo(f"   {line[:120]}")
            typer.echo()

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def update(
    path: Path = typer.Argument(..., help="Directory to update"),
):
    """Incrementally update the index."""
    typer.echo(f"Updating index for {path}...")

    try:
        (
            cfg,
            embedding_client,
            vector_store,
            metadata_store,
            file_scanner,
            chunker,
            reranker,
        ) = get_services()

        indexing_service = IndexingService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            metadata_store=metadata_store,
            file_scanner=file_scanner,
            chunker=chunker,
            max_workers=cfg.indexing.max_workers,
            batch_size=cfg.embedding.batch_size,
            progress_callback=ProgressReporter(),
        )

        result = asyncio.run(indexing_service.update_incremental(path))

        typer.echo("\nUpdate complete:")
        typer.echo(f"  New files: {result.new_files}")
        typer.echo(f"  Modified: {result.modified_files}")
        typer.echo(f"  Deleted: {result.deleted_files}")
        typer.echo(f"  Duration: {result.duration_seconds:.2f}s")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def reset():
    """Clear vector store collection and metadata."""
    typer.echo("Resetting index (vector store collection + metadata)...")
    try:
        (
            cfg,
            embedding_client,
            vector_store,
            metadata_store,
            file_scanner,
            chunker,
            reranker,
        ) = get_services()

        # Reset vector store if supported
        if hasattr(vector_store, "reset"):
            asyncio.run(vector_store.reset())
            typer.echo("Vector store collection reset.")
        else:
            typer.echo("Vector store does not support reset; skipping.", err=True)

        # Clear metadata
        metadata_store.clear_all()
        typer.echo("Metadata store cleared.")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def status():
    """Show index status, statistics, and health information."""
    try:
        (
            cfg,
            embedding_client,
            vector_store,
            metadata_store,
            file_scanner,
            chunker,
            reranker,
        ) = get_services()

        # Get metadata stats
        stats = metadata_store.get_stats()

        typer.echo("Index Status:")
        typer.echo(f"  Total files: {stats['total_files']}")
        typer.echo(f"  Total chunks: {stats['total_chunks']}")
        typer.echo(f"  Total lines: {stats['total_lines']}")

        if stats["languages"]:
            typer.echo("  Languages:")
            for lang, count in stats["languages"].items():
                typer.echo(f"    {lang}: {count} files")

        # Health checks (Req 8.3)
        typer.echo("\nHealth:")

        # Check vector store connectivity
        try:
            vector_stats = asyncio.run(vector_store.get_stats())
            typer.echo(f"  Vector Store: OK ({vector_stats.get('total_vectors', 0)} vectors)")
        except Exception as e:
            typer.echo(f"  Vector Store: ERROR - {e}", err=True)

        # Check embedding service connectivity (basic check)
        typer.echo(f"  Embedding API: {cfg.embedding.api_url}")
        typer.echo(f"  Embedding Model: {cfg.embedding.model}")

        # Show config info
        typer.echo("\nConfiguration:")
        typer.echo(f"  File extensions: {', '.join(cfg.indexing.file_extensions)}")
        typer.echo(f"  Max chunk tokens: {cfg.indexing.max_chunk_tokens}")
        typer.echo(f"  Rerank enabled: {cfg.search.use_rerank}")

    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="HTTP host"),
    port: int = typer.Option(8000, "--port", "-p", help="HTTP port"),
):
    """Start the HTTP API server."""
    import uvicorn

    try:
        # Build FastAPI app directly (ensures config + Qdrant checks)
        app = create_app()
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,
            log_level="info",
        )
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
