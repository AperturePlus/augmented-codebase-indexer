"""
CLI for Project ACI.

Provides command-line interface for indexing and searching codebases.
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer

from aci.core.config import load_config
from aci.infrastructure import (
    IndexMetadataStore,
    create_embedding_client,
    create_metadata_store,
    create_vector_store,
)
from aci.services import IndexingService, SearchService

app = typer.Typer(
    name="aci",
    help="Augmented Codebase Indexer - Semantic code search",
    add_completion=False,
)


def get_services(config_path: Optional[Path] = None):
    """Initialize services from config."""
    config = load_config(config_path)
    
    embedding_client = create_embedding_client(
        api_url=config.embedding.api_url,
        api_key=config.embedding.api_key,
        model=config.embedding.model,
        batch_size=config.embedding.batch_size,
        max_retries=config.embedding.max_retries,
    )
    
    vector_store = create_vector_store(
        host=config.vector_store.host,
        port=config.vector_store.port,
        collection_name=config.vector_store.collection_name,
        vector_size=config.vector_store.vector_size,
    )
    
    # Use default metadata store path
    metadata_store = create_metadata_store(Path(".aci/index.db"))
    
    return config, embedding_client, vector_store, metadata_store


@app.command()
def index(
    path: Path = typer.Argument(..., help="Directory to index"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers"),
):
    """Index a directory for semantic search."""
    typer.echo(f"Indexing {path}...")
    
    try:
        cfg, embedding_client, vector_store, metadata_store = get_services(config)
        
        indexing_service = IndexingService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            metadata_store=metadata_store,
            max_workers=workers,
            progress_callback=lambda cur, tot, msg: typer.echo(f"  [{cur}/{tot}] {msg}"),
        )
        
        result = asyncio.run(indexing_service.index_directory(path))
        
        typer.echo(f"\nIndexing complete:")
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
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
    file_filter: Optional[str] = typer.Option(None, "--filter", "-f", help="File path filter (glob)"),
):
    """Search the indexed codebase."""
    try:
        cfg, embedding_client, vector_store, metadata_store = get_services(config)
        
        search_service = SearchService(
            embedding_client=embedding_client,
            vector_store=vector_store,
        )
        
        results = asyncio.run(search_service.search(
            query=query,
            limit=limit,
            file_filter=file_filter,
        ))
        
        if not results:
            typer.echo("No results found.")
            return
        
        typer.echo(f"Found {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            typer.echo(f"{i}. {result.file_path}:{result.start_line}-{result.end_line} (score: {result.score:.3f})")
            # Show first 2 lines of content
            lines = result.content.split('\n')[:2]
            for line in lines:
                typer.echo(f"   {line[:80]}")
            typer.echo()
            
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def update(
    path: Path = typer.Argument(..., help="Directory to update"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path"),
):
    """Incrementally update the index."""
    typer.echo(f"Updating index for {path}...")
    
    try:
        cfg, embedding_client, vector_store, metadata_store = get_services(config)
        
        indexing_service = IndexingService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            metadata_store=metadata_store,
        )
        
        result = asyncio.run(indexing_service.update_incremental(path))
        
        typer.echo(f"\nUpdate complete:")
        typer.echo(f"  New files: {result.new_files}")
        typer.echo(f"  Modified: {result.modified_files}")
        typer.echo(f"  Deleted: {result.deleted_files}")
        typer.echo(f"  Duration: {result.duration_seconds:.2f}s")
        
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def status(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path"),
):
    """Show index status and statistics."""
    try:
        cfg, embedding_client, vector_store, metadata_store = get_services(config)
        
        # Get metadata stats
        stats = metadata_store.get_stats()
        
        typer.echo("Index Status:")
        typer.echo(f"  Total files: {stats['total_files']}")
        typer.echo(f"  Total chunks: {stats['total_chunks']}")
        typer.echo(f"  Total lines: {stats['total_lines']}")
        
        if stats['languages']:
            typer.echo("  Languages:")
            for lang, count in stats['languages'].items():
                typer.echo(f"    {lang}: {count} files")
                
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
