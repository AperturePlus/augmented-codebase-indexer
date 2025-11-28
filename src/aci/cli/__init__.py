"""
CLI for Project ACI.

Provides command-line interface for indexing and searching codebases.
"""

import asyncio
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.syntax import Syntax
from rich.table import Table

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

# Initialize Rich Console
console = Console()

app = typer.Typer(
    name="aci",
    help="Augmented Codebase Indexer - Semantic code search",
    add_completion=False,
)

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
    console.print(f"[bold blue]Indexing[/bold blue] {path}...")

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

        # Progress reporting with Rich
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing...", total=None)

            def update_progress(current: int, total: int, message: str) -> None:
                progress.update(task, completed=current, total=total, description=message)

            indexing_service = IndexingService(
                embedding_client=embedding_client,
                vector_store=vector_store,
                metadata_store=metadata_store,
                file_scanner=file_scanner,
                chunker=chunker,
                max_workers=actual_workers,
                batch_size=cfg.embedding.batch_size,
                progress_callback=update_progress,
            )

            result = asyncio.run(indexing_service.index_directory(path))

        # Summary Panel
        summary = Table.grid(padding=1)
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Total Files:", str(result.total_files))
        summary.add_row("Total Chunks:", str(result.total_chunks))
        summary.add_row("Duration:", f"{result.duration_seconds:.2f}s")
        
        if result.failed_files:
            summary.add_row("Failed Files:", f"[red]{len(result.failed_files)}[/red]")

        console.print(
            Panel(
                summary,
                title="[bold green]Indexing Complete[/bold green]",
                border_style="green",
                expand=False,
            )
        )

        if result.failed_files:
            console.print("\n[bold red]Failed Files:[/bold red]")
            for f in result.failed_files[:5]:
                console.print(f"  - {f}")
            if len(result.failed_files) > 5:
                console.print(f"  ... and {len(result.failed_files) - 5} more")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
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
            console.print(
                "[yellow]Warning: rerank requested but no reranker available; continuing without rerank[/yellow]"
            )
            use_rerank = False

        search_service = SearchService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            reranker=reranker,
            default_limit=actual_limit,
        )

        with console.status(f"[bold blue]Searching for[/bold blue] '{query}'..."):
            results = asyncio.run(
                search_service.search(
                    query=query,
                    limit=actual_limit,
                    file_filter=file_filter,
                    use_rerank=use_rerank and reranker is not None,
                )
            )

        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return

        console.print(f"\nFound [bold]{len(results)}[/bold] results:\n")

        for i, result in enumerate(results, 1):
            # Highlight code syntax
            language = result.metadata.get("language", "text")
            # Mapping basic names to rich syntax lexers if needed, usually auto-detected
            
            if verbose:
                code_content = result.content
            else:
                lines = result.content.split("\n")
                code_content = "\n".join(lines[:snippet_lines])
                if len(lines) > snippet_lines:
                    code_content += "\n..."

            syntax = Syntax(
                code_content,
                language,
                theme="monokai",
                line_numbers=True,
                start_line=result.start_line,
                word_wrap=True,
            )

            panel = Panel(
                syntax,
                title=f"[bold blue]{i}. {result.file_path}[/bold blue] : {result.start_line}-{result.end_line}",
                subtitle=f"Score: [yellow]{result.score:.3f}[/yellow]",
                border_style="blue",
                expand=True,
            )
            console.print(panel)
            console.print()  # spacing

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def update(
    path: Path = typer.Argument(..., help="Directory to update"),
):
    """Incrementally update the index."""
    console.print(f"[bold blue]Updating index for[/bold blue] {path}...")

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

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning...", total=None)

            def update_progress(current: int, total: int, message: str) -> None:
                progress.update(task, completed=current, total=total, description=message)

            indexing_service = IndexingService(
                embedding_client=embedding_client,
                vector_store=vector_store,
                metadata_store=metadata_store,
                file_scanner=file_scanner,
                chunker=chunker,
                max_workers=cfg.indexing.max_workers,
                batch_size=cfg.embedding.batch_size,
                progress_callback=update_progress,
            )

            result = asyncio.run(indexing_service.update_incremental(path))

        # Summary Panel
        summary = Table.grid(padding=1)
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("New Files:", f"[green]{result.new_files}[/green]")
        summary.add_row("Modified Files:", f"[yellow]{result.modified_files}[/yellow]")
        summary.add_row("Deleted Files:", f"[red]{result.deleted_files}[/red]")
        summary.add_row("Duration:", f"{result.duration_seconds:.2f}s")

        console.print(
            Panel(
                summary,
                title="[bold green]Update Complete[/bold green]",
                border_style="green",
                expand=False,
            )
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def reset():
    """Clear vector store collection and metadata."""
    if not typer.confirm("Are you sure you want to reset the index? This will delete all data."):
        raise typer.Abort()

    console.print("[bold yellow]Resetting index (vector store collection + metadata)...[/bold yellow]")
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
            console.print("  [green]✓[/green] Vector store collection reset.")
        else:
            console.print("  [yellow]![/yellow] Vector store does not support reset; skipping.")

        # Clear metadata
        metadata_store.clear_all()
        console.print("  [green]✓[/green] Metadata store cleared.")
        
        console.print("[bold green]Reset complete.[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command("list")
def list_repos():
    """List all indexed repositories."""
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

        repos = metadata_store.get_repositories()

        if not repos:
            console.print("[yellow]No repositories indexed.[/yellow]")
            return

        table = Table(title="Indexed Repositories", border_style="blue")
        table.add_column("Root Path", style="cyan", no_wrap=True)
        table.add_column("Last Updated", style="magenta")

        for repo in repos:
            table.add_row(repo["root_path"], str(repo["updated_at"]))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def status(
):
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

        # Status Grid
        grid = Table.grid(padding=1)
        grid.add_column(style="bold")
        grid.add_column()
        
        grid.add_row("Total Files:", str(stats["total_files"]))
        grid.add_row("Total Chunks:", str(stats["total_chunks"]))
        grid.add_row("Total Lines:", str(stats["total_lines"]))
        
        console.print(Panel(grid, title="Index Statistics", border_style="blue", expand=False))

        if stats["languages"]:
            lang_table = Table(title="Languages", box=None, show_header=True)
            lang_table.add_column("Language", style="cyan")
            lang_table.add_column("Files", justify="right")
            
            for lang, count in stats["languages"].items():
                lang_table.add_row(lang, str(count))
            
            console.print(Panel(lang_table, border_style="blue", expand=False))

        # Health checks (Req 8.3)
        console.print("\n[bold]System Health:[/bold]")
        
        # Check vector store connectivity
        try:
            vector_stats = asyncio.run(vector_store.get_stats())
            count = vector_stats.get("total_vectors", 0)
            console.print(f"  [green]✓[/green] Vector Store: Connected ({count} vectors)")
        except Exception as e:
            console.print(f"  [red]✗[/red] Vector Store: Error - {e}")

        # Check embedding service connectivity
        console.print(f"  [green]✓[/green] Embedding API: {cfg.embedding.api_url} ({cfg.embedding.model})")

        # Configuration summary
        config_summary = f"""File Extensions: {', '.join(cfg.indexing.file_extensions)}
Max Chunk Tokens: {cfg.indexing.max_chunk_tokens}
Rerank Enabled: {cfg.search.use_rerank}"""
        
        console.print(Panel(config_summary, title="Configuration", border_style="dim", expand=False))

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
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
        # Note: Ensure create_app is available in aci.__main__ or similar if not here
        from aci import create_app
        
        app = create_app()
        console.print(f"[bold green]Starting API server at http://{host}:{port}[/bold green]")
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,
            log_level="info",
        )
    except ImportError:
         console.print("[bold red]Error:[/bold red] Could not import create_app. Ensure aci package is installed correctly.")
         raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()