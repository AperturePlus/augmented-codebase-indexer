"""
CLI for Project ACI.

Provides command-line interface for indexing and searching codebases.
"""

import asyncio
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
from aci.core.path_utils import validate_indexable_path
from aci.core.qdrant_launcher import ensure_qdrant_running
from aci.infrastructure import (
    GrepSearcher,
    IndexMetadataStore,
    create_embedding_client,
    create_metadata_store,
    create_vector_store,
)
from aci.services import IndexingService, SearchMode, SearchService
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
    # Validate path before proceeding
    validation = validate_indexable_path(path)
    if not validation.valid:
        console.print(f"[bold red]Error:[/bold red] {validation.error_message}")
        raise typer.Exit(1)

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
    path: Optional[Path] = typer.Option(
        None, "--path", "-p", help="Target codebase path to search"
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

        # Determine the search base path
        search_base = path if path is not None else Path.cwd()
        
        # Validate path if provided
        if path is not None:
            if not path.exists():
                console.print(f"[bold red]Error:[/bold red] Path '{path}' does not exist")
                raise typer.Exit(1)
            if not path.is_dir():
                console.print(f"[bold red]Error:[/bold red] Path '{path}' is not a directory")
                raise typer.Exit(1)
            # Check if path is indexed
            resolved = str(path.resolve())
            if metadata_store.get_index_info(resolved) is None:
                console.print(
                    f"[bold red]Error:[/bold red] Path '{path}' has not been indexed. "
                    f"Run 'aci index {path}' first."
                )
                raise typer.Exit(1)

        # Use config values if not overridden by CLI
        actual_limit = limit if limit is not None else cfg.search.default_limit
        use_rerank = rerank if rerank is not None else cfg.search.use_rerank

        reranker = config_reranker if use_rerank else None
        if use_rerank and reranker is None:
            console.print(
                "[yellow]Warning: rerank requested but no reranker available; continuing without rerank[/yellow]"
            )
            use_rerank = False

        # Get collection name for this codebase (pass explicitly to search, no mutation)
        # For backward compatibility, generate collection name if not stored
        search_base_abs = str(search_base.resolve())
        collection_name = metadata_store.get_collection_name(search_base_abs)
        if not collection_name:
            # Legacy index without collection_name - generate and update
            from aci.core.path_utils import get_collection_name_for_path
            collection_name = get_collection_name_for_path(search_base_abs)
            metadata_store.register_repository(search_base_abs, collection_name)

        # Create GrepSearcher for hybrid search support
        grep_searcher = GrepSearcher(base_path=str(search_base))

        search_service = SearchService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            reranker=reranker,
            grep_searcher=grep_searcher,
            default_limit=actual_limit,
        )

        with console.status(f"[bold blue]Searching for[/bold blue] '{query}'..."):
            results = asyncio.run(
                search_service.search(
                    query=query,
                    limit=actual_limit,
                    file_filter=file_filter,  # User-provided filter only
                    use_rerank=use_rerank and reranker is not None,
                    collection_name=collection_name,  # Pass explicitly, no state mutation
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
    # Validate path before proceeding
    validation = validate_indexable_path(path)
    if not validation.valid:
        console.print(f"[bold red]Error:[/bold red] {validation.error_message}")
        raise typer.Exit(1)

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
    """Clear all vector store collections and metadata."""
    if not typer.confirm("Are you sure you want to reset the index? This will delete all data."):
        raise typer.Abort()

    console.print("[bold yellow]Resetting index (all vector store collections + metadata)...[/bold yellow]")
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

        # Get all registered repositories and their collections
        repos = metadata_store.get_repositories()
        deleted_collections = []
        failed_collections = []
        
        # Delete each repository's collection
        for repo in repos:
            collection_name = repo.get("collection_name")
            if collection_name:
                try:
                    deleted = asyncio.run(vector_store.delete_collection(collection_name))
                    if deleted:
                        deleted_collections.append(collection_name)
                except Exception as e:
                    failed_collections.append((collection_name, str(e)))
        
        # Also delete the default collection if it exists and wasn't already deleted
        default_collection = cfg.vector_store.collection_name
        if default_collection not in deleted_collections:
            try:
                deleted = asyncio.run(vector_store.delete_collection(default_collection))
                if deleted:
                    deleted_collections.append(default_collection)
            except Exception as e:
                failed_collections.append((default_collection, str(e)))
        
        if deleted_collections:
            console.print(f"  [green]✓[/green] Deleted {len(deleted_collections)} collection(s).")
        else:
            console.print("  [yellow]![/yellow] No collections found to delete.")
        
        if failed_collections:
            console.print(f"  [yellow]![/yellow] Failed to delete {len(failed_collections)} collection(s):")
            for name, error in failed_collections:
                console.print(f"      - {name}: {error}")

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


def _is_port_available(host: str, port: int) -> bool:
    """Check if a port is available for binding."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _find_available_port(host: str, start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for offset in range(max_attempts):
        port = start_port + offset
        if _is_port_available(host, port):
            return port
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts - 1}")


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host", help="HTTP host (default from ACI_SERVER_HOST or 0.0.0.0)"),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="HTTP port (default from ACI_SERVER_PORT or 8000)"),
):
    """Start the HTTP API server."""
    import uvicorn

    try:
        from aci import create_app
        from aci.core.config import load_config

        # Load config to get server defaults
        cfg = load_config()
        actual_host = host if host is not None else cfg.server.host
        requested_port = port if port is not None else cfg.server.port

        # Find available port if requested port is occupied
        actual_port = _find_available_port(actual_host, requested_port)
        if actual_port != requested_port:
            console.print(f"[yellow]Port {requested_port} is in use, using port {actual_port}[/yellow]")

        app = create_app()
        console.print(f"[bold green]Starting API server at http://{actual_host}:{actual_port}[/bold green]")
        uvicorn.run(
            app,
            host=actual_host,
            port=actual_port,
            reload=False,
            log_level="info",
        )
    except ImportError:
         console.print("[bold red]Error:[/bold red] Could not import create_app. Ensure aci package is installed correctly.")
         raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def shell():
    """Start interactive REPL mode for executing multiple commands."""
    try:
        from aci.cli.repl import REPLController
        from aci.cli.services import create_services

        # Initialize services once
        console.print("[dim]Initializing services...[/dim]")
        services = create_services()

        # Create and run REPL controller
        repl = REPLController(services=services, console=console)
        repl.run()

    except KeyboardInterrupt:
        console.print("\n[cyan]Goodbye![/cyan]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()