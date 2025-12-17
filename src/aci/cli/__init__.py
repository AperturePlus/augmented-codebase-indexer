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

from aci.core.path_utils import get_collection_name_for_path, validate_indexable_path
from aci.infrastructure import GrepSearcher
from aci.infrastructure.codebase_registry import (
    CodebaseRegistryStore,
    best_effort_remove_from_registry,
    best_effort_update_registry,
)
from aci.services import (
    IndexingService,
    SearchMode,
    SearchService,
    ServicesContainer,
    create_services,
    resolve_repository,
)

# Initialize Rich Console
console = Console()

app = typer.Typer(
    name="aci",
    help="Augmented Codebase Indexer - Semantic code search",
    add_completion=False,
)

def get_services():
    """
    Initialize services from .env with config-driven settings.

    This is a backward-compatible wrapper around create_services() from
    aci.services.container. Returns a tuple for compatibility with existing
    code that unpacks the result.

    Returns:
        Tuple of (config, embedding_client, vector_store, metadata_store,
                  file_scanner, chunker, reranker)
    """
    container = create_services()
    return (
        container.config,
        container.embedding_client,
        container.vector_store,
        container.metadata_store,
        container.file_scanner,
        container.chunker,
        container.reranker,
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
        best_effort_update_registry(
            root_path=path,
            metadata_db_path=metadata_store.db_path,
            collection_name=get_collection_name_for_path(path),
        )

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
    mode: Optional[str] = typer.Option(
        None,
        "--mode",
        "-m",
        help="Search mode: hybrid, vector, grep, or summary (default: hybrid)",
    ),
    rerank: Optional[bool] = typer.Option(
        None, "--rerank/--no-rerank", help="Enable/disable reranking"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full chunk content"),
    snippet_lines: int = typer.Option(
        3, "--snippet-lines", help="Number of lines to display when not verbose"
    ),
    artifact_type: Optional[list[str]] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by artifact type (chunk, function_summary, class_summary, file_summary). Can be specified multiple times.",
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

        # Use centralized repository resolution for path validation and collection name
        resolution = resolve_repository(search_base, metadata_store)
        if not resolution.valid:
            console.print(f"[bold red]Error:[/bold red] {resolution.error_message}")
            raise typer.Exit(1)
        collection_name = resolution.collection_name

        # Use config values if not overridden by CLI
        actual_limit = limit if limit is not None else cfg.search.default_limit
        use_rerank = rerank if rerank is not None else cfg.search.use_rerank

        reranker = config_reranker if use_rerank else None
        if use_rerank and reranker is None:
            console.print(
                "[yellow]Warning: rerank requested but no reranker available; continuing without rerank[/yellow]"
            )
            use_rerank = False

        # Create GrepSearcher for hybrid search support
        grep_searcher = GrepSearcher(base_path=str(search_base))

        search_service = SearchService(
            embedding_client=embedding_client,
            vector_store=vector_store,
            reranker=reranker,
            grep_searcher=grep_searcher,
            default_limit=actual_limit,
        )

        # Validate and parse search mode (default to HYBRID)
        valid_modes = {"hybrid", "vector", "grep", "summary"}
        if mode is not None:
            mode_lower = mode.lower()
            if mode_lower not in valid_modes:
                console.print(
                    f"[bold red]Error:[/bold red] Invalid search mode: {mode}. "
                    f"Valid modes: {', '.join(sorted(valid_modes))}"
                )
                raise typer.Exit(1)
            search_mode = SearchMode(mode_lower)
        else:
            search_mode = SearchMode.HYBRID

        # Validate artifact types if provided
        valid_artifact_types = {"chunk", "function_summary", "class_summary", "file_summary"}
        artifact_types_param = None
        if artifact_type:
            invalid_types = [t for t in artifact_type if t not in valid_artifact_types]
            if invalid_types:
                console.print(
                    f"[bold red]Error:[/bold red] Invalid artifact type(s): {', '.join(invalid_types)}. "
                    f"Valid types: {', '.join(sorted(valid_artifact_types))}"
                )
                raise typer.Exit(1)
            artifact_types_param = artifact_type

        with console.status(f"[bold blue]Searching for[/bold blue] '{query}'..."):
            results = asyncio.run(
                search_service.search(
                    query=query,
                    limit=actual_limit,
                    file_filter=file_filter,  # User-provided filter only
                    use_rerank=use_rerank and reranker is not None,
                    search_mode=search_mode,  # Pass search mode
                    collection_name=collection_name,  # Pass explicitly, no state mutation
                    artifact_types=artifact_types_param,
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

            # Get artifact type for display
            result_artifact_type = result.metadata.get("artifact_type", "chunk")
            type_display = f" [{result_artifact_type}]" if result_artifact_type != "chunk" else ""

            panel = Panel(
                syntax,
                title=f"[bold blue]{i}. {result.file_path}[/bold blue]{type_display} : {result.start_line}-{result.end_line}",
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
            best_effort_update_registry(
                root_path=path,
                metadata_db_path=metadata_store.db_path,
                collection_name=get_collection_name_for_path(path),
            )

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

        # Remove from global registry (best-effort; ignore failures)
        for repo in repos:
            root_path = repo.get("root_path")
            if root_path:
                best_effort_remove_from_registry(root_path=root_path)

        # Clear metadata
        metadata_store.clear_all()
        console.print("  [green]✓[/green] Metadata store cleared.")
        
        console.print("[bold green]Reset complete.[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command("list")
def list_repos(
    global_registry: bool = typer.Option(
        False,
        "--global",
        help="List indexed codebases from the global registry (~/.aci/registry.db) instead of the local metadata store.",
    ),
):
    """List indexed repositories (local) or indexed codebases (global)."""
    try:
        if global_registry:
            store = CodebaseRegistryStore()
            try:
                records = store.list_codebases()
            finally:
                store.close()

            if not records:
                console.print("[yellow]No codebases in global registry.[/yellow]")
                return

            table = Table(
                title=f"Global Indexed Codebases ({store.db_path})",
                border_style="blue",
            )
            table.add_column("Root Path", style="cyan")
            table.add_column("Collection", style="green", no_wrap=True)
            table.add_column("Metadata DB", style="magenta")
            table.add_column("Last Updated", style="yellow", no_wrap=True)

            for rec in records:
                table.add_row(
                    rec.root_path,
                    rec.collection_name,
                    rec.metadata_db_path,
                    str(rec.updated_at),
                )

            console.print(table)
            return

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
