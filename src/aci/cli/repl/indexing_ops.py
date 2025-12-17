"""
Indexing operations for the ACI REPL.

Provides index, update, and reset command implementations with
progress display in both compact and verbose modes.
"""

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from prompt_toolkit.shortcuts import confirm
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from aci.cli.ui import render_error
from aci.core.path_utils import get_collection_name_for_path, validate_indexable_path
from aci.infrastructure.codebase_registry import (
    best_effort_remove_from_registry,
    best_effort_update_registry,
)

if TYPE_CHECKING:
    from aci.cli.repl.event_loop import EventLoopManager
    from aci.cli.services import ServicesContainer
    from aci.services.indexing_service import IndexResult, UpdateResult


class IndexingOperations:
    """
    Handles indexing-related REPL commands.
    
    Provides index, update, and reset operations with both
    compact and verbose progress display modes.
    """

    def __init__(
        self,
        services: "ServicesContainer",
        console: Console,
        verbose: bool = False,
        event_loop_manager: Optional["EventLoopManager"] = None,
    ):
        """
        Initialize indexing operations.
        
        Args:
            services: Services container with initialized services.
            console: Rich console for output.
            verbose: Whether to use verbose output mode.
            event_loop_manager: Optional event loop manager for async operations.
        """
        self.services = services
        self.console = console
        self._verbose = verbose
        self._event_loop_manager = event_loop_manager

    @property
    def verbose(self) -> bool:
        """Get verbose mode setting."""
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """Set verbose mode setting."""
        self._verbose = value

    def run_index(self, path: str) -> None:
        """
        Run the index command with path validation.
        
        Args:
            path: Directory path to index.
        """
        # Validate path before indexing
        validation = validate_indexable_path(path)
        if not validation.valid:
            render_error(validation.error_message or "Invalid path", self.console)
            return

        self.console.print(f"[bold blue]Indexing[/bold blue] {path}...")

        try:
            if self._verbose:
                idx_result = self._run_index_verbose(Path(path))
            else:
                idx_result = self._run_index_compact(Path(path))

            self.console.print(
                f"[green]✓[/green] Indexed {idx_result.total_files} files, "
                f"{idx_result.total_chunks} chunks in {idx_result.duration_seconds:.2f}s"
            )
            try:
                best_effort_update_registry(
                    root_path=Path(path),
                    metadata_db_path=getattr(self.services.metadata_store, "db_path", ".aci/index.db"),
                    collection_name=get_collection_name_for_path(Path(path)),
                )
            except Exception:
                pass
        except Exception as e:
            render_error(str(e), self.console)

    def _run_index_compact(self, path: Path) -> "IndexResult":
        """
        Run index with compact output (2 lines max).
        
        Args:
            path: Directory path to index.
            
        Returns:
            IndexResult from the indexing service.
        """
        from aci.services import IndexingService

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            transient=True,  # Clear after completion
        ) as progress:
            task = progress.add_task("Initializing...", total=None)

            def update_progress(current: int, total: int, message: str) -> None:
                progress.update(task, completed=current, total=total, description=message)

            indexing_service = IndexingService(
                embedding_client=self.services.embedding_client,
                vector_store=self.services.vector_store,
                metadata_store=self.services.metadata_store,
                file_scanner=self.services.file_scanner,
                chunker=self.services.chunker,
                max_workers=self.services.config.indexing.max_workers,
                batch_size=self.services.config.embedding.batch_size,
                progress_callback=update_progress,
            )

            coro = indexing_service.index_directory(path)
            if self._event_loop_manager:
                return self._event_loop_manager.run_async(coro)
            else:
                import asyncio
                return asyncio.run(coro)

    def _run_index_verbose(self, path: Path) -> "IndexResult":
        """
        Run index with verbose output (multi-phase progress + log window).
        
        Args:
            path: Directory path to index.
            
        Returns:
            IndexResult from the indexing service.
        """
        from aci.services import IndexingService

        # Log buffer for scrolling log window
        log_lines: deque = deque(maxlen=5)
        current_phase = [None]
        tasks = {}

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )

        def make_display():
            """Create the combined display with progress and logs."""
            table = Table.grid(expand=True)
            table.add_row(progress)
            if log_lines:
                log_text = "\n".join(log_lines)
                table.add_row(Panel(log_text, title="[dim]Log[/dim]", border_style="dim"))
            return table

        def update_progress(current: int, total: int, message: str) -> None:
            phase = message.rstrip(".")
            if phase != current_phase[0]:
                # New phase - mark previous complete
                if current_phase[0] and current_phase[0] in tasks:
                    prev_task = tasks[current_phase[0]]
                    task_obj = progress._tasks.get(prev_task)
                    if task_obj and task_obj.total:
                        progress.update(prev_task, completed=task_obj.total)
                    log_lines.append(f"[green]✓[/green] {current_phase[0]} complete")
                current_phase[0] = phase
                tasks[phase] = progress.add_task(message, total=total if total > 0 else None)
            else:
                if phase in tasks:
                    progress.update(
                        tasks[phase],
                        completed=current,
                        total=total if total > 0 else None,
                        description=message,
                    )
            live.update(make_display())

        indexing_service = IndexingService(
            embedding_client=self.services.embedding_client,
            vector_store=self.services.vector_store,
            metadata_store=self.services.metadata_store,
            file_scanner=self.services.file_scanner,
            chunker=self.services.chunker,
            max_workers=self.services.config.indexing.max_workers,
            batch_size=self.services.config.embedding.batch_size,
            progress_callback=update_progress,
        )

        with Live(make_display(), console=self.console, refresh_per_second=10) as live:
            coro = indexing_service.index_directory(path)
            if self._event_loop_manager:
                result = self._event_loop_manager.run_async(coro)
            else:
                import asyncio
                result = asyncio.run(coro)
            # Mark final phase complete
            if current_phase[0] and current_phase[0] in tasks:
                task_obj = progress._tasks.get(tasks[current_phase[0]])
                if task_obj and task_obj.total:
                    progress.update(tasks[current_phase[0]], completed=task_obj.total)
                log_lines.append(f"[green]✓[/green] {current_phase[0]} complete")
            live.update(make_display())

        return result


    def run_update(self, path: str) -> None:
        """
        Run the update command with path validation.
        
        Args:
            path: Directory path to update.
        """
        # Validate path before updating
        validation = validate_indexable_path(path)
        if not validation.valid:
            render_error(validation.error_message or "Invalid path", self.console)
            return

        self.console.print(f"[bold blue]Updating index for[/bold blue] {path}...")

        try:
            if self._verbose:
                upd_result = self._run_update_verbose(Path(path))
            else:
                upd_result = self._run_update_compact(Path(path))

            self.console.print(
                f"[green]✓[/green] Updated: {upd_result.new_files} new, "
                f"{upd_result.modified_files} modified, "
                f"{upd_result.deleted_files} deleted in {upd_result.duration_seconds:.2f}s"
            )
            try:
                best_effort_update_registry(
                    root_path=Path(path),
                    metadata_db_path=getattr(self.services.metadata_store, "db_path", ".aci/index.db"),
                    collection_name=get_collection_name_for_path(Path(path)),
                )
            except Exception:
                pass
        except Exception as e:
            render_error(str(e), self.console)

    def _run_update_compact(self, path: Path) -> "UpdateResult":
        """
        Run update with compact output.
        
        Args:
            path: Directory path to update.
            
        Returns:
            UpdateResult from the indexing service.
        """
        from aci.services import IndexingService

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Scanning...", total=None)

            def update_progress(current: int, total: int, message: str) -> None:
                progress.update(task, completed=current, total=total, description=message)

            indexing_service = IndexingService(
                embedding_client=self.services.embedding_client,
                vector_store=self.services.vector_store,
                metadata_store=self.services.metadata_store,
                file_scanner=self.services.file_scanner,
                chunker=self.services.chunker,
                max_workers=self.services.config.indexing.max_workers,
                batch_size=self.services.config.embedding.batch_size,
                progress_callback=update_progress,
            )

            coro = indexing_service.update_incremental(path)
            if self._event_loop_manager:
                return self._event_loop_manager.run_async(coro)
            else:
                import asyncio
                return asyncio.run(coro)

    def _run_update_verbose(self, path: Path) -> "UpdateResult":
        """
        Run update with verbose output.
        
        Args:
            path: Directory path to update.
            
        Returns:
            UpdateResult from the indexing service.
        """
        from aci.services import IndexingService

        log_lines: deque = deque(maxlen=5)
        current_phase = [None]
        tasks = {}

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        )

        def make_display():
            table = Table.grid(expand=True)
            table.add_row(progress)
            if log_lines:
                log_text = "\n".join(log_lines)
                table.add_row(Panel(log_text, title="[dim]Log[/dim]", border_style="dim"))
            return table

        def update_progress(current: int, total: int, message: str) -> None:
            phase = message.rstrip(".")
            if phase != current_phase[0]:
                if current_phase[0] and current_phase[0] in tasks:
                    prev_task = tasks[current_phase[0]]
                    task_obj = progress._tasks.get(prev_task)
                    if task_obj and task_obj.total:
                        progress.update(prev_task, completed=task_obj.total)
                    log_lines.append(f"[green]✓[/green] {current_phase[0]} complete")
                current_phase[0] = phase
                tasks[phase] = progress.add_task(message, total=total if total > 0 else None)
            else:
                if phase in tasks:
                    progress.update(
                        tasks[phase],
                        completed=current,
                        total=total if total > 0 else None,
                        description=message,
                    )
            live.update(make_display())

        indexing_service = IndexingService(
            embedding_client=self.services.embedding_client,
            vector_store=self.services.vector_store,
            metadata_store=self.services.metadata_store,
            file_scanner=self.services.file_scanner,
            chunker=self.services.chunker,
            max_workers=self.services.config.indexing.max_workers,
            batch_size=self.services.config.embedding.batch_size,
            progress_callback=update_progress,
        )

        with Live(make_display(), console=self.console, refresh_per_second=10) as live:
            coro = indexing_service.update_incremental(path)
            if self._event_loop_manager:
                result = self._event_loop_manager.run_async(coro)
            else:
                import asyncio
                result = asyncio.run(coro)
            if current_phase[0] and current_phase[0] in tasks:
                task_obj = progress._tasks.get(tasks[current_phase[0]])
                if task_obj and task_obj.total:
                    progress.update(tasks[current_phase[0]], completed=task_obj.total)
                log_lines.append(f"[green]✓[/green] {current_phase[0]} complete")
            live.update(make_display())

        return result

    def run_reset(self) -> None:
        """Run the reset command with confirmation."""
        try:
            if not confirm(
                "Are you sure you want to reset the index? This will delete all data."
            ):
                self.console.print("[yellow]Reset cancelled.[/yellow]")
                return
        except (KeyboardInterrupt, EOFError):
            self.console.print("\n[yellow]Reset cancelled.[/yellow]")
            return

        self.console.print("[bold yellow]Resetting index (all collections)...[/bold yellow]")

        try:
            # Get all registered repositories and their collections
            repos = self.services.metadata_store.get_repositories()
            deleted_collections = []
            failed_collections = []
            
            # Delete each repository's collection
            for repo in repos:
                collection_name = repo.get("collection_name")
                if collection_name:
                    try:
                        coro = self.services.vector_store.delete_collection(collection_name)
                        if self._event_loop_manager:
                            deleted = self._event_loop_manager.run_async(coro)
                        else:
                            import asyncio
                            deleted = asyncio.run(coro)
                        if deleted:
                            deleted_collections.append(collection_name)
                    except Exception as e:
                        failed_collections.append((collection_name, str(e)))
            
            # Also delete the default collection if it exists and wasn't already deleted
            default_collection = self.services.config.vector_store.collection_name
            if default_collection not in deleted_collections:
                try:
                    coro = self.services.vector_store.delete_collection(default_collection)
                    if self._event_loop_manager:
                        deleted = self._event_loop_manager.run_async(coro)
                    else:
                        import asyncio
                        deleted = asyncio.run(coro)
                    if deleted:
                        deleted_collections.append(default_collection)
                except Exception as e:
                    failed_collections.append((default_collection, str(e)))
            
            if deleted_collections:
                self.console.print(f"  [green]✓[/green] Deleted {len(deleted_collections)} collection(s).")
            else:
                self.console.print("  [yellow]![/yellow] No collections found to delete.")
            
            if failed_collections:
                self.console.print(f"  [yellow]![/yellow] Failed to delete {len(failed_collections)} collection(s):")
                for name, error in failed_collections:
                    self.console.print(f"      - {name}: {error}")

            # Remove from global registry (best-effort; ignore failures)
            for repo in repos:
                root_path = repo.get("root_path")
                if root_path:
                    best_effort_remove_from_registry(root_path=root_path)

            # Clear metadata
            self.services.metadata_store.clear_all()
            self.console.print("  [green]✓[/green] Metadata store cleared.")

            self.console.print("[bold green]Reset complete.[/bold green]")
        except Exception as e:
            render_error(str(e), self.console)
