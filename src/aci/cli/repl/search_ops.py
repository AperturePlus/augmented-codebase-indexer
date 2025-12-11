"""
Search operations for the ACI REPL.

Provides search command implementation with result display
and integration with REPLContext for codebase selection.
"""

from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from aci.cli.ui import render_error

if TYPE_CHECKING:
    from aci.cli.repl.context import REPLContext
    from aci.cli.repl.event_loop import EventLoopManager
    from aci.cli.services import ServicesContainer


class SearchOperations:
    """
    Handles search-related REPL commands.
    
    Provides search operations with result rendering and
    integration with REPLContext for codebase selection.
    """

    def __init__(
        self,
        services: "ServicesContainer",
        console: Console,
        context: "REPLContext",
        event_loop_manager: Optional["EventLoopManager"] = None,
    ):
        """
        Initialize search operations.
        
        Args:
            services: Services container with initialized services.
            console: Rich console for output.
            context: REPL context for codebase selection.
            event_loop_manager: Optional event loop manager for async operations.
        """
        self.services = services
        self.console = console
        self.context = context
        self._event_loop_manager = event_loop_manager

    def run_search(
        self,
        query: str,
        limit: Optional[str] = None,
        artifact_types: Optional[list[str]] = None,
    ) -> None:
        """
        Run the search command.
        
        Uses the codebase from REPLContext if set, otherwise
        defaults to the current working directory.
        
        Args:
            query: Search query string.
            limit: Optional result limit as string.
            artifact_types: Optional list of artifact types to filter by.
        """
        from aci.infrastructure import GrepSearcher
        from aci.services import SearchService

        # Get the codebase path from context
        codebase_path = self.context.get_codebase()
        
        # Check if the path is indexed
        if not self.context.is_path_indexed(codebase_path):
            render_error(
                f"Path '{codebase_path}' has not been indexed. "
                f"Run 'index {codebase_path}' first.",
                self.console,
            )
            return

        actual_limit = int(limit) if limit else self.services.config.search.default_limit

        try:
            # Get collection name for this codebase (pass explicitly to search, no mutation)
            # For backward compatibility, generate collection name if not stored
            from aci.core.path_utils import get_collection_name_for_path

            codebase_abs = str(codebase_path.resolve())
            collection_name = self.services.metadata_store.get_collection_name(codebase_abs)
            if not collection_name:
                # Legacy index without collection_name - generate and update
                collection_name = get_collection_name_for_path(codebase_abs)
                self.services.metadata_store.register_repository(codebase_abs, collection_name)

            grep_searcher = GrepSearcher(base_path=str(codebase_path))

            search_service = SearchService(
                embedding_client=self.services.embedding_client,
                vector_store=self.services.vector_store,
                reranker=self.services.reranker,
                grep_searcher=grep_searcher,
                default_limit=actual_limit,
            )

            with self.console.status(f"[bold blue]Searching for[/bold blue] '{query}'..."):
                search_coro = search_service.search(
                    query=query,
                    limit=actual_limit,
                    use_rerank=self.services.reranker is not None,
                    collection_name=collection_name,  # Pass explicitly, no state mutation
                    artifact_types=artifact_types,
                )
                if self._event_loop_manager:
                    results = self._event_loop_manager.run_async(search_coro)
                else:
                    import asyncio
                    results = asyncio.run(search_coro)

            if not results:
                self.console.print("[yellow]No results found.[/yellow]")
                return

            self._render_results(results)

        except Exception as e:
            render_error(str(e), self.console)

    def _render_results(self, results: list) -> None:
        """
        Render search results with syntax highlighting.
        
        Args:
            results: List of search results to display.
        """
        self.console.print(f"\nFound [bold]{len(results)}[/bold] results:\n")

        for i, res in enumerate(results, 1):
            language = res.metadata.get("language", "text")
            lines = res.content.split("\n")
            code_content = "\n".join(lines[:5])
            if len(lines) > 5:
                code_content += "\n..."

            syntax = Syntax(
                code_content,
                language,
                theme="monokai",
                line_numbers=True,
                start_line=res.start_line,
                word_wrap=True,
            )

            # Get artifact type for display
            artifact_type = res.metadata.get("artifact_type", "chunk")
            type_display = f" [{artifact_type}]" if artifact_type != "chunk" else ""

            panel = Panel(
                syntax,
                title=f"[bold blue]{i}. {res.file_path}[/bold blue]{type_display} : {res.start_line}-{res.end_line}",
                subtitle=f"Score: [yellow]{res.score:.3f}[/yellow]",
                border_style="blue",
                expand=True,
            )
            self.console.print(panel)
            self.console.print()
