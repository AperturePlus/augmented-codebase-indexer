"""
Abstract interfaces for the chunker module.

Contains ChunkerInterface and ImportExtractorInterface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from aci.core.ast_parser import ASTNode
from aci.core.file_scanner import ScannedFile

from .models import ChunkingResult


@dataclass
class ChunkerConfig:
    """Configuration for chunker instances.

    Used to transfer chunker settings across process boundaries
    (e.g., for parallel worker initialization).
    """
    max_tokens: int = 8192
    fixed_chunk_lines: int = 50
    overlap_lines: int = 5


class ChunkerInterface(ABC):
    """Abstract interface for code chunking operations."""

    @abstractmethod
    def chunk(self, file: ScannedFile, ast_nodes: list[ASTNode]) -> ChunkingResult:
        """
        Split a file into code chunks and generate summaries.

        Args:
            file: The scanned file to chunk
            ast_nodes: AST nodes extracted from the file (may be empty)

        Returns:
            ChunkingResult containing chunks and summary artifacts

        Notes:
            - Uses AST nodes for semantic chunking when available
            - Falls back to fixed-size chunking when no AST nodes
            - Splits oversized chunks to fit within token limits
            - Generates function/class/file summaries when SummaryGenerator is available
        """
        pass

    @abstractmethod
    def set_max_tokens(self, max_tokens: int) -> None:
        """
        Set the maximum token count per chunk.

        Args:
            max_tokens: Maximum tokens allowed per chunk
        """
        pass

    @abstractmethod
    def get_config(self) -> ChunkerConfig:
        """
        Get the chunker configuration.

        Returns configuration values needed to recreate a chunker
        with equivalent settings (e.g., for parallel workers).

        Returns:
            ChunkerConfig with current settings
        """
        pass


class ImportExtractorInterface(ABC):
    """Abstract interface for language-specific import extraction."""

    @abstractmethod
    def extract(self, content: str) -> list[str]:
        """
        Extract import statements from code content.

        Args:
            content: Source code content

        Returns:
            List of import statement strings
        """
        pass
