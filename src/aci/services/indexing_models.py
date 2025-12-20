"""
Indexing Service data models.

Contains dataclasses for indexing results and processed files.
"""

from dataclasses import dataclass, field

from aci.core.chunker import CodeChunk
from aci.core.summary_artifact import SummaryArtifact


@dataclass
class IndexingResult:
    """Result of an indexing operation."""

    total_files: int = 0
    total_chunks: int = 0
    new_files: int = 0
    modified_files: int = 0
    deleted_files: int = 0
    failed_files: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class ProcessedFile:
    """Result of processing a single file."""

    file_path: str
    chunks: list[CodeChunk]
    language: str
    line_count: int
    content_hash: str
    summaries: list[SummaryArtifact] = field(default_factory=list)
    error: str | None = None


class IndexingError(Exception):
    """Raised when indexing fails due to embedding count mismatch or other errors."""

    def __init__(
        self,
        message: str,
        batch_index: int | None = None,
        expected: int | None = None,
        actual: int | None = None,
    ):
        self.batch_index = batch_index
        self.expected = expected
        self.actual = actual
        super().__init__(message)
