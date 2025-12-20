"""
Data models for the chunker module.

Contains CodeChunk and ChunkingResult dataclasses.
"""

import uuid
from dataclasses import dataclass, field

from aci.core.summary_artifact import SummaryArtifact


@dataclass
class CodeChunk:
    """
    Represents a chunk of code for indexing.

    Attributes:
        chunk_id: Unique identifier for the chunk
        file_path: Path to the source file
        start_line: Start line number (1-based)
        end_line: End line number (1-based, inclusive)
        content: The actual code content
        language: Programming language identifier
        chunk_type: Type of chunk ('function', 'class', 'method', 'fixed')
        metadata: Additional metadata (function_name, parent_class, imports, file_hash)
    """

    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    content: str = ""
    language: str = ""
    chunk_type: str = ""  # 'function', 'class', 'method', 'fixed'
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkingResult:
    """
    Result of chunking a file.

    Contains both code chunks and summary artifacts generated during
    the chunking process.

    Attributes:
        chunks: List of code chunks extracted from the file
        summaries: List of summary artifacts (function, class, file summaries)
    """

    chunks: list[CodeChunk] = field(default_factory=list)
    summaries: list[SummaryArtifact] = field(default_factory=list)
