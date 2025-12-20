"""
Summary Artifact module for multi-granularity indexing.

Provides data models for representing summary artifacts (function summaries,
class summaries, file summaries) that complement code chunks for semantic search.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ArtifactType(str, Enum):
    """
    Types of artifacts that can be indexed.

    Inherits from str to enable JSON serialization and string comparison.
    """
    CHUNK = "chunk"
    FUNCTION_SUMMARY = "function_summary"
    CLASS_SUMMARY = "class_summary"
    FILE_SUMMARY = "file_summary"


@dataclass
class SummaryArtifact:
    """
    Represents a summary artifact for indexing.

    Summary artifacts provide higher-level semantic descriptions of code
    structures (functions, classes, files) to complement fine-grained
    code chunks in search results.

    Attributes:
        artifact_id: Unique identifier (UUID)
        file_path: Path to the source file
        artifact_type: Type of artifact (function_summary, class_summary, file_summary)
        name: Name of the summarized entity (function/class/file name)
        content: Generated summary text for embedding
        start_line: Start line in source (0 for file summary)
        end_line: End line in source (0 for file summary)
        metadata: Type-specific metadata (params, return_type, base_classes, etc.)
    """

    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = ""
    artifact_type: ArtifactType = ArtifactType.CHUNK
    name: str = ""
    content: str = ""
    start_line: int = 0
    end_line: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the artifact to a dictionary suitable for JSON serialization.

        Returns:
            Dictionary representation with artifact_type as string value.
        """
        result = asdict(self)
        # Convert ArtifactType enum to its string value
        result["artifact_type"] = self.artifact_type.value
        return result

    def to_json(self) -> str:
        """
        Serialize the artifact to a JSON string.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SummaryArtifact":
        """
        Create a SummaryArtifact from a dictionary.

        Args:
            data: Dictionary containing artifact fields.

        Returns:
            SummaryArtifact instance.
        """
        # Handle artifact_type conversion from string to enum
        artifact_type_value = data.get("artifact_type", ArtifactType.CHUNK.value)
        if isinstance(artifact_type_value, str):
            artifact_type = ArtifactType(artifact_type_value)
        elif isinstance(artifact_type_value, ArtifactType):
            artifact_type = artifact_type_value
        else:
            artifact_type = ArtifactType.CHUNK

        return cls(
            artifact_id=data.get("artifact_id", str(uuid.uuid4())),
            file_path=data.get("file_path", ""),
            artifact_type=artifact_type,
            name=data.get("name", ""),
            content=data.get("content", ""),
            start_line=data.get("start_line", 0),
            end_line=data.get("end_line", 0),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SummaryArtifact":
        """
        Create a SummaryArtifact from a JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            SummaryArtifact instance.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
