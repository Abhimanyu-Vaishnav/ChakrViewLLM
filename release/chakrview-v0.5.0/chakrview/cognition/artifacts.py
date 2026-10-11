"""
Artifact & Document Workflow Foundation for ChakrView (Step 15).

Provides generic representations for artifacts produced by cognitive tasks:
text, structured JSON, reports, markdown, spreadsheets, or code.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, List, Optional, Any
import uuid


class ArtifactType(str, Enum):
    """Supported artifact formats."""
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    REPORT = "report"
    CSV = "csv"
    CODE = "code"


@dataclass
class CognitiveArtifact:
    """
    Structured outcome artifact produced by a cognitive step or task.

    Attributes:
        artifact_id: Unique artifact identifier.
        name: User-facing or file name for the artifact.
        artifact_type: Artifact category/format.
        content: The text or serialized string content of the artifact.
        metadata: Domain, provenance, and generation metadata.
        created_at: Epoch timestamp of creation.
    """
    artifact_id: str
    name: str
    artifact_type: ArtifactType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "name": self.name,
            "artifact_type": self.artifact_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitiveArtifact":
        d = dict(data)
        d["artifact_type"] = ArtifactType(d["artifact_type"])
        return cls(**d)


class ArtifactManager:
    """
    In-memory registry and store for cognitive artifacts generated during execution.
    """

    def __init__(self) -> None:
        self._artifacts: Dict[str, CognitiveArtifact] = {}

    def create_artifact(
        self,
        name: str,
        content: str,
        artifact_type: ArtifactType = ArtifactType.TEXT,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CognitiveArtifact:
        """Create and store a new CognitiveArtifact."""
        artifact_id = f"art_{uuid.uuid4().hex[:8]}"
        art = CognitiveArtifact(
            artifact_id=artifact_id,
            name=name,
            artifact_type=artifact_type,
            content=content,
            metadata=metadata or {},
        )
        self._artifacts[artifact_id] = art
        return art

    def get(self, artifact_id: str) -> Optional[CognitiveArtifact]:
        """Look up an artifact by ID."""
        return self._artifacts.get(artifact_id)

    def list_artifacts(self) -> List[CognitiveArtifact]:
        """List all managed artifacts."""
        return list(self._artifacts.values())

    def clear(self) -> None:
        """Clear all artifacts."""
        self._artifacts.clear()
