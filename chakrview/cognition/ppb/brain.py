"""
ChakrView Step 78: First-Class Persistent Project Brain (PPB) Core Adapter.

Unifies:
- Project identity and persistent storage lifecycle
- Epistemic integrity boundaries (Project Knowledge vs Episodic Experience vs External Research vs Task Context)
- Integration with RepositoryContextStore, RepositoryMemoryIndex, and Hardware/Resource detection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.runtime.resource import (
    HardwareCapability,
    ResourceDetector,
    ResourcePolicy,
    RuntimeStrategy,
)
from chakrview.cognition.ppb.models import (
    EpistemicStatus,
    KnowledgeRecord,
    KnowledgeRecordType,
    ProjectBrainState,
    ProjectIdentity,
)
from chakrview.cognition.ppb.storage import PersistentBrainStorage


class PersistentProjectBrain:
    """
    First-class Persistent Project Brain for ChakrView.
    Maintains durable, cross-session project knowledge with strict provenance and epistemic tracking.
    """

    def __init__(
        self,
        db_path: str,
        project_identity: Optional[ProjectIdentity] = None,
        hardware_capability: Optional[HardwareCapability] = None,
    ) -> None:
        self.db_path = str(Path(db_path).resolve())
        self.storage = PersistentBrainStorage(self.db_path, project_identity)
        self.hardware_capability = hardware_capability or ResourceDetector.detect()
        self.resource_policy = ResourceDetector.determine_strategy(self.hardware_capability)

        if project_identity is not None:
            self._project_identity = project_identity
        else:
            loaded = self.storage.get_project_identity()
            self._project_identity = loaded or ProjectIdentity(
                project_id="unknown_project",
                project_root=str(Path(self.db_path).parent),
            )

    @property
    def project_id(self) -> str:
        return self._project_identity.project_id

    @property
    def project_identity(self) -> ProjectIdentity:
        return self._project_identity

    def get_state(self) -> ProjectBrainState:
        """Return current state summary from persistent storage."""
        return self.storage.get_state_summary(self.project_id)

    def store_knowledge(self, record: KnowledgeRecord) -> None:
        """Store or update a verified knowledge record in persistent memory."""
        self.storage.insert_record(record)

    def get_knowledge(self, record_id: str) -> Optional[KnowledgeRecord]:
        """Fetch knowledge record by record_id."""
        return self.storage.get_record(record_id)

    def query_knowledge(
        self,
        *,
        file_path: Optional[str] = None,
        symbol_name: Optional[str] = None,
        record_type: Optional[KnowledgeRecordType] = None,
        epistemic_status: Optional[EpistemicStatus] = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> List[KnowledgeRecord]:
        """Filtered deterministic lookup over persistent knowledge records."""
        return self.storage.query_records(
            project_id=self.project_id,
            file_path=file_path,
            symbol_name=symbol_name,
            record_type=record_type,
            epistemic_status=epistemic_status,
            active_only=active_only,
            limit=limit,
        )

    def mark_file_stale(self, file_path: str) -> int:
        """Mark all knowledge derived from file_path as STALE."""
        return self.storage.update_epistemic_status_by_file(
            file_path=file_path,
            new_status=EpistemicStatus.STALE,
            project_id=self.project_id,
        )

    def mark_record_reverified(self, record_id: str) -> bool:
        """Mark a stale record as REVERIFIED after successful re-analysis."""
        return self.storage.update_record_status(record_id, EpistemicStatus.REVERIFIED)
