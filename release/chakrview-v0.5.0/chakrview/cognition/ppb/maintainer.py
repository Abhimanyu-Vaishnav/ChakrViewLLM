"""
ChakrView Step 81: Change-Aware Persistent Brain Maintenance & Revalidation.

Connects PPB to the repository change lifecycle:
- Detects modified, added, and deleted files via RepositoryDiff / ImpactReport.
- Identifies affected symbols and dependent knowledge.
- Marks affected knowledge STALE without invalidating unrelated project modules.
- Preserves historical knowledge records with full audit trail (version increment, supersedes).
- Re-analyzes only affected regions (targeted refresh).
- Persists task history and learning records (solution, root cause, verification).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from chakrview.cognition.repository.change_detector import RepositoryDiff, FileChange, ChangeCategory
from chakrview.cognition.repository.impact_analyzer import ImpactReport
from chakrview.cognition.repository.inspector import RepositoryInspector, ModuleInspection
from chakrview.cognition.ppb.brain import PersistentProjectBrain
from chakrview.cognition.ppb.models import (
    EpistemicStatus,
    KnowledgeRecord,
    KnowledgeRecordType,
)


@dataclass
class MaintenanceReport:
    """Audit report for brain maintenance after repository modifications."""
    modified_files: List[str]
    affected_dependents: List[str]
    records_marked_stale: int
    records_reverified: int
    new_records_created: int
    unaffected_files_preserved: List[str]
    maintenance_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modified_files": sorted(self.modified_files),
            "affected_dependents": sorted(self.affected_dependents),
            "records_marked_stale": self.records_marked_stale,
            "records_reverified": self.records_reverified,
            "new_records_created": self.new_records_created,
            "unaffected_files_preserved": sorted(self.unaffected_files_preserved),
            "maintenance_time_ms": round(self.maintenance_time_ms, 2),
        }


class ChangeAwareBrainMaintainer:
    """
    Maintains PersistentProjectBrain freshness across code modifications and patches.
    """

    def __init__(self, brain: PersistentProjectBrain) -> None:
        self.brain = brain

    def handle_repository_changes(
        self,
        diff: RepositoryDiff,
        impact_report: Optional[ImpactReport] = None,
        updated_files_content: Optional[Dict[str, str]] = None,
    ) -> MaintenanceReport:
        """
        Process repository changes:
        1. Identify modified and affected files.
        2. Mark existing records for these files as STALE.
        3. If updated content is provided, re-analyze affected files and mark updated records as REVERIFIED.
        4. Leave unrelated files untouched and VALID.
        """
        start_t = time.perf_counter()

        modified_files = sorted(list(diff.file_changes.keys()))
        affected_dependents: Set[str] = set()

        if impact_report:
            deps = getattr(impact_report, "transitive_affected_modules", None) or getattr(impact_report, "transitively_affected_modules", [])
            affected_dependents.update(deps)

        # 1. Invalidate affected files
        files_to_stale = set(modified_files) | affected_dependents
        total_staled = 0

        for f_path in files_to_stale:
            total_staled += self.brain.mark_file_stale(f_path)

        # 2. Re-analyze modified files if content is available
        reverified_count = 0
        new_records_count = 0

        if updated_files_content:
            for rel_path, content in updated_files_content.items():
                if rel_path in files_to_stale:
                    insp = RepositoryInspector.inspect_source(rel_path, content)
                    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    project_id = self.brain.project_id

                    # Update module record
                    mod_id = f"mod_{project_id}_{rel_path.replace('/', '_').replace('.', '_')}"
                    mod_record = KnowledgeRecord(
                        record_id=mod_id,
                        project_id=project_id,
                        record_type=KnowledgeRecordType.MODULE,
                        file_path=rel_path,
                        summary=f"Module {insp.module_name} in {rel_path} (updated)",
                        details={
                            "functions": insp.functions,
                            "classes": insp.classes,
                            "imports": insp.imports,
                            "is_test": insp.is_test,
                            "lines_count": len(content.splitlines()),
                        },
                        dependencies=insp.imports,
                        related_symbols=insp.functions + insp.classes,
                        evidence_ids=[f"ast_{sha[:12]}"],
                        epistemic_status=EpistemicStatus.REVERIFIED,
                        confidence=1.0,
                        repo_fingerprint=sha[:16],
                        source_chunk="reanalysis",
                    )
                    self.brain.store_knowledge(mod_record)
                    reverified_count += 1

                    # Update or insert symbols
                    for sym in insp.functions + insp.classes:
                        sym_id = f"sym_{project_id}_{rel_path.replace('/', '_')}_{sym}"
                        sym_record = KnowledgeRecord(
                            record_id=sym_id,
                            project_id=project_id,
                            record_type=KnowledgeRecordType.SYMBOL,
                            file_path=rel_path,
                            symbol_name=sym,
                            summary=f"Symbol {sym} in {rel_path} (updated)",
                            details={"module": insp.module_name},
                            dependencies=insp.imports,
                            related_symbols=[],
                            evidence_ids=[f"ast_{sha[:12]}"],
                            epistemic_status=EpistemicStatus.REVERIFIED,
                            confidence=1.0,
                            repo_fingerprint=sha[:16],
                            source_chunk="reanalysis",
                        )
                        self.brain.store_knowledge(sym_record)
                        reverified_count += 1

                    # Update scan progress
                    self.brain.storage.record_scan_progress(
                        project_id=project_id,
                        file_path=rel_path,
                        content_sha256=sha,
                        source_chunk="maintenance",
                        status="COMPLETED",
                    )

        # Unaffected files
        all_active_files = set(self.brain.storage.get_all_active_files(self.brain.project_id))
        unaffected = sorted(list(all_active_files - files_to_stale))

        elapsed = (time.perf_counter() - start_t) * 1000.0

        return MaintenanceReport(
            modified_files=modified_files,
            affected_dependents=sorted(list(affected_dependents)),
            records_marked_stale=total_staled,
            records_reverified=reverified_count,
            new_records_created=new_records_count,
            unaffected_files_preserved=unaffected,
            maintenance_time_ms=elapsed,
        )

    def record_completed_task(
        self,
        task_id: str,
        task_description: str,
        affected_files: List[str],
        solution_summary: str,
        verification_passed: bool,
        root_cause: str = "",
        rejected_alternatives: Optional[List[str]] = None,
    ) -> KnowledgeRecord:
        """
        Record durable task learning in PersistentProjectBrain.
        Stores verified facts only if verification passed; otherwise marks as UNCERTAIN or HYPOTHESIS.
        """
        project_id = self.brain.project_id
        record_id = f"task_{project_id}_{task_id}"

        status = EpistemicStatus.FACT if verification_passed else EpistemicStatus.INSUFFICIENT

        task_record = KnowledgeRecord(
            record_id=record_id,
            project_id=project_id,
            record_type=KnowledgeRecordType.TASK_HISTORY,
            file_path=affected_files[0] if affected_files else "project",
            summary=f"Completed Task: {task_description}",
            details={
                "task_id": task_id,
                "solution_summary": solution_summary,
                "verification_passed": verification_passed,
                "root_cause": root_cause,
                "rejected_alternatives": rejected_alternatives or [],
                "affected_files": affected_files,
            },
            dependencies=[],
            related_symbols=[],
            evidence_ids=[f"task_eval_{task_id}"],
            epistemic_status=status,
            confidence=1.0 if verification_passed else 0.5,
            source_chunk="task_completion",
        )
        self.brain.store_knowledge(task_record)
        return task_record
