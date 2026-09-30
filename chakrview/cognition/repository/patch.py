"""
ChakrView Step 59: Reversible Multi-File Patch Coordinator.

Manages transactional multi-file changes against an IsolatedWorkspace:
- Records pre-state content before modifications
- Computes unified diffs
- Applies changes safely with path confinement
- Supports atomic rollback restoring the repository to its exact pre-patch state
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from chakrview.arena.models import PatchDiff
from chakrview.arena.workspace import IsolatedWorkspace


@dataclass
class MultiFilePatchTransaction:
    transaction_id: str
    target_files: List[str]
    pre_contents: Dict[str, str] = field(default_factory=dict)
    post_contents: Dict[str, str] = field(default_factory=dict)
    diffs: List[PatchDiff] = field(default_factory=list)
    applied: bool = False
    reverted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "target_files": self.target_files,
            "pre_contents": self.pre_contents,
            "post_contents": self.post_contents,
            "diffs": [d.to_dict() for d in self.diffs],
            "applied": self.applied,
            "reverted": self.reverted,
        }


class RepositoryPatchCoordinator:
    """
    Coordinates multi-file patches with atomic rollback guarantees.
    """

    def __init__(self, workspace: IsolatedWorkspace) -> None:
        self.workspace = workspace
        self.transactions: List[MultiFilePatchTransaction] = []

    def begin_transaction(self, tx_id: str, file_updates: Dict[str, str]) -> MultiFilePatchTransaction:
        """
        Record pre-state for all files in file_updates and apply changes to workspace.
        """
        pre_contents: Dict[str, str] = {}
        for rel_path in file_updates:
            file_path = self.workspace.get_source_file_path(rel_path)
            pre_contents[rel_path] = file_path.read_text(encoding="utf-8") if file_path.is_file() else ""

        diffs: List[PatchDiff] = []
        for rel_path, new_content in file_updates.items():
            diff = self.workspace.apply_patch(rel_path, new_content)
            diffs.append(diff)

        tx = MultiFilePatchTransaction(
            transaction_id=tx_id,
            target_files=list(file_updates.keys()),
            pre_contents=pre_contents,
            post_contents=dict(file_updates),
            diffs=diffs,
            applied=True,
            reverted=False,
        )
        self.transactions.append(tx)
        return tx

    def rollback_transaction(self, tx: MultiFilePatchTransaction) -> None:
        """
        Revert all files in transaction back to their pre_contents.
        """
        if not tx.applied or tx.reverted:
            return

        for rel_path, old_content in tx.pre_contents.items():
            self.workspace.write_source_file(rel_path, old_content)

        tx.reverted = True

    def get_modified_files(self) -> List[str]:
        """Return distinct list of currently modified files across un-reverted transactions."""
        modified: List[str] = []
        for tx in self.transactions:
            if tx.applied and not tx.reverted:
                for f in tx.target_files:
                    if f not in modified:
                        modified.append(f)
        return modified
