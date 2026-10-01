"""
ChakrView Step 61: Deterministic Repository Change Detection & Classification.

Classifies modifications between two RepositoryState snapshots into cognitive impact categories:
- COSMETIC: Whitespace, comments, formatting with zero AST signature changes
- LOCAL: Function bodies changed without altering public signatures, imports, or dependencies
- DEPENDENCY: Import changes or modified external module interactions
- BEHAVIORAL: Signature or call additions/removals impacting callers
- ARCHITECTURAL: New or removed files, modules, classes altering topology
- TEST_ONLY: Pure modifications to test suites without affecting source modules
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from chakrview.cognition.repository.state import RepositoryState, FileState


class ChangeCategory(Enum):
    COSMETIC = auto()
    LOCAL = auto()
    DEPENDENCY = auto()
    BEHAVIORAL = auto()
    ARCHITECTURAL = auto()
    TEST_ONLY = auto()


@dataclass
class FileChange:
    """Detailed change breakdown for a single repository file."""
    rel_path: str
    change_type: str  # "ADDED", "REMOVED", "MODIFIED"
    category: ChangeCategory
    is_test: bool
    old_sha256: Optional[str] = None
    new_sha256: Optional[str] = None
    added_functions: List[str] = field(default_factory=list)
    removed_functions: List[str] = field(default_factory=list)
    added_classes: List[str] = field(default_factory=list)
    removed_classes: List[str] = field(default_factory=list)
    added_imports: List[str] = field(default_factory=list)
    removed_imports: List[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.name
        return d


@dataclass
class RepositoryDiff:
    """
    Complete, explainable difference between two repository states.
    """
    from_fingerprint: str
    to_fingerprint: str
    added_files: List[str] = field(default_factory=list)
    removed_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    file_changes: Dict[str, FileChange] = field(default_factory=dict)
    highest_category: ChangeCategory = ChangeCategory.COSMETIC
    has_architectural_changes: bool = False
    has_dependency_changes: bool = False
    affected_modules: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_fingerprint": self.from_fingerprint,
            "to_fingerprint": self.to_fingerprint,
            "added_files": sorted(self.added_files),
            "removed_files": sorted(self.removed_files),
            "modified_files": sorted(self.modified_files),
            "file_changes": {p: fc.to_dict() for p, fc in sorted(self.file_changes.items())},
            "highest_category": self.highest_category.name,
            "has_architectural_changes": self.has_architectural_changes,
            "has_dependency_changes": self.has_dependency_changes,
            "affected_modules": sorted(list(self.affected_modules)),
        }


class RepositoryChangeDetector:
    """
    Deterministic change detector comparing before and after repository states.
    """

    @classmethod
    def detect_changes(cls, before: RepositoryState, after: RepositoryState) -> RepositoryDiff:
        """Analyze changes between two repository state snapshots."""
        before_paths = set(before.files.keys())
        after_paths = set(after.files.keys())

        added_paths = sorted(list(after_paths - before_paths))
        removed_paths = sorted(list(before_paths - after_paths))
        common_paths = sorted(list(before_paths & after_paths))

        modified_paths: List[str] = []
        file_changes: Dict[str, FileChange] = {}
        affected_modules: Set[str] = set()

        # Added files
        for p in added_paths:
            f_after = after.files[p]
            is_test = f_after.is_test
            cat = ChangeCategory.TEST_ONLY if is_test else ChangeCategory.ARCHITECTURAL
            fc = FileChange(
                rel_path=p,
                change_type="ADDED",
                category=cat,
                is_test=is_test,
                old_sha256=None,
                new_sha256=f_after.sha256,
                added_functions=f_after.module_inspection.functions if f_after.module_inspection else [],
                added_classes=f_after.module_inspection.classes if f_after.module_inspection else [],
                added_imports=f_after.module_inspection.imports if f_after.module_inspection else [],
                rationale="New file added to repository",
            )
            file_changes[p] = fc
            affected_modules.add(p)

        # Removed files
        for p in removed_paths:
            f_before = before.files[p]
            is_test = f_before.is_test
            cat = ChangeCategory.TEST_ONLY if is_test else ChangeCategory.ARCHITECTURAL
            fc = FileChange(
                rel_path=p,
                change_type="REMOVED",
                category=cat,
                is_test=is_test,
                old_sha256=f_before.sha256,
                new_sha256=None,
                removed_functions=f_before.module_inspection.functions if f_before.module_inspection else [],
                removed_classes=f_before.module_inspection.classes if f_before.module_inspection else [],
                removed_imports=f_before.module_inspection.imports if f_before.module_inspection else [],
                rationale="File deleted from repository",
            )
            file_changes[p] = fc
            affected_modules.add(p)

        # Common files: detect modifications and classify
        for p in common_paths:
            f_before = before.files[p]
            f_after = after.files[p]

            if f_before.sha256 == f_after.sha256:
                continue  # Content identical

            modified_paths.append(p)
            affected_modules.add(p)

            insp_b = f_before.module_inspection
            insp_a = f_after.module_inspection

            is_test = f_after.is_test or f_before.is_test

            added_fn = sorted(list(set(insp_a.functions if insp_a else []) - set(insp_b.functions if insp_b else [])))
            removed_fn = sorted(list(set(insp_b.functions if insp_b else []) - set(insp_a.functions if insp_a else [])))
            added_cls = sorted(list(set(insp_a.classes if insp_a else []) - set(insp_b.classes if insp_b else [])))
            removed_cls = sorted(list(set(insp_b.classes if insp_b else []) - set(insp_a.classes if insp_a else [])))
            added_imp = sorted(list(set(insp_a.imports if insp_a else []) - set(insp_b.imports if insp_b else [])))
            removed_imp = sorted(list(set(insp_b.imports if insp_b else []) - set(insp_a.imports if insp_a else [])))

            # Classification logic
            if is_test:
                cat = ChangeCategory.TEST_ONLY
                rat = "Test code modified"
            elif added_imp or removed_imp:
                cat = ChangeCategory.DEPENDENCY
                rat = f"Import changes detected: +{added_imp} -{removed_imp}"
            elif added_fn or removed_fn or added_cls or removed_cls:
                cat = ChangeCategory.BEHAVIORAL
                rat = f"Public signatures changed: +fn:{added_fn} -fn:{removed_fn} +cls:{added_cls} -cls:{removed_cls}"
            elif insp_b and insp_a and (
                insp_b.functions == insp_a.functions and
                insp_b.classes == insp_a.classes and
                insp_b.imports == insp_a.imports
            ):
                # Functions, classes, and imports identical
                if insp_b.ast_hash and insp_a.ast_hash and insp_b.ast_hash == insp_a.ast_hash:
                    cat = ChangeCategory.COSMETIC
                    rat = "Whitespace, comments, or format changes with identical AST representation"
                else:
                    cat = ChangeCategory.LOCAL
                    rat = "Internal function body logic modified without changing function signatures or imports"
            else:
                cat = ChangeCategory.LOCAL
                rat = "Local logic modifications within existing module scope"

            file_changes[p] = FileChange(
                rel_path=p,
                change_type="MODIFIED",
                category=cat,
                is_test=is_test,
                old_sha256=f_before.sha256,
                new_sha256=f_after.sha256,
                added_functions=added_fn,
                removed_functions=removed_fn,
                added_classes=added_cls,
                removed_classes=removed_cls,
                added_imports=added_imp,
                removed_imports=removed_imp,
                rationale=rat,
            )

        # Compute highest category
        severity_order = [
            ChangeCategory.COSMETIC,
            ChangeCategory.TEST_ONLY,
            ChangeCategory.LOCAL,
            ChangeCategory.DEPENDENCY,
            ChangeCategory.BEHAVIORAL,
            ChangeCategory.ARCHITECTURAL,
        ]
        highest = ChangeCategory.COSMETIC
        has_arch = bool(added_paths or removed_paths)
        has_dep = False

        for fc in file_changes.values():
            if severity_order.index(fc.category) > severity_order.index(highest):
                highest = fc.category
            if fc.category == ChangeCategory.ARCHITECTURAL:
                has_arch = True
            if fc.category == ChangeCategory.DEPENDENCY:
                has_dep = True

        return RepositoryDiff(
            from_fingerprint=before.state_fingerprint,
            to_fingerprint=after.state_fingerprint,
            added_files=added_paths,
            removed_files=removed_paths,
            modified_files=modified_paths,
            file_changes=file_changes,
            highest_category=highest,
            has_architectural_changes=has_arch,
            has_dependency_changes=has_dep,
            affected_modules=affected_modules,
        )
