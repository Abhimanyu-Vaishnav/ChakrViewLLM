"""
Tracked QA Review Validator and Admission Gate.

Canonical Location: tools/qa_review/qa_review_validator.py

Enforces:
1. Valid JSONL syntax per record in review queues and manifests.
2. Mandatory fields present: candidate_id, question, proposed_answer, source_file,
   source_locator, source_split, provenance_status, semantic_review_status, reviewer_decision.
3. No duplicate candidate_ids.
4. Allowed statuses:
   - semantic_review_status in {"UNREVIEWED", "TRAIN-ELIGIBLE", "REJECT", "NEEDS-REVISION"}
   - reviewer_decision in {"UNREVIEWED", "APPROVE", "REJECT", "REVISE", "TRAIN-ELIGIBLE"}
5. Fail-Closed Admissibility Rules for Training Datasets:
   - Candidate MUST have semantic_review_status == "TRAIN-ELIGIBLE".
   - Candidate MUST have source_split == "train" (validation and test origin strictly forbidden from training).
   - Candidate MUST have provenance_status == "PROVENANCE_VERIFIED".
   - Candidate MUST have reviewer_decision in {"APPROVE", "TRAIN-ELIGIBLE"}.
   - Candidate MUST have non-empty reviewer_notes rationale.
   - Advisory recommendations (e.g. APPROVE or REVISE) NEVER count as founder approval.
   - Candidates marked NEEDS_NEW_SOURCE are strictly blocked.
6. Fail-Closed Advisory Artifact Validation:
   - Label MUST be "ASSISTANT_ADVISORY_NOT_FOUNDER_DECISION".
   - Cross-checks with official queue (1-to-1 candidate ID, matching text, locators, splits).
   - Derived split accounting verification (27 train, 2 val, 1 test).
   - Prevents partial dataset writes on failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


class ReviewQueueValidationError(Exception):
    """Raised when review queue or advisory artifact violates schema, provenance, or split invariants."""
    pass


class AdmissionGateError(ReviewQueueValidationError):
    """Raised when an example fails dataset admission quality gates."""
    pass


class ReviewQueueValidator:
    ALLOWED_SEMANTIC_STATUSES = {"UNREVIEWED", "TRAIN-ELIGIBLE", "REJECT", "NEEDS-REVISION"}
    ALLOWED_REVIEWER_DECISIONS = {"UNREVIEWED", "APPROVE", "REJECT", "REVISE", "TRAIN-ELIGIBLE"}
    ALLOWED_ADVISORY_RECOMMENDATIONS = {"APPROVE", "REVISE", "APPROVE_TEST_ONLY", "NEEDS_NEW_SOURCE"}
    ALLOWED_SPLITS = {"train", "validation", "test"}

    @classmethod
    def validate_file(cls, path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Validates an official review queue file (.jsonl)."""
        if not path.is_file():
            raise FileNotFoundError(f"Review queue file not found: {path}")

        lines = path.read_text(encoding="utf-8").splitlines()
        records: List[Dict[str, Any]] = []
        seen_ids = set()

        stats = {
            "total_records": 0,
            "unreviewed": 0,
            "train_eligible": 0,
            "rejected": 0,
            "needs_revision": 0,
            "source_splits": {},
            "validation_passed": False,
        }

        for idx, line in enumerate(lines, 1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                record = json.loads(line_str)
            except json.JSONDecodeError as e:
                raise ReviewQueueValidationError(f"Malformed JSON at line {idx}: {e}")

            cid = record.get("candidate_id")
            if not cid:
                raise ReviewQueueValidationError(f"Missing candidate_id at line {idx}")
            if cid in seen_ids:
                raise ReviewQueueValidationError(f"Duplicate candidate_id '{cid}' at line {idx}")
            seen_ids.add(cid)

            # Check required fields
            req_fields = [
                "question", "proposed_answer", "source_file", "source_locator",
                "source_split", "provenance_status", "semantic_review_status", "reviewer_decision"
            ]
            for f in req_fields:
                if f not in record:
                    raise ReviewQueueValidationError(f"Record '{cid}' missing required field '{f}'")

            s_status = record["semantic_review_status"]
            if s_status not in cls.ALLOWED_SEMANTIC_STATUSES:
                raise ReviewQueueValidationError(
                    f"Record '{cid}' invalid semantic_review_status: '{s_status}'"
                )

            r_decision = record["reviewer_decision"]
            if r_decision not in cls.ALLOWED_REVIEWER_DECISIONS:
                raise ReviewQueueValidationError(
                    f"Record '{cid}' invalid reviewer_decision: '{r_decision}'"
                )

            sp = record["source_split"]
            if sp not in cls.ALLOWED_SPLITS:
                raise ReviewQueueValidationError(
                    f"Record '{cid}' invalid source_split: '{sp}'"
                )

            # Track stats
            stats["total_records"] += 1
            if s_status == "UNREVIEWED":
                stats["unreviewed"] += 1
            elif s_status == "TRAIN-ELIGIBLE":
                stats["train_eligible"] += 1
            elif s_status == "REJECT":
                stats["rejected"] += 1
            elif s_status == "NEEDS-REVISION":
                stats["needs_revision"] += 1

            stats["source_splits"][sp] = stats["source_splits"].get(sp, 0) + 1

            # Check split admissibility
            if s_status == "TRAIN-ELIGIBLE":
                cls.check_train_admission(record)

            records.append(record)

        stats["validation_passed"] = True
        return records, stats

    @classmethod
    def check_train_admission(cls, record: Dict[str, Any]) -> bool:
        """
        Fail-closed admission check for admitting a record to training.
        Raises AdmissionGateError on any violation.
        """
        cid = record.get("candidate_id", "UNKNOWN")
        sp = record.get("source_split")

        if sp != "train":
            raise AdmissionGateError(
                f"CRITICAL LEAKAGE: Candidate '{cid}' marked for training but has source_split='{sp}'! (Validation/Test permanently excluded)"
            )

        if record.get("provenance_status") != "PROVENANCE_VERIFIED":
            raise AdmissionGateError(
                f"Admission rejected: Candidate '{cid}' lacks PROVENANCE_VERIFIED status."
            )

        s_status = record.get("semantic_review_status")
        if s_status != "TRAIN-ELIGIBLE":
            raise AdmissionGateError(
                f"Admission rejected: Candidate '{cid}' semantic_review_status='{s_status}' (must be TRAIN-ELIGIBLE)."
            )

        r_decision = record.get("reviewer_decision")
        if r_decision not in {"APPROVE", "TRAIN-ELIGIBLE"}:
            raise AdmissionGateError(
                f"Admission rejected: Candidate '{cid}' reviewer_decision='{r_decision}' (must be APPROVE or TRAIN-ELIGIBLE)."
            )

        notes = record.get("reviewer_notes")
        if not notes or not str(notes).strip():
            raise AdmissionGateError(
                f"Candidate '{cid}' marked TRAIN-ELIGIBLE without non-empty reviewer_notes rationale!"
            )

        return True

    @classmethod
    def validate_advisory_artifact(
        cls,
        path: Path,
        queue_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Validates the structure, non-authority labelling, split isolation,
        and integrity of an advisory artifact. Fail-closed on any discrepancy.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Advisory artifact file not found: {path}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ReviewQueueValidationError(f"Malformed JSON in advisory artifact: {e}")

        # Label check
        if data.get("advisory_label") != "ASSISTANT_ADVISORY_NOT_FOUNDER_DECISION":
            raise ReviewQueueValidationError(
                "Advisory artifact missing mandatory 'ASSISTANT_ADVISORY_NOT_FOUNDER_DECISION' label"
            )

        candidates = data.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 30:
            raise ReviewQueueValidationError(
                f"Advisory artifact must contain exactly 30 candidates, found: {len(candidates) if isinstance(candidates, list) else type(candidates)}"
            )

        # Cross-validation with official queue if queue_path provided
        queue_map = None
        if queue_path and queue_path.is_file():
            q_records, _ = cls.validate_file(queue_path)
            queue_map = {r["candidate_id"]: r for r in q_records}
            if len(queue_map) != len(candidates):
                raise ReviewQueueValidationError(
                    f"Candidate count mismatch: advisory has {len(candidates)}, queue has {len(queue_map)}"
                )

        seen_ids = set()
        for c in candidates:
            cid = c.get("candidate_id")
            if not cid:
                raise ReviewQueueValidationError("Candidate missing candidate_id")
            if cid in seen_ids:
                raise ReviewQueueValidationError(f"Duplicate candidate_id in advisory: {cid}")
            seen_ids.add(cid)

            split = c.get("source_split")
            if split not in cls.ALLOWED_SPLITS:
                raise ReviewQueueValidationError(f"Candidate {cid} invalid source_split: '{split}'")

            # Cross-check with official queue
            if queue_map:
                if cid not in queue_map:
                    raise ReviewQueueValidationError(f"Unknown candidate_id '{cid}' not found in official queue")
                q_rec = queue_map[cid]
                for field in ["source_split", "source_file", "source_locator", "source_hash", "source_passage"]:
                    if c.get(field) != q_rec.get(field):
                        raise ReviewQueueValidationError(
                            f"Field mismatch for candidate '{cid}': advisory has {field}='{c.get(field)}', queue has '{q_rec.get(field)}'"
                        )

            rec = c.get("advisory_recommendation")
            if rec not in cls.ALLOWED_ADVISORY_RECOMMENDATIONS:
                raise ReviewQueueValidationError(f"Candidate {cid} invalid recommendation: {rec}")

            # Fail-closed split isolation:
            if split == "test":
                if rec != "APPROVE_TEST_ONLY":
                    raise ReviewQueueValidationError(
                        f"CRITICAL SPLIT VIOLATION: Test-split candidate '{cid}' must be 'APPROVE_TEST_ONLY', got '{rec}'"
                    )
            elif split == "validation":
                if rec == "APPROVE_TEST_ONLY":
                    raise ReviewQueueValidationError(
                        f"CRITICAL SPLIT VIOLATION: Validation-split candidate '{cid}' cannot be 'APPROVE_TEST_ONLY'"
                    )

            # Revision details check
            if rec in {"REVISE", "NEEDS_NEW_SOURCE"}:
                rev = c.get("revision_details") or c.get("proposed_revision")
                if not rev:
                    raise ReviewQueueValidationError(
                        f"Candidate '{cid}' marked {rec} but missing revision details"
                    )

        # Derived split accounting
        train_count = sum(1 for c in candidates if c.get("source_split") == "train")
        val_count = sum(1 for c in candidates if c.get("source_split") == "validation")
        test_count = sum(1 for c in candidates if c.get("source_split") == "test")

        if train_count != 27 or val_count != 2 or test_count != 1:
            raise ReviewQueueValidationError(
                f"Split accounting violation: expected 27 train, 2 val, 1 test; got {train_count} train, {val_count} val, {test_count} test"
            )

        return {
            "total_candidates": len(candidates),
            "train_candidates": train_count,
            "validation_candidates": val_count,
            "test_candidates": test_count,
            "approve_count": sum(1 for c in candidates if c.get("advisory_recommendation") == "APPROVE"),
            "revise_count": sum(1 for c in candidates if c.get("advisory_recommendation") == "REVISE"),
            "needs_new_source_count": sum(1 for c in candidates if c.get("advisory_recommendation") == "NEEDS_NEW_SOURCE"),
            "test_only_count": sum(1 for c in candidates if c.get("advisory_recommendation") == "APPROVE_TEST_ONLY"),
            "validation_passed": True,
        }

    @classmethod
    def compile_training_dataset_atomic(
        cls,
        queue_records: List[Dict[str, Any]],
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Compiles founder-approved candidates into a training manifest atomically.
        If any record fails admission checks, zero files are written (fail-closed).
        """
        admitted = []
        for r in queue_records:
            if r.get("semantic_review_status") == "TRAIN-ELIGIBLE":
                cls.check_train_admission(r)
                admitted.append(r)

        if not admitted:
            raise AdmissionGateError("No candidate records qualified for training admission.")

        # Atomic write
        temp_out = output_path.with_suffix(".tmp")
        temp_out.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_out, "w", encoding="utf-8") as f:
            for item in admitted:
                f.write(json.dumps(item) + "\n")
        temp_out.replace(output_path)

        return {
            "admitted_count": len(admitted),
            "output_path": str(output_path),
            "status": "COMPILED"
        }
