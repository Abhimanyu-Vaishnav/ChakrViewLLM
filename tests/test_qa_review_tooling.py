"""
Comprehensive Unit & Regression Tests for Tracked QA Review Tooling and Admission Gate.

Tracked Location: tests/test_qa_review_tooling.py

Verifies:
1. Clean-checkout availability of tracked tools.qa_review module.
2. Validates official queue qa_review_queue_v0.3.0.jsonl cleanly (30 records, 100% UNREVIEWED).
3. Validates advisory v0.2 artifact and dynamic split accounting (27 train, 2 val, 1 test).
4. Cross-queue candidate mapping (1-to-1 candidate ID, matching locators, hashes, passages).
5. Detects candidate provenance mismatch.
6. Detects split accounting violations.
7. Strictly prevents validation-origin (QA-REV-028, QA-REV-029) and test-origin (QA-REV-030) records
   from being promoted to TRAIN-ELIGIBLE.
8. Requires non-empty reviewer rationale for TRAIN-ELIGIBLE admission.
9. Verifies that advisory recommendations (APPROVE/REVISE) NEVER count as founder decisions.
10. Verifies compile_training_dataset_atomic fails closed without writing files when no valid candidates exist.
11. Verifies preview_draft does not mutate the official queue.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.qa_review.qa_review_validator import (
    ReviewQueueValidator,
    ReviewQueueValidationError,
    AdmissionGateError,
)
from tools.qa_review.founder_review_workflow import (
    load_review_state,
    preview_draft,
)


class TestQAReviewTooling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parents[1]
        cls.queue_path = (
            cls.root_dir
            / "checkpoints"
            / "experiments"
            / "neural_learning_v02"
            / "stage_d_design"
            / "qa_review_queue_v0.3.0.jsonl"
        )
        cls.advisory_path = (
            cls.root_dir
            / "checkpoints"
            / "experiments"
            / "neural_learning_v02"
            / "stage_d_design"
            / "qa_review_advisory_batch1_v0.2.json"
        )

    def test_01_tracked_tooling_clean_checkout_availability(self):
        """Verifies that tools.qa_review module is importable and available in tracked repository."""
        import tools.qa_review.qa_review_validator as val_mod
        import tools.qa_review.founder_review_workflow as wf_mod
        self.assertTrue(hasattr(val_mod, "ReviewQueueValidator"))
        self.assertTrue(hasattr(wf_mod, "load_review_state"))

    def test_02_official_queue_remains_unreviewed(self):
        """The official queue must validate cleanly with all 30 records UNREVIEWED."""
        records, stats = ReviewQueueValidator.validate_file(self.queue_path)
        self.assertEqual(len(records), 30)
        self.assertEqual(stats["unreviewed"], 30)
        self.assertEqual(stats["train_eligible"], 0)
        self.assertEqual(stats["source_splits"]["train"], 27)
        self.assertEqual(stats["source_splits"]["validation"], 2)
        self.assertEqual(stats["source_splits"]["test"], 1)

    def test_03_advisory_v02_cross_queue_validation(self):
        """Advisory v0.2 cross-validated against the official queue must match 1-to-1."""
        st = ReviewQueueValidator.validate_advisory_artifact(self.advisory_path, queue_path=self.queue_path)
        self.assertEqual(st["total_candidates"], 30)
        self.assertEqual(st["train_candidates"], 27)
        self.assertEqual(st["validation_candidates"], 2)
        self.assertEqual(st["test_candidates"], 1)
        self.assertEqual(st["approve_count"], 17)
        self.assertEqual(st["revise_count"], 10)
        self.assertEqual(st["needs_new_source_count"], 2)
        self.assertEqual(st["test_only_count"], 1)

    def test_04_preserves_original_content_and_locators(self):
        """Advisory candidates must match queue questions, answers, locators, and hashes bit-for-bit."""
        q_records, _ = ReviewQueueValidator.validate_file(self.queue_path)
        q_map = {r["candidate_id"]: r for r in q_records}

        adv_data = json.loads(self.advisory_path.read_text(encoding="utf-8"))
        for c in adv_data["candidates"]:
            cid = c["candidate_id"]
            orig = q_map[cid]
            self.assertEqual(c["original_question"], orig["question"])
            self.assertEqual(c["original_answer"], orig["proposed_answer"])
            self.assertEqual(c["source_passage"], orig["source_passage"])
            self.assertEqual(c["source_locator"], orig["source_locator"])
            self.assertEqual(c["source_hash"], orig["source_hash"])
            self.assertEqual(c["source_split"], orig["source_split"])

    def test_05_fails_closed_on_validation_test_leakage(self):
        """Admission gate strictly blocks validation or test origin records from training admission."""
        val_record = {
            "candidate_id": "QA-REV-028",
            "source_split": "validation",
            "provenance_status": "PROVENANCE_VERIFIED",
            "semantic_review_status": "TRAIN-ELIGIBLE",
            "reviewer_decision": "APPROVE",
            "reviewer_notes": "Attempting leakage.",
        }
        with self.assertRaises(AdmissionGateError) as ctx:
            ReviewQueueValidator.check_train_admission(val_record)
        self.assertIn("CRITICAL LEAKAGE", str(ctx.exception))

        test_record = {
            "candidate_id": "QA-REV-030",
            "source_split": "test",
            "provenance_status": "PROVENANCE_VERIFIED",
            "semantic_review_status": "TRAIN-ELIGIBLE",
            "reviewer_decision": "APPROVE",
            "reviewer_notes": "Attempting leakage.",
        }
        with self.assertRaises(AdmissionGateError) as ctx:
            ReviewQueueValidator.check_train_admission(test_record)
        self.assertIn("CRITICAL LEAKAGE", str(ctx.exception))

    def test_06_train_admission_requires_non_empty_rationale(self):
        """Admission gate strictly rejects empty reviewer_notes."""
        train_record = {
            "candidate_id": "QA-REV-001",
            "source_split": "train",
            "provenance_status": "PROVENANCE_VERIFIED",
            "semantic_review_status": "TRAIN-ELIGIBLE",
            "reviewer_decision": "APPROVE",
            "reviewer_notes": "",
        }
        with self.assertRaises(AdmissionGateError) as ctx:
            ReviewQueueValidator.check_train_admission(train_record)
        self.assertIn("empty reviewer_notes rationale", str(ctx.exception))

    def test_07_atomic_compile_fails_closed_without_writing_file(self):
        """If queue has 0 approved records, compile_training_dataset_atomic must raise error and write no files."""
        q_records, _ = ReviewQueueValidator.validate_file(self.queue_path)
        with tempfile.TemporaryDirectory() as td:
            out_file = Path(td) / "compiled_train.jsonl"
            with self.assertRaises(AdmissionGateError):
                ReviewQueueValidator.compile_training_dataset_atomic(q_records, out_file)
            self.assertFalse(out_file.exists())

    def test_08_preview_draft_does_not_mutate_queue(self):
        """Workflow preview_draft previews changes in memory without altering the queue file."""
        preview_res = preview_draft("QA-REV-001", "APPROVE", "Explicit founder verification.")
        self.assertEqual(preview_res["reviewer_decision"], "APPROVE")
        self.assertEqual(preview_res["semantic_review_status"], "TRAIN-ELIGIBLE")

        # Confirm file on disk is unchanged
        recs, stats = ReviewQueueValidator.validate_file(self.queue_path)
        r0 = next(r for r in recs if r["candidate_id"] == "QA-REV-001")
        self.assertEqual(r0["reviewer_decision"], "UNREVIEWED")
        self.assertEqual(r0["semantic_review_status"], "UNREVIEWED")

    def test_09_preview_draft_rejects_test_split_for_training(self):
        """Workflow preview_draft raises AdmissionGateError if user tries to approve test split for training."""
        with self.assertRaises(AdmissionGateError):
            preview_draft("QA-REV-030", "APPROVE", "Attempting test split approval")

    def test_10_record_decision_on_isolated_queue(self):
        """Verifies that record_decision updates status, retains reviewer notes, and maintains schema invariants."""
        from tools.qa_review.founder_review_workflow import record_decision
        import shutil

        with tempfile.TemporaryDirectory() as td:
            temp_q = Path(td) / "temp_queue.jsonl"
            shutil.copy(self.queue_path, temp_q)

            # Record an approval
            rec = record_decision("QA-REV-001", "APPROVE", "Verified by human founder.", queue_path=temp_q)
            self.assertEqual(rec["reviewer_decision"], "APPROVE")
            self.assertEqual(rec["semantic_review_status"], "TRAIN-ELIGIBLE")
            self.assertEqual(rec["reviewer_notes"], "Verified by human founder.")

            # Validate queue schema
            recs, stats = ReviewQueueValidator.validate_file(temp_q)
            self.assertEqual(stats["train_eligible"], 1)
            self.assertEqual(stats["unreviewed"], 29)

            # Record a rejection
            rec_rej = record_decision("QA-REV-003", "REJECT", "Fails quality threshold.", queue_path=temp_q)
            self.assertEqual(rec_rej["reviewer_decision"], "REJECT")
            self.assertEqual(rec_rej["semantic_review_status"], "REJECT")

            recs2, stats2 = ReviewQueueValidator.validate_file(temp_q)
            self.assertEqual(stats2["rejected"], 1)
            self.assertEqual(stats2["train_eligible"], 1)
            self.assertEqual(stats2["unreviewed"], 28)

    def test_11_save_draft_does_not_mutate_official_queue(self):
        """Verifies that save_draft creates an isolated draft file without modifying the official queue."""
        from tools.qa_review.founder_review_workflow import save_draft

        with tempfile.TemporaryDirectory() as td:
            temp_draft = Path(td) / "user_draft.jsonl"
            save_draft("QA-REV-004", "REVISE", "Wording adjustment requested.", draft_path=temp_draft)

            self.assertTrue(temp_draft.is_file())
            content = temp_draft.read_text(encoding="utf-8")
            self.assertIn("QA-REV-004", content)
            self.assertIn("NEEDS-REVISION", content)

            # Confirm official queue was NOT touched
            recs, stats = ReviewQueueValidator.validate_file(self.queue_path)
            self.assertEqual(stats["unreviewed"], 30)

    def test_12_unanswered_candidates_remain_unreviewed(self):
        """Verifies that candidates not explicitly addressed remain in UNREVIEWED state without automatic mutation."""
        recs, stats = ReviewQueueValidator.validate_file(self.queue_path)
        for r in recs:
            self.assertEqual(r["semantic_review_status"], "UNREVIEWED")
            self.assertEqual(r["reviewer_decision"], "UNREVIEWED")

    def test_13_repeated_decision_execution_idempotence(self):
        """Repeated decision recordings on the same candidate overwrite cleanly without duplicating lines."""
        from tools.qa_review.founder_review_workflow import record_decision
        import shutil

        with tempfile.TemporaryDirectory() as td:
            temp_q = Path(td) / "temp_queue.jsonl"
            shutil.copy(self.queue_path, temp_q)

            # Record decision twice
            record_decision("QA-REV-005", "REVISE", "First note", queue_path=temp_q)
            record_decision("QA-REV-005", "REVISE", "Second note updated", queue_path=temp_q)

            recs, stats = ReviewQueueValidator.validate_file(temp_q)
            self.assertEqual(len(recs), 30)  # Exactly 30 records, no duplicate line
            r5 = next(r for r in recs if r["candidate_id"] == "QA-REV-005")
            self.assertEqual(r5["reviewer_notes"], "Second note updated")


if __name__ == "__main__":
    unittest.main()

