"""
tests/test_coding_suite_custom_checkpoint.py

Unit tests for custom checkpoint resolution and CLI arguments in evaluation/coding_suite/run_coding_evaluation.py.
Verifies:
1. Canonical allowlist key lookup remains completely intact.
2. Custom checkpoint path parsing and resolution.
3. Accurate SHA-256 computation and is_custom metadata flag.
4. Missing / invalid custom checkpoint paths fail closed or report missing.
5. Absolute immutability of CANONICAL_CHECKPOINTS_ALLOWLIST.
"""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chakrview.runtime.neural_runtime_adapter import (
    CANONICAL_CHECKPOINTS_ALLOWLIST,
    DEFAULT_CHECKPOINT_KEY,
)
from evaluation.coding_suite.run_coding_evaluation import evaluate_coding_suite


class TestCodingSuiteCustomCheckpoint(unittest.TestCase):
    def setUp(self):
        self.original_allowlist = copy.deepcopy(CANONICAL_CHECKPOINTS_ALLOWLIST)

    def tearDown(self):
        # Invariant check: Allowlist must never be modified
        self.assertEqual(
            CANONICAL_CHECKPOINTS_ALLOWLIST,
            self.original_allowlist,
            "CANONICAL_CHECKPOINTS_ALLOWLIST was modified during test!",
        )

    def test_canonical_checkpoint_key_behavior_unchanged(self):
        """Smoke evaluation using existing checkpoint key preserves allowlist behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "output.json"
            res = evaluate_coding_suite(
                checkpoint_key=DEFAULT_CHECKPOINT_KEY,
                output_path=str(out_path),
                smoke_mode=True,
            )
            self.assertFalse(res.get("checkpoint_is_custom", False))
            self.assertIn("Canonical", res["checkpoint_label"])
            self.assertEqual(res["total_tasks"], 12)
            self.assertTrue(out_path.is_file())

    def test_custom_checkpoint_path_resolution_and_sha256(self):
        """Providing a valid custom file computes its exact SHA-256 and labels it custom."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_ckpt = Path(tmpdir) / "candidate_model.pt"
            content = b"CHAKRVIEW_DUMMY_WEIGHTS_FOR_TESTING"
            dummy_ckpt.write_bytes(content)
            expected_hash = hashlib.sha256(content).hexdigest()

            out_path = Path(tmpdir) / "output_custom.json"
            res = evaluate_coding_suite(
                custom_checkpoint_path=str(dummy_ckpt),
                output_path=str(out_path),
                smoke_mode=True,
            )

            self.assertTrue(res.get("checkpoint_is_custom"))
            self.assertEqual(res["checkpoint_sha256"], expected_hash)
            self.assertIn("candidate_model.pt", res["checkpoint_label"])
            self.assertIn("Custom Checkpoint", res["checkpoint_label"])

    def test_missing_custom_checkpoint_path(self):
        """Nonexistent custom path is reported truthfully as missing without crashing smoke mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_ckpt = Path(tmpdir) / "does_not_exist.pt"
            out_path = Path(tmpdir) / "output_missing.json"

            res = evaluate_coding_suite(
                custom_checkpoint_path=str(missing_ckpt),
                output_path=str(out_path),
                smoke_mode=True,
            )

            self.assertTrue(res.get("checkpoint_is_custom"))
            self.assertEqual(res["checkpoint_sha256"], "missing")
            self.assertIn("Custom Checkpoint Not Found", res["checkpoint_label"])

    def test_allowlist_untouched_after_custom_evaluation(self):
        """Ensure evaluating a custom checkpoint never injects keys into canonical allowlist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_ckpt = Path(tmpdir) / "arm_a_checkpoint.pt"
            dummy_ckpt.write_bytes(b"dummy_weights")
            out_path = Path(tmpdir) / "output_isolated.json"

            initial_keys = set(CANONICAL_CHECKPOINTS_ALLOWLIST.keys())

            evaluate_coding_suite(
                custom_checkpoint_path=str(dummy_ckpt),
                output_path=str(out_path),
                smoke_mode=True,
            )

            current_keys = set(CANONICAL_CHECKPOINTS_ALLOWLIST.keys())
            self.assertEqual(initial_keys, current_keys)
            self.assertNotIn("arm_a_checkpoint", CANONICAL_CHECKPOINTS_ALLOWLIST)
            self.assertNotIn(str(dummy_ckpt), CANONICAL_CHECKPOINTS_ALLOWLIST)


if __name__ == "__main__":
    unittest.main()

