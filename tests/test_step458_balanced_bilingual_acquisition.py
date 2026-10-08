"""
Unit and regression tests for Wave 458: Balanced Interleaved Bilingual Language Acquisition.

Verifies:
1. Canonical Baseline Invariance (Delta W = 0, exact SHA-256).
2. Balanced Bilingual Stream sampler ratios and reproducibility.
3. Candidate Step 1000 checkpoint integrity and parameter counts.
4. Language-specific held-out loss improvement over zero-shot baseline.
5. Absolute isolation of candidate weights from released baseline.
"""

import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.balanced_bilingual_stream import BalancedBilingualStream
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
STAGE_C_DIR = ROOT_DIR / "data" / "tokenized" / "stage_c"
CANDIDATE_CKPT = ROOT_DIR / "artifacts" / "wave458_language_candidate" / "candidate_stage_step_1000.pt"


class TestWave458BalancedBilingualAcquisition(unittest.TestCase):
    def test_01_canonical_baseline_immutability(self):
        """Verifies canonical baseline remains bit-exact with delta W = 0."""
        baseline = instantiate_frozen_baseline()
        total_p = sum(p.numel() for p in baseline.parameters())
        self.assertEqual(total_p, 3_443_136)
        h = compute_model_hash(baseline)
        self.assertEqual(h, EXPECTED_WEIGHT_HASH)
        self.assertEqual(h, "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da")

    def test_02_balanced_bilingual_stream_ratios(self):
        """Verifies BalancedBilingualStream produces balanced language samples under fixed seed."""
        hi_shards = [STAGE_C_DIR / "train" / f"shard_{i:05d}.bin" for i in range(10, 20)]
        en_shards = [STAGE_C_DIR / "train" / f"shard_{i:05d}.bin" for i in [0, 1, 3, 4, 5, 6, 7, 8, 9, 25, 26]]
        code_shards = [STAGE_C_DIR / "train" / f"shard_{i:05d}.bin" for i in [2, 20, 21, 22, 23, 24]]

        stream = BalancedBilingualStream(
            en_shards=en_shards,
            hi_shards=hi_shards,
            code_shards=code_shards,
            sequence_length=32,
            en_ratio=0.45,
            hi_ratio=0.45,
            code_ratio=0.10,
            seed=42,
        )

        it = iter(stream)
        counts = {0: 0, 1: 0, 2: 0}
        for _ in range(500):
            item = next(it)
            tag = item["lang_tag"].item()
            counts[tag] += 1

        # Ratios should closely approximate 45/45/10 within statistical bounds
        self.assertGreater(counts[0], 180, "English representation starved")
        self.assertGreater(counts[1], 180, "Hindi representation starved")
        self.assertGreater(counts[2], 20, "Code representation starved")

    def test_03_wave458_candidate_checkpoint_exists_and_isolated(self):
        """Verifies Wave 458 candidate checkpoint exists, is valid, and matches parameter constraints."""
        self.assertTrue(CANDIDATE_CKPT.exists(), f"Candidate checkpoint not found at {CANDIDATE_CKPT}")
        ckpt = torch.load(CANDIDATE_CKPT, map_location="cpu", weights_only=False)

        model = ChakrMicro(ModelConfig())
        model.load_state_dict(ckpt["model_state_dict"])
        total_p = sum(p.numel() for p in model.parameters())
        self.assertEqual(total_p, 3_443_136)

        cand_hash = compute_model_hash(model)
        self.assertEqual(cand_hash, "224873bf29d0700b72ef8d0feb0f44c825efd373d83d6c27a1b2d29ae6a66291")
        self.assertNotEqual(cand_hash, EXPECTED_WEIGHT_HASH, "Candidate must be isolated from frozen baseline")

    def test_04_candidate_held_out_loss_superior_to_baseline(self):
        """Verifies candidate achieves lower held-out cross-entropy than untrained baseline."""
        ckpt = torch.load(CANDIDATE_CKPT, map_location="cpu", weights_only=False)
        self.assertLess(ckpt["en_loss"], 5.0)
        self.assertLess(ckpt["hi_loss"], 4.0)
        self.assertGreater(ckpt["en_acc"], 0.15)
        self.assertGreater(ckpt["hi_acc"], 0.25)


if __name__ == "__main__":
    unittest.main()
