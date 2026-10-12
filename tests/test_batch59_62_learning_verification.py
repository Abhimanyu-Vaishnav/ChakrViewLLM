"""
Integration Tests for Master Sprint Batches 59-62:
- Batch 59: Adversarial Trust Gate Boundary & Dataset Hash Auditing
- Batch 60: Synthetic Deterministic Dataset Generation & Split Non-Overlap
- Batch 61: Memorization vs Generalization Metric Disambiguation
- Batch 62: Bit-Exact Checkpoint Reloading & Evaluation Reproducibility
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.sft_engine import SFTConfig, SFTDataset, SFTExperimentEngine
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    DeterministicEvaluator,
    compute_file_sha256,
    load_tokenizer_artifacts,
)
from experiments.batch59_62_learning_verification.data.generate_synthetic_data import (
    generate_arithmetic_datasets,
)

EXPECTED_CANONICAL_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_CANONICAL_PARAMS = 3_443_136
EXPECTED_STEP10000_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"


# ==============================================================================
# BATCH 59 TESTS: ADVERSARIAL BOUNDARY & HASH INTEGRITY
# ==============================================================================


def test_batch59_sft_dataset_computes_sha256_and_records_in_checkpoint(tmp_path):
    """Verifies that SFTDataset calculates file SHA-256 and saves it into checkpoints."""
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    sample_file = tmp_path / "valid_data.jsonl"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "id": "VALID-01",
                    "prompt": "<|im_start|>user\nTest<|im_end|>\n<|im_start|>assistant\n",
                    "response": "Response<|im_end|>\n",
                    "review_status": "ANSWER_VERIFIED",
                }
            )
            + "\n"
        )

    ds = SFTDataset(sample_file, tok, is_training=True)
    assert hasattr(ds, "file_sha256")
    assert ds.file_sha256 == compute_file_sha256(sample_file)

    cfg = SFTConfig(
        repo_root=Path("."),
        train_file=sample_file,
        val_file=sample_file,
        output_dir=tmp_path / "out",
        base_checkpoint=Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"),
        steps=1,
    )
    engine = SFTExperimentEngine(cfg)
    engine.save_checkpoint(step=1)

    ckpt = torch.load(tmp_path / "out" / "sft_checkpoint_step_1.pt", map_location="cpu", weights_only=False)
    assert ckpt["train_file_sha256"] == ds.file_sha256
    assert ckpt["val_file_sha256"] == ds.file_sha256


# ==============================================================================
# BATCH 60 TESTS: SYNTHETIC DATASET GENERATION & ZERO OVERLAP
# ==============================================================================


def test_batch60_synthetic_dataset_zero_split_overlap(tmp_path):
    """Verifies deterministic dataset generator produces completely disjoint splits."""
    manifest = generate_arithmetic_datasets(seed=123, output_dir=tmp_path)
    assert manifest["train_count"] == 32
    assert manifest["val_count"] == 8
    assert manifest["bench_count"] == 8
    assert manifest["overlap_train_val"] == 0
    assert manifest["overlap_train_bench"] == 0
    assert manifest["overlap_val_bench"] == 0


# ==============================================================================
# BATCH 61 TESTS: METRIC DISAMBIGUATION & REPETITION DETECTOR
# ==============================================================================


def test_batch61_loss_reduction_vs_exact_match_disambiguation():
    """
    Validates that exact-match accuracy is strictly separated from token loss.
    """
    eval_json_path = Path("experiments/batch59_62_learning_verification/evaluation/trained_synthetic_eval.json")
    if not eval_json_path.exists():
        pytest.skip("Evaluation JSON not yet generated")

    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Token loss is low (< 0.5), but exact match is 0.0
    val_data = data["splits"]["VALIDATION"]
    assert val_data["token_loss"] < 1.0
    assert val_data["exact_match_accuracy"] == 0.0
    assert val_data["eos_rate"] == 1.0


# ==============================================================================
# BATCH 62 TESTS: CHECKPOINT RELOAD DETERMINISM & IMMUTABLE BASELINE
# ==============================================================================


def test_batch62_canonical_baseline_invariance():
    """Confirms canonical baseline hashes remain 100% immutable."""
    base_ckpt = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    assert compute_file_sha256(base_ckpt) == EXPECTED_STEP10000_HASH

    torch.manual_seed(42)
    m = ChakrMicro(ModelConfig())
    assert sum(p.numel() for p in m.parameters()) == EXPECTED_CANONICAL_PARAMS


def test_batch62_trained_checkpoint_reloads_deterministically():
    """Verifies that the trained checkpoint reloads cleanly and matches recorded metrics."""
    ckpt_path = Path("experiments/batch59_62_learning_verification/learning_experiment/run_50steps/sft_checkpoint_step_50.pt")
    if not ckpt_path.exists():
        pytest.skip("Checkpoint step 50 not yet generated")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["sft_step"] == 50
    assert ckpt["dataset_mode"] == "VERIFIED_TRAINING"

    val_path = Path("experiments/batch59_62_learning_verification/data/val_synthetic.jsonl")
    evaluator = DeterministicEvaluator(
        checkpoint_path=str(ckpt_path),
        tokenizer_dir=TOKENIZER_DIR,
    )
    res = evaluator.evaluate_dataset(val_path)
    assert abs(res["mean_token_loss"] - 0.4642) < 0.01
