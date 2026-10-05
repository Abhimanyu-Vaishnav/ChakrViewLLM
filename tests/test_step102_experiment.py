"""
Tests for Step 102 Phase 6: Experiment Evaluation and Canonical Invariant Verification.

Verifies:
1. Step 102 experiment summary artifact exists and contains complete trajectory and evaluation data.
2. Step 102 trained checkpoint exists, loads cleanly, and differs from canonical baseline.
3. Candidate achieves validation loss <= 4.80 and validation perplexity <= 120.0.
4. Candidate achieves test loss <= 4.80 and test perplexity <= 120.0.
5. Candidate improves top-5 syntactic probe accuracy over baseline (15.0% vs 5.0%).
6. Inference after loading the candidate checkpoint preserves candidate weight hash (ΔW = 0).
7. Canonical frozen baseline remains exactly 3,443,136 parameters and bit-exact SHA-256 c5571c...
"""

import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
    load_trained_checkpoint,
)
from chakrview.runtime.neural_inference_contract import (
    NeuralInferenceContract,
    NeuralInferencePayload,
    NeuralInferenceConfig,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

ROOT_DIR = Path(__file__).resolve().parents[1]
SUMMARY_FILE = ROOT_DIR / "artifacts" / "step102_stage_b_curriculum" / "step102_experiment_summary.json"
TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"


@pytest.fixture
def tokenizer():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    return tok


def test_step102_experiment_summary_integrity():
    """Verify that Step 102 experiment summary exists and documents valid empirical measurements."""
    assert SUMMARY_FILE.is_file(), f"Summary file missing: {SUMMARY_FILE}"
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["milestone"] == "Step 102"
    assert data["steps_completed"] == 500
    assert len(data["trajectory"]) == 1000  # 500 optimizer steps * 2 accumulation micro-steps

    cand = data["trained_candidate"]
    base = data["canonical_baseline"]

    # Verify quantitative targets
    assert cand["validation_loss"] <= 4.80, f"Validation loss {cand['validation_loss']} > 4.80"
    assert cand["validation_perplexity"] <= 120.0, f"Val PPL {cand['validation_perplexity']} > 120.0"
    assert cand["test_loss"] <= 4.80, f"Test loss {cand['test_loss']} > 4.80"
    assert cand["test_perplexity"] <= 120.0, f"Test PPL {cand['test_perplexity']} > 120.0"

    # Verify probe gains
    assert cand["probe_top5_accuracy"] > base["probe_top5_accuracy"]
    assert cand["probe_mean_log_prob"] > base["probe_mean_log_prob"]

    # Verify checkpoint exists
    ckpt_path = Path(cand["checkpoint_path"])
    assert ckpt_path.is_file()


def test_step102_trained_checkpoint_inference_and_immutability(tokenizer):
    """Verify that trained candidate checkpoint loads, runs inference, and preserves ΔW = 0."""
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    ckpt_path = Path(data["trained_candidate"]["checkpoint_path"])
    contract = NeuralInferenceContract.from_checkpoint(ckpt_path, tokenizer=tokenizer)
    trained_hash = contract.compute_weight_hash()

    assert trained_hash == data["trained_candidate"]["final_weight_hash"]
    assert trained_hash != EXPECTED_WEIGHT_HASH

    payload = NeuralInferencePayload(
        prompt="def process_data(items):\n   ",
        config=NeuralInferenceConfig(max_new_tokens=6, temperature=0.0),
    )
    out = contract.generate(payload)
    assert out.weight_hash_verified is True
    assert contract.compute_weight_hash() == trained_hash
    assert len(out.generated_tokens) > 0


def test_step102_canonical_baseline_bit_exactness():
    """Verify that the canonical frozen baseline has NOT been modified by the Step 102 training."""
    canonical_model = instantiate_frozen_baseline()
    param_count = sum(p.numel() for p in canonical_model.parameters())
    assert param_count == 3_443_136
    h = compute_model_hash(canonical_model)
    assert h == EXPECTED_WEIGHT_HASH
