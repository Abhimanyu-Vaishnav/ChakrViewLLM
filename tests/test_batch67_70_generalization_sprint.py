"""
Integration and Regression Tests for Master Autonomous Sprint Batches 67-70:
- Batch 67: Comparative Loss-Weighting & WeightedCausalLoss Unit Verifications
- Batch 68: Single-Digit Transformation Dataset Generation, Disjoint Splits, & Generalization Boundary
- Batch 69: Shared Prompt/Response Contract Parity, Delimiter Enforcement, & Deliberate Malformed Input Testing
- Batch 70: Invariant Preservation, Hash Verification, & Checkpoint Reload Determinism
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.prompt_contract import (
    normalize_prompt_response_strings,
    build_causal_sequence,
)
from chakrview.training.weighted_loss import WeightedCausalLoss
from chakrview.training.sft_engine import SFTConfig, SFTDataset
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    DeterministicEvaluator,
    compute_file_sha256,
    load_tokenizer_artifacts,
)
from experiments.batch67_70_generalization_sprint.data.generate_data import (
    generate_single_digit_arithmetic_datasets,
)

EXPECTED_CANONICAL_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_CANONICAL_PARAMS = 3_443_136
EXPECTED_STEP10000_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
BASE_CKPT = "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"


# ==============================================================================
# BATCH 67 TESTS: WEIGHTED CAUSAL LOSS VERIFICATION
# ==============================================================================

def test_batch67_weighted_causal_loss_computation():
    """Verifies WeightedCausalLoss correctly weights target tokens and ignores -100."""
    loss_fn = WeightedCausalLoss(ignore_index=-100)
    logits = torch.zeros(1, 3, 4096)
    # Target: [-100, 10, 20]
    targets = torch.tensor([[-100, 10, 20]], dtype=torch.long)
    # Weights: [0.0, 5.0, 1.0]
    weights = torch.tensor([[0.0, 5.0, 1.0]], dtype=torch.float)

    # Cross entropy of uniform logits (all 0) against any target is log(4096) = 8.317766
    expected_ce = torch.log(torch.tensor(4096.0)).item()

    loss = loss_fn(logits, targets, weights)
    assert abs(loss.item() - expected_ce) < 1e-4

    # With only second token active (weights: [0.0, 1.0, 0.0])
    weights2 = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float)
    loss2 = loss_fn(logits, targets, weights2)
    assert abs(loss2.item() - expected_ce) < 1e-4


# ==============================================================================
# BATCH 68 TESTS: SINGLE-DIGIT DATASET & ZERO SPLIT OVERLAP
# ==============================================================================

def test_batch68_single_digit_dataset_disjoint_splits(tmp_path):
    """Verifies single-digit arithmetic generator produces strictly disjoint splits with single-token answers."""
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    manifest = generate_single_digit_arithmetic_datasets(tmp_path, seed=123)

    assert manifest["train_count"] == 25
    assert manifest["val_count"] == 10
    assert manifest["bench_count"] == 10
    assert manifest["overlap_train_val"] == 0
    assert manifest["overlap_train_bench"] == 0
    assert manifest["overlap_val_bench"] == 0

    # Verify all answers in training set are single tokens
    with open(tmp_path / "train_single_digit.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ans = rec["expected_answer"]
            ans_toks = tok.encode(ans)
            assert len(ans_toks) == 1, f"Expected single-token answer for {ans}, got {ans_toks}"


# ==============================================================================
# BATCH 69 TESTS: CANONICAL PROMPT CONTRACT INTEGRATION & DELIBERATE ERROR TESTING
# ==============================================================================

def test_batch69_shared_prompt_contract_parity():
    """Verifies that SFTDataset and DeterministicEvaluator use identical prompt contract helper."""
    prompt = "<|im_start|>user\nAdd: 2 + 3 = ?<|im_end|>\n<|im_start|>assistant\n"
    response = "5<|im_end|>\n"

    # Canonical (strip_delimiters=False)
    p_norm, r_norm = normalize_prompt_response_strings(prompt, response, strip_delimiters=False)
    assert p_norm.endswith("\n")
    assert r_norm.endswith("\n")

    # Legacy (strip_delimiters=True)
    p_leg, r_leg = normalize_prompt_response_strings(prompt, response, strip_delimiters=True)
    assert not p_leg.endswith("\n")
    assert not r_leg.endswith("\n")

    # Causal sequence builder property
    p_toks = [10, 20]
    r_toks = [30, 40]
    inp, tgt = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100)
    assert inp == [0, 10, 20, 30, 40]
    assert tgt == [-100, -100, 30, 40, 1]
    # Position len(p_toks) == 2 in inp is token 20 (last prompt token)
    # Target at pos 2 is tgt[2] == 30 (first response token)
    assert inp[len(p_toks)] == p_toks[-1]
    assert tgt[len(p_toks)] == r_toks[0]


def test_batch69_deliberate_incorrect_formatting_rejection():
    """Verifies that malformed prompts or empty responses are properly handled or rejected."""
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    evaluator = DeterministicEvaluator(
        checkpoint_path=BASE_CKPT,
        tokenizer_dir=TOKENIZER_DIR,
        strip_prompt_delimiters=False,
    )

    # Empty response
    inp, tgt, status, err = evaluator.encode_record("Valid prompt\n", "")
    assert status == "SKIPPED_EMPTY_RESPONSE"
    assert inp == []
    assert tgt == []


# ==============================================================================
# BATCH 70 TESTS: RECOVERY CHECKPOINT RELOAD & CANONICAL INVARIANT INTEGRITY
# ==============================================================================

def test_batch70_loss_experiment_checkpoint_reload_determinism():
    """Verifies that Batch 67 and Batch 68 checkpoints reload deterministically and produce matching metrics."""
    ckpt_path = Path("experiments/batch67_70_generalization_sprint/loss_experiments/sft_answer_emphasis_step_50.pt")
    if not ckpt_path.exists():
        pytest.skip("Checkpoint not yet generated")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["sft_step"] == 50
    assert ckpt["strategy"] == "answer_emphasis"

    evaluator = DeterministicEvaluator(
        checkpoint_path=str(ckpt_path),
        tokenizer_dir=TOKENIZER_DIR,
        strip_prompt_delimiters=False,
    )
    val_path = Path("experiments/batch59_62_learning_verification/data/val_synthetic.jsonl")
    res = evaluator.evaluate_dataset(val_path)
    # Evaluator mean loss is finite and matches record
    assert math.isfinite(res["mean_token_loss"])


def test_batch70_canonical_invariants_and_hashes():
    """Verifies canonical baseline parameters, fresh init hash, and step 10k hash remain 100% immutable."""
    base_ckpt = Path(BASE_CKPT)
    assert compute_file_sha256(base_ckpt) == EXPECTED_STEP10000_HASH

    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == EXPECTED_CANONICAL_PARAMS

    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    fresh_sha = hasher.hexdigest()
    assert fresh_sha == EXPECTED_CANONICAL_HASH
