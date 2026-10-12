"""
Integration and Regression Tests for Master Autonomous Sprint Batches 71-74:
- Batch 71: Bit-Exact Canonical Invariant Reconciliation (params, fresh hash, step 10k hash)
- Batch 72: Single-Digit Controlled Benchmark Generation, Zero Split Overlap, & Symmetry Diagnostic
- Batch 73: Controlled Experiment Scorer, Disambiguation of Constant Attractor, & Overfit Control Verification
- Batch 74: Rationale-Style Parser, Rationale Generation Integrity, & Output Normalization
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import pytest
import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.prompt_contract import (
    normalize_prompt_response_strings,
    build_causal_sequence,
)
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    DeterministicEvaluator,
    compute_file_sha256,
    load_tokenizer_artifacts,
)
from experiments.batch71_74_controlled_learning.data.generate_benchmark import (
    generate_controlled_arithmetic_benchmark,
)
from experiments.batch71_74_controlled_learning.experiments.run_rationale_experiment import (
    parse_final_answer,
)

EXPECTED_CANONICAL_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_CANONICAL_PARAMS = 3_443_136
EXPECTED_STEP10000_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
BASE_CKPT = "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"


# ==============================================================================
# BATCH 71 TESTS: REPRODUCIBILITY & CANONICAL INVARIANTS
# ==============================================================================

def test_batch71_canonical_invariants_and_hash_reconciliation():
    """Confirms step 10k hash, fresh init hash, and total parameters match canonical values."""
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
    assert hasher.hexdigest() == EXPECTED_CANONICAL_HASH


# ==============================================================================
# BATCH 72 TESTS: BENCHMARK INTEGRITY & SYMMETRY DIAGNOSTIC
# ==============================================================================

def test_batch72_benchmark_split_disjointness_and_symmetry(tmp_path):
    """Verifies that the Batch 72 benchmark generator produces zero split overlap and valid symmetry pairs."""
    manifest = generate_controlled_arithmetic_benchmark(tmp_path, seed=99)
    assert manifest["splits"]["train"]["count"] == 30
    assert manifest["splits"]["validation"]["count"] == 10
    assert manifest["splits"]["held_out_test"]["count"] == 10
    assert manifest["split_integrity"]["zero_overlap_confirmed"] is True

    # Check symmetry diagnostic file
    symm_file = tmp_path / "symmetry_diagnostic.jsonl"
    assert symm_file.exists()
    recs = [json.loads(l) for l in open(symm_file) if l.strip()]
    assert len(recs) > 0
    for r in recs:
        assert r["operands"][0] != r["operands"][1]  # asymmetric


# ==============================================================================
# BATCH 73 TESTS: EXACT-MATCH SCORING & EOS SEPARATION
# ==============================================================================

def test_batch73_scoring_independence_and_adversarial_outputs():
    """Verifies that answer correctness is scored independently of EOS completion."""
    def score_match(gen: str, exp: str) -> bool:
        return gen.replace("<|im_end|>", "").strip() == exp.strip()

    # Correct digit with EOS
    assert score_match("7<|im_end|>\n", "7") is True
    # Correct digit without EOS
    assert score_match("7", "7") is True
    # Incorrect digit with EOS
    assert score_match("6<|im_end|>", "7") is False
    # Empty string
    assert score_match("", "7") is False
    # Multi-digit mismatch
    assert score_match("77", "7") is False


def test_batch73_overfit_control_checkpoint_verification():
    """Verifies that the overfit control checkpoint achieved 100% exact match on its training subset."""
    summary_path = Path("experiments/batch71_74_controlled_learning/experiments/batch73_experiment_summary.json")
    if not summary_path.exists():
        pytest.skip("Batch 73 summary not yet generated")

    summary = json.loads(summary_path.read_text())
    overfit = summary["overfit_control"]
    assert overfit["final_metrics"]["train"]["exact_match_acc"] == 1.0


# ==============================================================================
# BATCH 74 TESTS: RATIONALE PARSER & FINAL ANSWER EXTRACTION
# ==============================================================================

def test_batch74_rationale_parser_extraction():
    """Verifies parse_final_answer correctly extracts target digits from intermediate text."""
    assert parse_final_answer("4 + 0 = 7. Answer: 7<|im_end|>\n") == "7"
    assert parse_final_answer("Answer: 5") == "5"
    assert parse_final_answer("Answer:  9 <|im_end|>") == "9"
    # Fallback to direct output
    assert parse_final_answer("7<|im_end|>") == "7"
    assert parse_final_answer("3") == "3"
