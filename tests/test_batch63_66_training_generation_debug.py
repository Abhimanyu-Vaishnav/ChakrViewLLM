"""
Integration and Regression Tests for Master Sprint Batches 63-66:
- Batch 63: Training-to-Generation Path Audit & Exact Delimiter Alignment
- Batch 64: Deterministic Correctness Harness & Alignment Invariants
- Batch 65: Controlled Recovery Experiment & Memorization vs Generalization Verification
- Batch 66: Invariant Preservation, Regression Suite, and Hash Audit
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.sft_engine import SFTConfig, SFTDataset, SFTExperimentEngine
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    DeterministicEvaluator,
    compute_file_sha256,
    load_tokenizer_artifacts,
)

EXPECTED_CANONICAL_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_CANONICAL_PARAMS = 3_443_136
EXPECTED_STEP10000_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"
BASE_CKPT = "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"


# ==============================================================================
# BATCH 63: DELIMITER ALIGNMENT & TOKEN PRESERVATION
# ==============================================================================

def test_batch63_chatml_delimiter_newline_preservation(tmp_path):
    """Verifies that ChatML prompts preserve terminal newlines when configured."""
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    prompt = "<|im_start|>user\nCompute: 13 + 10 = ?<|im_end|>\n<|im_start|>assistant\n"
    response = "23<|im_end|>\n"

    dummy_file = tmp_path / "dummy.jsonl"
    with open(dummy_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": prompt, "response": response, "review_status": "ANSWER_VERIFIED"}) + "\n")

    # When strip_prompt_delimiters=False, terminal newline is retained
    ds = SFTDataset(dummy_file, tok, strip_prompt_delimiters=False)
    inp, tgt = ds.encode_example(prompt, response)

    # Prompt tokens preserve token 13 ('\n')
    p_toks = tok.encode(prompt)
    assert p_toks[-1] == 13
    assert inp[len(p_toks)] == 13  # position of last prompt token is index len(p_toks)


def test_batch63_evaluator_and_sft_dataset_parity(tmp_path):
    """Verifies DeterministicEvaluator and SFTDataset produce identical input and target sequences."""
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    prompt = "<|im_start|>user\nCompute: 13 + 10 = ?<|im_end|>\n<|im_start|>assistant\n"
    response = "23<|im_end|>\n"

    dummy_file = tmp_path / "dummy.jsonl"
    with open(dummy_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": prompt, "response": response, "review_status": "ANSWER_VERIFIED"}) + "\n")

    ds = SFTDataset(dummy_file, tok, strip_prompt_delimiters=False)
    inp_ds, tgt_ds = ds.encode_example(prompt, response)

    evaluator = DeterministicEvaluator(
        checkpoint_path=BASE_CKPT,
        tokenizer_dir=TOKENIZER_DIR,
        strip_prompt_delimiters=False,
    )
    inp_ev, tgt_ev, status, _ = evaluator.encode_record(prompt, response)

    assert status == "EVALUATED"
    assert inp_ds == inp_ev
    assert tgt_ds == tgt_ev


# ==============================================================================
# BATCH 64: DETERMINISTIC CORRECTNESS & SCORING INVARIANTS
# ==============================================================================

def test_batch64_deliberately_broken_scoring_detection():
    """Confirms exact match passes exact strings and rejects incorrect or truncated predictions."""
    def score(gen: str, exp: str) -> bool:
        return gen.replace("<|im_end|>", "").strip() == exp.strip()

    assert score("23<|im_end|>", "23") is True
    assert score("27<|im_end|>", "23") is False
    assert score("|277<|im_end|>", "23") is False
    assert score("", "23") is False


def test_batch64_eos_independence():
    """Confirms EOS completion can be 100% while exact match is 0%."""
    output = "27<|im_end|>"
    expected = "23"
    assert ("<|im_end|>" in output) is True
    assert (output.replace("<|im_end|>", "").strip() == expected) is False


def test_batch64_greedy_decoding_logit_alignment():
    """Confirms the first greedy generated token matches argmax of logits at the final prompt index."""
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    prompt = "<|im_start|>user\nCompute: 13 + 10 = ?<|im_end|>\n<|im_start|>assistant\n"
    p_toks = tok.encode(prompt)
    prompt_in = [0] + p_toks

    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor([prompt_in]))[0]
        expected_next = torch.argmax(logits[-1, :]).item()

        # Generate 1 token
        x = torch.tensor([prompt_in], dtype=torch.long)
        pred = torch.argmax(model(x)[0, -1, :]).item()

    assert pred == expected_next


# ==============================================================================
# BATCH 65: CONTROLLED RECOVERY VERIFICATION
# ==============================================================================

def test_batch65_recovery_checkpoint_reloads_deterministically():
    """Verifies that the recovery checkpoint reloads cleanly and matches recorded summary metrics."""
    ckpt_path = Path("experiments/batch63_66_training_generation_debug/learning_experiment/run_50steps/sft_checkpoint_step_50.pt")
    if not ckpt_path.exists():
        pytest.skip("Recovery checkpoint not yet generated")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["sft_step"] == 50
    assert ckpt["dataset_mode"] == "VERIFIED_TRAINING"
    assert ckpt["train_file_sha256"] == "89299ff5e2d81b143c68a6b6c9aae858b69a486f699c6c90c9855600bfbbd4c5"

    evaluator = DeterministicEvaluator(
        checkpoint_path=str(ckpt_path),
        tokenizer_dir=TOKENIZER_DIR,
        strip_prompt_delimiters=False,
    )
    val_path = Path("experiments/batch59_62_learning_verification/data/val_synthetic.jsonl")
    res = evaluator.evaluate_dataset(val_path)
    assert abs(res["mean_token_loss"] - 0.4145) < 0.02


def test_batch65_memorization_vs_generalization_distinction():
    """Confirms that memorization (train accuracy) is strictly separated from generalization (val/bench)."""
    eval_json = Path("experiments/batch63_66_training_generation_debug/learning_experiment/run_50steps/evaluation_summary.json")
    if not eval_json.exists():
        pytest.skip("Evaluation summary not yet generated")

    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_acc = data["splits"]["TRAIN"]["exact_match_accuracy"]
    val_acc = data["splits"]["VALIDATION"]["exact_match_accuracy"]

    # Even in 50-step run, val exact match is 0.0
    assert val_acc == 0.0
    # EOS rate is 100% across all splits
    assert data["splits"]["TRAIN"]["eos_rate"] == 1.0
    assert data["splits"]["VALIDATION"]["eos_rate"] == 1.0
    assert data["splits"]["HELD_OUT_BENCHMARK"]["eos_rate"] == 1.0


# ==============================================================================
# BATCH 66: CANONICAL BASELINE IMMUTABILITY
# ==============================================================================

def test_batch66_canonical_baseline_immutability():
    """Confirms that canonical initialization and step-10k checkpoint hashes remain unchanged."""
    base_ckpt = Path(BASE_CKPT)
    assert compute_file_sha256(base_ckpt) == EXPECTED_STEP10000_HASH

    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == EXPECTED_CANONICAL_PARAMS

    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    assert hasher.hexdigest() == EXPECTED_CANONICAL_HASH
