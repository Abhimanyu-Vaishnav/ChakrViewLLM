"""
tests/test_evaluation_contracts.py

Deterministic test suite verifying model loadability, canonical weights integrity,
tokenizer contracts, and deterministic generation behaviour for ChakrMicro.
"""

import sys
import hashlib
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.neural_runtime_adapter import (
    NeuralRuntimeAdapter,
    DEFAULT_CHECKPOINT_KEY,
)


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


CANONICAL_CHECKPOINT_SHA256 = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
EXPECTED_VOCAB_SHA256 = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"
EXPECTED_MERGES_SHA256 = "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7"
EXPECTED_MODEL_PY_SHA256 = "1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361"
EXPECTED_SESSION_PY_SHA256 = "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"


def test_canonical_artifact_hashes():
    """Verify bit-exact hashes of checkpoint, tokenizer files, and governed runtime files."""
    ckpt_path = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
    vocab_path = REPO_ROOT / "data" / "experiments" / "vocab_4096" / "vocab.json"
    merges_path = REPO_ROOT / "data" / "experiments" / "vocab_4096" / "merges.json"
    model_py_path = REPO_ROOT / "chakrview" / "brain" / "model.py"
    session_py_path = REPO_ROOT / "chakrview" / "runtime" / "session.py"

    assert ckpt_path.is_file(), f"Canonical checkpoint not found at {ckpt_path}"
    assert compute_sha256(ckpt_path) == CANONICAL_CHECKPOINT_SHA256

    assert vocab_path.is_file(), f"Vocab not found at {vocab_path}"
    assert compute_sha256(vocab_path) == EXPECTED_VOCAB_SHA256

    assert merges_path.is_file(), f"Merges not found at {merges_path}"
    assert compute_sha256(merges_path) == EXPECTED_MERGES_SHA256

    assert compute_sha256(model_py_path) == EXPECTED_MODEL_PY_SHA256
    assert compute_sha256(session_py_path) == EXPECTED_SESSION_PY_SHA256


def test_model_loadability_and_tied_weights():
    """Verify the neural adapter initializes cleanly and embedding weights are tied."""
    adapter = NeuralRuntimeAdapter(
        checkpoint_path=REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt",
        max_new_tokens=16,
        decoding="greedy",
    )
    adapter.initialize()
    model = adapter._contract.model

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 3_443_136, f"Expected 3,443,136 parameters, got {total_params}"

    # Verify tied embedding weights (lm_head.weight sharing memory with embedding.weight)
    assert model.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_tokenizer_roundtrip_contract():
    """Verify tokenizer encode-decode identity contract on clean text."""
    adapter = NeuralRuntimeAdapter(
        checkpoint_path=REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt",
        max_new_tokens=16,
        decoding="greedy",
    )
    adapter.initialize()
    tok = adapter._tokenizer

    sample = "ChakrView indigenous micro-intelligence model."
    token_ids = tok.encode(sample)
    assert len(token_ids) > 0
    decoded = tok.decode(token_ids)
    assert decoded == sample


def test_greedy_generation_determinism():
    """Verify that greedy generation produces bit-exact identical strings across calls."""
    adapter = NeuralRuntimeAdapter(
        checkpoint_path=REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt",
        max_new_tokens=16,
        decoding="greedy",
    )
    adapter.initialize()

    prompt = "The capital of France is"
    out1 = adapter.generate(prompt)["generated_text"]
    out2 = adapter.generate(prompt)["generated_text"]

    assert out1 == out2, f"Greedy generation was non-deterministic: '{out1}' != '{out2}'"
    assert len(out1) > 0


def test_scoring_rules_prevent_false_positives():
    """Verify that comment-only, empty completions, or trivial punctuation cannot pass benchmarks."""
    from evaluation.intelligence_suite.scoring_evaluator import evaluate_response
    from evaluation.coding_suite.run_coding_evaluation import evaluate_coding_suite

    # 1. Punctuation-only output must fail the sentence probe
    passed, reason = evaluate_response(
        generated_text="\n* [http://www.gov.co.com/books?id=200",
        expected_concept="complete_three_word_sentence",
        scoring_rule="three_word_sentence"
    )
    assert not passed, f"Punctuation substring incorrectly passed: {reason}"
    assert "INVALID_THREE_WORD_SENTENCE" in reason

    # Genuine 3-word sentence must pass
    valid_passed, valid_reason = evaluate_response(
        generated_text="Cats are fast.",
        expected_concept="complete_three_word_sentence",
        scoring_rule="three_word_sentence"
    )
    assert valid_passed, f"Valid sentence failed: {valid_reason}"

    # 2. Empty or comment-only completion in coding evaluation must NOT pass functional tests
    mock_comment_generator = lambda task: "\n    # Just a comment with no actual code\n    # Another comment\n"
    res_comment = evaluate_coding_suite(smoke_mode=False, mock_generator=mock_comment_generator)
    assert res_comment["test_passed_count"] == 0, "Comment-only completions must never pass functional coding tests!"

    mock_empty_generator = lambda task: "   \n  "
    res_empty = evaluate_coding_suite(smoke_mode=False, mock_generator=mock_empty_generator)
    assert res_empty["test_passed_count"] == 0, "Empty completions must never pass functional coding tests!"
