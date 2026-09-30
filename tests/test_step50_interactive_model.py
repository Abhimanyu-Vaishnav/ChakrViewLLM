"""
Step 50 Automated Tests: Trained ChakrView Interactive Model Evaluation.

Tests verify:
  01. Correct checkpoint discovery.
  02. Correct checkpoint selection.
  03. Checkpoint integrity.
  04. Baseline immutability.
  05. Correct tokenizer compatibility.
  06. CLI coordinator startup.
  07. One-shot generation.
  08. Multi-turn context.
  09. /reset command clears turns.
  10. /info command returns full metadata.
  11. /stats command returns quantitative metrics.
  12. Deterministic seed behavior.
  13. Same prompt produces same output under deterministic settings.
  14. Baseline vs experimental comparison.
  15. Probability distribution validity.
  16. Top-k probability ordering.
  17. Context divergence calculation.
  18. Repetition metric correctness.
  19. Session log serialization.
  20. No model weight mutation across all operations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from typing import Dict, Any, List

import pytest
import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    InteractiveModelSessionCoordinator,
    ModelComparator,
    StructuredProbeEvaluator,
    GenerationMetricsCalculator,
    ResponseMetrics,
    ComparisonResult,
    compute_model_hash,
    instantiate_frozen_baseline,
    load_trained_checkpoint,
    compute_cosine_distance,
    compute_js_divergence,
    compute_topk_overlap,
    DEFAULT_TRAINED_CHECKPOINT,
    DEFAULT_TOKENIZER_DIR,
    EXPECTED_TOKENIZER_CHECKSUM,
)
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH, EXPECTED_VOCAB_SIZE, MAX_CONTEXT_WINDOW
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

ROOT_DIR = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────────────
# Test Cases 01–20
# ─────────────────────────────────────────────────────────────────────────────

def test_01_checkpoint_discovery():
    """Verify discovery of candidate checkpoints in repository."""
    pts = list(ROOT_DIR.glob("artifacts/**/*.pt")) + list(ROOT_DIR.glob("checkpoints/**/*.pt"))
    assert len(pts) >= 1, "At least one checkpoint must be discovered in repository"
    assert any("checkpoint" in p.name for p in pts)


def test_02_checkpoint_selection():
    """Verify DEFAULT_TRAINED_CHECKPOINT exists, is readable, and is explicitly selected."""
    assert DEFAULT_TRAINED_CHECKPOINT.is_file(), f"Selected checkpoint {DEFAULT_TRAINED_CHECKPOINT} must exist"
    data = torch.load(DEFAULT_TRAINED_CHECKPOINT, map_location="cpu", weights_only=False)
    assert "model_state_dict" in data
    assert data.get("checkpoint_type") == "training"


def test_03_checkpoint_integrity():
    """Verify parameter count, tensor shapes, and finite weights in selected checkpoint."""
    model, payload, w_hash = load_trained_checkpoint(DEFAULT_TRAINED_CHECKPOINT, allow_baseline=False)
    assert sum(p.numel() for p in model.parameters()) == 3_443_136
    assert len(list(model.parameters())) == 56

    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), f"Tensor {name} contains non-finite weights"

    assert w_hash != EXPECTED_WEIGHT_HASH, "Trained checkpoint must differ from baseline"


def test_04_baseline_immutability():
    """Verify frozen baseline model instantiates with exact ratified SHA-256."""
    baseline = instantiate_frozen_baseline()
    base_hash = compute_model_hash(baseline)
    assert base_hash == EXPECTED_WEIGHT_HASH, f"Baseline hash {base_hash} != {EXPECTED_WEIGHT_HASH}"


def test_05_tokenizer_compatibility():
    """Verify tokenizer artifacts load cleanly and match active checksum."""
    tok, meta = load_tokenizer_artifacts(DEFAULT_TOKENIZER_DIR)
    assert tok.vocab_size == EXPECTED_VOCAB_SIZE
    checksum = meta.get("checksums", {}).get("merges_sha256") or meta.get("checksum")
    assert checksum == EXPECTED_TOKENIZER_CHECKSUM


def test_06_cli_startup():
    """Verify InteractiveModelSessionCoordinator initializes without error."""
    coord = InteractiveModelSessionCoordinator(
        checkpoint_path=DEFAULT_TRAINED_CHECKPOINT,
        tokenizer_dir=DEFAULT_TOKENIZER_DIR,
        deterministic=True,
        seed=42,
    )
    assert coord.trained_model is not None
    assert coord.baseline_model is not None
    assert coord.deterministic is True
    assert coord.seed == 42


def test_07_oneshot_generation():
    """Verify one-shot generation produces text, metrics, and positive latency."""
    coord = InteractiveModelSessionCoordinator()
    text, metrics, _ = coord.generate_turn("Hello from developer.")
    assert isinstance(text, str)
    assert isinstance(metrics, ResponseMetrics)
    assert metrics.length >= 0
    assert metrics.elapsed_sec > 0.0


def test_08_multiturn_context():
    """Verify multi-turn interaction records user and assistant turns consecutively."""
    coord = InteractiveModelSessionCoordinator()
    coord.generate_turn("Turn 1: My favorite color is green.")
    assert len(coord.session.turns) == 2

    coord.generate_turn("Turn 2: What was the color?")
    assert len(coord.session.turns) == 4
    assert coord.session.turns[0].role.value == "user"
    assert coord.session.turns[1].role.value == "assistant"
    assert coord.session.turns[2].role.value == "user"
    assert coord.session.turns[3].role.value == "assistant"


def test_09_reset_command():
    """Verify /reset command clears active turns from session."""
    coord = InteractiveModelSessionCoordinator()
    coord.generate_turn("First prompt before reset.")
    assert len(coord.session.turns) == 2

    coord.reset_context()
    assert len(coord.session.turns) == 0


def test_10_info_command():
    """Verify /info command returns full metadata dictionary."""
    coord = InteractiveModelSessionCoordinator()
    info = coord.get_info()
    required_keys = {
        "checkpoint",
        "training_steps",
        "parameters",
        "vocabulary",
        "context_length",
        "tokenizer_checksum",
        "model_weight_hash",
        "baseline_hash",
        "deterministic",
        "seed",
        "raw_diagnostics",
    }
    assert required_keys.issubset(set(info.keys()))
    assert info["parameters"] == 3_443_136
    assert info["vocabulary"] == 4_096
    assert info["context_length"] == 512


def test_11_stats_command():
    """Verify /stats command tracks prompts, tokens, throughput, and repetition."""
    coord = InteractiveModelSessionCoordinator()
    coord.generate_turn("First test prompt for statistics.")
    coord.generate_turn("Second test prompt for statistics.")

    stats = coord.get_session_stats()
    assert stats["prompts"] == 2
    assert stats["generated_tokens"] >= 2
    assert stats["average_latency_ms"] > 0.0
    assert "average_tokens_per_sec" in stats
    assert "repetition_ratio" in stats


def test_12_deterministic_seed_behavior():
    """Verify setting numeric seed or random mode."""
    coord = InteractiveModelSessionCoordinator()
    s = coord.set_seed(1234)
    assert s == 1234
    assert coord.deterministic is True
    assert coord.seed == 1234

    r = coord.set_seed("random")
    assert r is None
    assert coord.deterministic is False
    assert coord.seed is None


def test_13_deterministic_same_prompt_same_output():
    """Verify identical prompt with fixed seed produces bit-exact identical token sequence."""
    coord1 = InteractiveModelSessionCoordinator(deterministic=True, seed=42)
    coord1.gen_config.sampling.temperature = 0.5
    coord1.gen_config.max_new_tokens = 16
    coord1.gen_config.min_new_tokens = 8

    _, metrics1, _ = coord1.generate_turn("Language models predict")
    tokens1 = list(coord1.history[0]["token_ids"])

    coord2 = InteractiveModelSessionCoordinator(deterministic=True, seed=42)
    coord2.gen_config.sampling.temperature = 0.5
    coord2.gen_config.max_new_tokens = 16
    coord2.gen_config.min_new_tokens = 8

    _, metrics2, _ = coord2.generate_turn("Language models predict")
    tokens2 = list(coord2.history[0]["token_ids"])

    assert tokens1 == tokens2, f"Deterministic generation mismatch: {tokens1} != {tokens2}"


def test_14_baseline_experimental_comparison():
    """Verify side-by-side comparison executes on both models and populates metrics."""
    coord = InteractiveModelSessionCoordinator()
    res = coord.compare_prompt("The capital of India is")
    assert isinstance(res, ComparisonResult)
    assert isinstance(res.baseline_text, str)
    assert isinstance(res.trained_text, str)
    assert isinstance(res.baseline_metrics, ResponseMetrics)
    assert isinstance(res.trained_metrics, ResponseMetrics)
    assert res.baseline_metrics.elapsed_sec > 0.0
    assert res.trained_metrics.elapsed_sec > 0.0


def test_15_probability_distribution_validity():
    """Verify that model output logits produce valid softmax probabilities summing to 1.0."""
    coord = InteractiveModelSessionCoordinator()
    toks = coord.tokenizer.encode("Verification of softmax probabilities", add_bos=True)
    with torch.no_grad():
        logits = coord.trained_model(torch.tensor([toks]))[:, -1, :]
        probs = F.softmax(logits, dim=-1)[0]

    assert probs.shape[0] == EXPECTED_VOCAB_SIZE
    assert torch.all(probs >= 0.0)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_16_topk_probability_ordering():
    """Verify top-k tokens are strictly ordered by descending probability."""
    coord = InteractiveModelSessionCoordinator()
    toks = coord.tokenizer.encode("The weather today is", add_bos=True)
    with torch.no_grad():
        logits = coord.trained_model(torch.tensor([toks]))[:, -1, :]
        probs = F.softmax(logits, dim=-1)[0]
        vals, ids = torch.topk(probs, 5)

    probs_list = vals.tolist()
    for i in range(len(probs_list) - 1):
        assert probs_list[i] >= probs_list[i + 1], "Top-k probabilities must be non-increasing"


def test_17_context_divergence_calculation():
    """Verify JS divergence and Cosine distance properties."""
    p = torch.tensor([0.4, 0.4, 0.2, 0.0])
    q = torch.tensor([0.4, 0.4, 0.2, 0.0])
    r = torch.tensor([0.0, 0.0, 0.5, 0.5])

    # Identical distributions
    assert math.isclose(compute_js_divergence(p, q), 0.0, abs_tol=1e-6)
    assert math.isclose(compute_cosine_distance(p, q), 0.0, abs_tol=1e-6)
    assert compute_topk_overlap(p, q, k=2) == 1.0

    # Distinct distributions
    assert compute_js_divergence(p, r) > 0.0
    assert compute_cosine_distance(p, r) > 0.0


def test_18_repetition_metric_correctness():
    """Verify exact calculation of token diversity and repetition ratios."""
    # Completely distinct tokens
    tokens_distinct = [10, 11, 12, 13]
    m_distinct = GenerationMetricsCalculator.calculate(tokens_distinct, elapsed_sec=0.1)
    assert m_distinct.length == 4
    assert m_distinct.token_diversity == 1.0
    assert m_distinct.repetition_unigram == 0.0
    assert m_distinct.repetition_bigram == 0.0
    assert m_distinct.repetition_trigram == 0.0

    # Fully repeated tokens
    tokens_repeated = [10, 10, 10, 10]
    m_repeated = GenerationMetricsCalculator.calculate(tokens_repeated, elapsed_sec=0.1)
    assert m_repeated.length == 4
    assert m_repeated.token_diversity == 0.25
    assert m_repeated.repetition_unigram == 0.75
    assert math.isclose(m_repeated.repetition_bigram, 1.0 - (1.0 / 3.0), abs_tol=1e-4)
    assert math.isclose(m_repeated.repetition_trigram, 1.0 - (1.0 / 2.0), abs_tol=1e-4)


def test_19_session_log_serialization():
    """Verify session log saves to disk in valid JSON format."""
    coord = InteractiveModelSessionCoordinator()
    coord.generate_turn("Saving session test prompt.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_file = Path(tmp_dir) / "test_session.json"
        saved_path = coord.save_session_log(save_file)
        assert saved_path.is_file()

        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "session_id" in data
        assert "model_info" in data
        assert "history" in data
        assert len(data["history"]) == 1


def test_20_no_model_weight_mutation():
    """Verify Delta W = 0 across all evaluation, comparison, and probe actions."""
    coord = InteractiveModelSessionCoordinator()
    pre_trained_hash = coord.trained_hash
    pre_baseline_hash = coord.baseline_hash

    # Run conversational generation
    coord.generate_turn("Prompt for mutation check 1")
    coord.generate_turn("Prompt for mutation check 2")

    # Run comparison
    coord.compare_prompt("Comparison mutation check")

    # Run probes
    coord.run_probes()

    # Verify hashes unchanged
    post_trained_hash = compute_model_hash(coord.trained_model)
    post_baseline_hash = compute_model_hash(coord.baseline_model)

    assert post_trained_hash == pre_trained_hash, "Trained model weights mutated during evaluation!"
    assert post_baseline_hash == pre_baseline_hash, "Baseline model weights mutated during evaluation!"
    assert post_baseline_hash == EXPECTED_WEIGHT_HASH, "Frozen baseline hash invariant broken!"
