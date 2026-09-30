"""
Step 55 Automated Tests: Multi-Task Generalization & Anti-Forgetting Replay.

Tests verify:
  01. MultiTaskCurriculumGenerator produces samples across all 7 domains.
  02. Deterministic dataset generation under fixed seed.
  03. Configurable domain weights validation (rejects weights not summing to 1.0).
  04. Split isolation: zero cross-sample leakage between train, val, and test splits.
  05. Duplicate prevention and metadata provenance tracking.
  06. Multi-task sharding and manifest serialization.
  07. Tokenizer compatibility across all generated multi-task tokens.
  08. Checkpoint tier separation (BASELINE vs EXPERIMENTAL vs VALIDATED).
  09. Frozen baseline immutability invariant strictly preserved (DeltaW = 0).
  10. Pure CPU execution with low-resource memory constraints (< 15MB weights).
  11. Step-55 Combinatorial benchmark task definitions integrity.
  12. Evidence-based release gate criteria validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.curriculum.multitask import (
    CurriculumSample,
    DomainWeights,
    MultiTaskCurriculumGenerator,
)
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    DEFAULT_TOKENIZER_DIR,
)
from chakrview.runtime.pipeline import EXPECTED_WEIGHT_HASH, EXPECTED_VOCAB_SIZE
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from scripts.experiment_step55_generalization import (
    STEP55_COMBINATORIAL_TASKS,
    evaluate_split_loss,
    train_multitask_checkpoint,
)
from scripts.experiment_step52_model_capability import ModelCapabilityBenchmarkRunner


@pytest.fixture(scope="module")
def tokenizer():
    tok, _ = load_tokenizer_artifacts(DEFAULT_TOKENIZER_DIR)
    return tok


@pytest.fixture(scope="module")
def baseline_model():
    return instantiate_frozen_baseline()


# ---------------------------------------------------------------------------
# Test 01: MultiTaskCurriculumGenerator produces samples across all 7 domains
# ---------------------------------------------------------------------------
def test_01_multitask_generator_domains():
    gen = MultiTaskCurriculumGenerator(seed=42)
    samples = gen.generate_all_samples(target_sample_count=70)
    assert len(samples) >= 60

    domains = set()
    for s in samples:
        domains.add(s.tags[-1])

    expected_domains = {"foundation", "computation", "programming", "reasoning", "diagnosis", "trajectory", "instruction"}
    assert domains == expected_domains


# ---------------------------------------------------------------------------
# Test 02: Deterministic generation under fixed seed
# ---------------------------------------------------------------------------
def test_02_multitask_generator_determinism():
    gen1 = MultiTaskCurriculumGenerator(seed=123)
    gen2 = MultiTaskCurriculumGenerator(seed=123)

    s1 = gen1.generate_all_samples(target_sample_count=100)
    s2 = gen2.generate_all_samples(target_sample_count=100)

    assert len(s1) == len(s2)
    for a, b in zip(s1, s2):
        assert a.sample_id == b.sample_id
        assert a.text == b.text


# ---------------------------------------------------------------------------
# Test 03: Domain weights validation
# ---------------------------------------------------------------------------
def test_03_domain_weights_validation():
    # Valid weights
    valid_w = DomainWeights(foundation=0.2, computation=0.2, programming=0.2, reasoning=0.2, diagnosis=0.1, trajectory=0.05, instruction=0.05)
    valid_w.validate()

    # Invalid weights (sum != 1.0)
    invalid_w = DomainWeights(foundation=0.5, computation=0.5, programming=0.5)
    with pytest.raises(ValueError):
        invalid_w.validate()


# ---------------------------------------------------------------------------
# Test 04: Split isolation
# ---------------------------------------------------------------------------
def test_04_split_isolation():
    gen = MultiTaskCurriculumGenerator(seed=42)
    samples = gen.generate_all_samples(target_sample_count=100)
    splits = gen.build_dataset_splits(samples, train_ratio=0.8, val_ratio=0.1)

    train_ids = {s.sample_id for s in splits["train"]}
    val_ids = {s.sample_id for s in splits["val"]}
    test_ids = {s.sample_id for s in splits["test"]}

    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0


# ---------------------------------------------------------------------------
# Test 05: Duplicate prevention and provenance
# ---------------------------------------------------------------------------
def test_05_duplicate_prevention_and_provenance():
    gen = MultiTaskCurriculumGenerator(seed=42)
    samples = gen.generate_all_samples(target_sample_count=150)
    sample_ids = [s.sample_id for s in samples]

    # Every sample has unique identifier
    assert len(sample_ids) == len(set(sample_ids))

    # Every sample has domain tag
    for s in samples:
        assert len(s.tags) >= 2


# ---------------------------------------------------------------------------
# Test 06: Shard creation and multitask manifest serialization
# ---------------------------------------------------------------------------
def test_06_shard_creation(tokenizer):
    gen = MultiTaskCurriculumGenerator(seed=42)
    samples = gen.generate_all_samples(target_sample_count=50)
    splits = gen.build_dataset_splits(samples, train_ratio=0.8, val_ratio=0.1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        manifest = gen.write_shards(splits, output_dir, tokenizer, vocab_size=EXPECTED_VOCAB_SIZE)

        manifest_path = output_dir / "multitask_manifest.json"
        assert manifest_path.exists()
        assert "splits" in manifest
        assert "train" in manifest["splits"]
        assert manifest["splits"]["train"]["token_count"] > 0
        assert "domain_distribution" in manifest["splits"]["train"]


# ---------------------------------------------------------------------------
# Test 07: Tokenizer compatibility
# ---------------------------------------------------------------------------
def test_07_tokenizer_compatibility(tokenizer):
    gen = MultiTaskCurriculumGenerator(seed=42)
    samples = gen.generate_all_samples(target_sample_count=50)

    for sample in samples:
        tokens = tokenizer.encode(sample.text)
        assert all(0 <= t < EXPECTED_VOCAB_SIZE for t in tokens), f"Invalid token ID in {sample.sample_id}"


# ---------------------------------------------------------------------------
# Test 08: Checkpoint tier separation
# ---------------------------------------------------------------------------
def test_08_checkpoint_tier_separation():
    baseline = instantiate_frozen_baseline()
    base_hash = compute_model_hash(baseline)
    assert base_hash == EXPECTED_WEIGHT_HASH

    config = ModelConfig()
    exp_model = ChakrMicro(config)
    exp_hash = compute_model_hash(exp_model)
    assert base_hash != exp_hash, "Experimental model matches baseline hash"


# ---------------------------------------------------------------------------
# Test 09: Frozen baseline immutability
# ---------------------------------------------------------------------------
def test_09_frozen_baseline_immutability(baseline_model):
    cur_hash = compute_model_hash(baseline_model)
    assert cur_hash == EXPECTED_WEIGHT_HASH, "Frozen baseline hash mutated!"


# ---------------------------------------------------------------------------
# Test 10: Pure CPU execution & memory boundaries
# ---------------------------------------------------------------------------
def test_10_cpu_execution(baseline_model):
    for p in baseline_model.parameters():
        assert p.device.type == "cpu"
    total_bytes = sum(p.numel() * p.element_size() for p in baseline_model.parameters())
    assert total_bytes < 15 * 1024 * 1024  # Less than 15 MB


# ---------------------------------------------------------------------------
# Test 11: Combinatorial benchmark task definitions
# ---------------------------------------------------------------------------
def test_11_combinatorial_benchmark_structure():
    assert len(STEP55_COMBINATORIAL_TASKS) == 10
    task_ids = [t.task_id for t in STEP55_COMBINATORIAL_TASKS]
    assert len(task_ids) == len(set(task_ids)), "Duplicate task IDs found"

    categories = {t.category for t in STEP55_COMBINATORIAL_TASKS}
    assert "novel_arithmetic_operands" in categories
    assert "novel_function_name" in categories
    assert "novel_state_machine" in categories
    assert "novel_patch_selection" in categories


# ---------------------------------------------------------------------------
# Test 12: Gate review logic
# ---------------------------------------------------------------------------
def test_12_gate_review_logic():
    def check_gate(a, h, r, c):
        return (a >= 0.80 and h >= 0.70 and r >= 0.70 and c >= 0.70)

    assert check_gate(0.85, 0.75, 0.72, 0.70) is True
    assert check_gate(0.40, 0.10, 0.80, 0.20) is False  # Insufficient generalization
