"""
Unit and integration tests for Step 103: Neural Capability Consolidation & Pre-Stage-C Readiness.

Verifies:
1. Canonical baseline invariant remains strictly bit-exact (3,443,136 params, hash c5571c...).
2. Step 102 candidate checkpoint remains intact (hash d5886e...) and reproducible.
3. Step 103 consolidation checkpoint exists (hash b98bf8...) in isolated directory.
4. Pre-Stage-C Decision Gate prerequisites are verified (all Stage C splits present and valid).
5. Inference over both trained candidates preserves their respective weight hashes (ΔW = 0).
"""

import json
from pathlib import Path
import pytest
import torch

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
TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"
STEP102_CKPT = ROOT_DIR / "artifacts" / "step102_stage_b_curriculum" / "checkpoints" / "checkpoint_0000500.pt"
STEP103_CKPT = ROOT_DIR / "artifacts" / "step103_stage_b_consolidation" / "checkpoints" / "checkpoint_0000250.pt"
STAGE_C_MANIFEST = ROOT_DIR / "data" / "manifests" / "stage_c_manifest.json"
STAGE_C_TRAIN = ROOT_DIR / "data" / "tokenized" / "stage_c" / "train"
STAGE_C_VAL = ROOT_DIR / "data" / "tokenized" / "stage_c" / "validation"
STAGE_C_TEST = ROOT_DIR / "data" / "tokenized" / "stage_c" / "test"

EXPECTED_STEP102_HASH = "d5886e02e62df29fc88379ed31b37239c85fcda9648ffaf0cf92191730df9a01"
EXPECTED_STEP103_HASH = "b98bf8b2dac2d06e0e71b813e4049f10baa04b5ed2ea6ad31866435eaf47019b"


@pytest.fixture
def tokenizer():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    return tok


def test_step103_canonical_baseline_bit_exactness():
    """Verify that canonical baseline remains bit-exact throughout Step 103."""
    canonical = instantiate_frozen_baseline()
    param_count = sum(p.numel() for p in canonical.parameters())
    assert param_count == 3_443_136
    h = compute_model_hash(canonical)
    assert h == EXPECTED_WEIGHT_HASH


def test_step103_checkpoints_integrity_and_isolation(tokenizer):
    """Verify that both Step 102 and Step 103 checkpoints exist, have distinct hashes, and load cleanly."""
    assert STEP102_CKPT.is_file(), f"Step 102 checkpoint missing: {STEP102_CKPT}"
    assert STEP103_CKPT.is_file(), f"Step 103 checkpoint missing: {STEP103_CKPT}"

    m102, _, h102 = load_trained_checkpoint(STEP102_CKPT, allow_baseline=True)
    m103, _, h103 = load_trained_checkpoint(STEP103_CKPT, allow_baseline=True)

    assert h102 == EXPECTED_STEP102_HASH
    assert h103 == EXPECTED_STEP103_HASH
    assert h102 != EXPECTED_WEIGHT_HASH
    assert h103 != EXPECTED_WEIGHT_HASH
    assert h102 != h103

    # Verify read-only inference invariant on both
    contract102 = NeuralInferenceContract.from_checkpoint(STEP102_CKPT, tokenizer=tokenizer)
    contract103 = NeuralInferenceContract.from_checkpoint(STEP103_CKPT, tokenizer=tokenizer)

    payload = NeuralInferencePayload(
        prompt="def process(x):\n   ",
        config=NeuralInferenceConfig(max_new_tokens=4, temperature=0.0),
    )
    out102 = contract102.generate(payload)
    out103 = contract103.generate(payload)

    assert out102.weight_hash_verified is True
    assert out103.weight_hash_verified is True
    assert contract102.compute_weight_hash() == EXPECTED_STEP102_HASH
    assert contract103.compute_weight_hash() == EXPECTED_STEP103_HASH


def test_step103_pre_stage_c_prerequisites():
    """Verify that all Stage C artifacts, shards, and manifests required for Step 104 exist and are valid."""
    assert STAGE_C_MANIFEST.is_file()
    assert STAGE_C_TRAIN.is_dir()
    assert STAGE_C_VAL.is_dir()
    assert STAGE_C_TEST.is_dir()

    train_shards = list(STAGE_C_TRAIN.glob("*.bin"))
    val_shards = list(STAGE_C_VAL.glob("*.bin"))
    test_shards = list(STAGE_C_TEST.glob("*.bin"))

    assert len(train_shards) >= 20, f"Expected >= 20 Stage C train shards, found {len(train_shards)}"
    assert len(val_shards) >= 2, f"Expected >= 2 Stage C val shards, found {len(val_shards)}"
    assert len(test_shards) >= 3, f"Expected >= 3 Stage C test shards, found {len(test_shards)}"
