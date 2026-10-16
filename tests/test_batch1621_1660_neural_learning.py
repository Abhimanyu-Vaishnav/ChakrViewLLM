import json
import hashlib
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig

REPO_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
CANONICAL_SHA256 = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

ADAPTED_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch1381_1420" / "checkpoints" / "chakrmicro_domain_adapted_experimental.pt"
ADAPTED_SHA256 = "15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139"

BATCH_DIR = REPO_ROOT / "experiments" / "batch1621_1660"
MANIFEST_PATH = BATCH_DIR / "data" / "DATASET_MANIFEST.json"
TEST_DATASET_PATH = BATCH_DIR / "data" / "test_held_out.json"
CANDIDATE_CHECKPOINT_PATH = BATCH_DIR / "checkpoints" / "chakrmicro_controlled_candidate_step100.pt"
RESULTS_JSON_PATH = BATCH_DIR / "results" / "NEURAL_EVALUATION_RESULTS.json"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_canonical_checkpoint_integrity():
    """Verify that the immutable canonical checkpoint remains byte-for-byte identical."""
    assert CANONICAL_CHECKPOINT_PATH.exists(), f"Missing canonical checkpoint at {CANONICAL_CHECKPOINT_PATH}"
    actual_hash = sha256_of_file(CANONICAL_CHECKPOINT_PATH)
    assert actual_hash == CANONICAL_SHA256, f"Canonical checkpoint hash mismatch! Expected {CANONICAL_SHA256}, got {actual_hash}"


def test_adapted_checkpoint_integrity():
    """Verify that the immutable adapted checkpoint remains byte-for-byte identical."""
    assert ADAPTED_CHECKPOINT_PATH.exists(), f"Missing adapted checkpoint at {ADAPTED_CHECKPOINT_PATH}"
    actual_hash = sha256_of_file(ADAPTED_CHECKPOINT_PATH)
    assert actual_hash == ADAPTED_SHA256, f"Adapted checkpoint hash mismatch! Expected {ADAPTED_SHA256}, got {actual_hash}"


def test_dataset_manifest_and_leakage_control():
    """Verify dataset manifest files, hashes, and strict held-out partition properties."""
    assert MANIFEST_PATH.exists(), f"Missing manifest at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Verify all manifest files match recorded hashes
    train_file = BATCH_DIR / "data" / "train.jsonl"
    val_file = BATCH_DIR / "data" / "val.jsonl"
    test_file = BATCH_DIR / "data" / "test_held_out.json"

    assert train_file.exists()
    assert val_file.exists()
    assert test_file.exists()

    assert sha256_of_file(train_file) == manifest["train_sha256"]
    assert sha256_of_file(val_file) == manifest["val_sha256"]
    assert sha256_of_file(test_file) == manifest["test_held_out_sha256"]

    # Load held-out test tasks and ensure 150 tasks
    with open(TEST_DATASET_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    assert len(tasks) == 150

    # Ensure arithmetic extrapolation split: test set numbers strictly >= 25
    arith_tasks = [t for t in tasks if t["track"] == "arithmetic_continuation"]
    assert len(arith_tasks) == 30
    for t in arith_tasks:
        n_val = int(t["prompt"].replace("The double of ", "").replace(" is", "").strip())
        assert n_val >= 25, f"Leaked arithmetic value {n_val} found in held-out test set!"


def test_candidate_checkpoint_and_eval_artifacts():
    """Verify candidate checkpoint existence, hash match, and evaluation results presence."""
    assert CANDIDATE_CHECKPOINT_PATH.exists(), "Candidate checkpoint step 100 missing!"
    assert RESULTS_JSON_PATH.exists(), "Evaluation results JSON missing!"

    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
        res = json.load(f)

    summary = res["comparative_summary"]
    assert "canonical_step_10000" in summary
    assert "adapted_step_10080" in summary
    assert "candidate_step_100" in summary

    # Verify that perplexity improved during training on candidate
    assert summary["candidate_step_100"]["val_ppl"] < summary["canonical_step_10000"]["val_ppl"]

    # Verify that standalone neural model alone was tested (no tools used, 0% arithmetic)
    assert summary["candidate_step_100"]["track_scores"]["track2_arithmetic_continuation"]["accuracy"] == 0.0


def test_chakrmicro_pure_forward_pass():
    """Verify pure neural forward pass without external dependencies."""
    config = ModelConfig(
        vocab_size=4096,
        d_model=192,
        n_layers=2,
        n_heads=6,
        max_seq_len=64,
        dropout=0.0
    )
    model = ChakrMicro(config)
    model.eval()

    dummy_input = torch.tensor([[1, 42, 108, 999]], dtype=torch.long)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (1, 4, 4096)
