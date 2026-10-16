import json
import hashlib
from pathlib import Path
import pytest
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from chakrview.runtime.governed_tool_router import GovernedToolRouter

REPO_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
CANONICAL_SHA256 = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

ADAPTED_CHECKPOINT_PATH = REPO_ROOT / "experiments" / "batch1381_1420" / "checkpoints" / "chakrmicro_domain_adapted_experimental.pt"
ADAPTED_SHA256 = "15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139"

BATCH_DIR = REPO_ROOT / "experiments" / "batch1701_1740"
MANIFEST_PATH = BATCH_DIR / "data" / "DATASET_MANIFEST.json"
TEST_DATASET_PATH = BATCH_DIR / "data" / "test_held_out.json"
VARIANT_A_CKPT = BATCH_DIR / "checkpoints" / "variant_a_frozen_backbone.pt"
VARIANT_B_CKPT = BATCH_DIR / "checkpoints" / "variant_b_finetuned_100pct.pt"
RESULTS_JSON_PATH = BATCH_DIR / "results" / "CLASSIFICATION_EVALUATION_RESULTS.json"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_reference_checkpoints_integrity():
    """Verify that both canonical and adapted checkpoints remain byte-for-byte unmodified."""
    assert CANONICAL_CHECKPOINT_PATH.exists()
    assert ADAPTED_CHECKPOINT_PATH.exists()

    assert sha256_of_file(CANONICAL_CHECKPOINT_PATH) == CANONICAL_SHA256
    assert sha256_of_file(ADAPTED_CHECKPOINT_PATH) == ADAPTED_SHA256


def test_scaling_dataset_manifest_and_leakage():
    """Verify dataset manifest files, hashes, class balances, and partition properties."""
    assert MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for part_name, meta in manifest["partitions"].items():
        fname = f"{part_name}.jsonl" if not part_name.endswith(".json") and part_name != "test_held_out" else f"{part_name}.json"
        if part_name == "val":
            fname = "val.jsonl"
        fpath = BATCH_DIR / "data" / fname
        assert fpath.exists(), f"Missing partition file {fpath}"
        assert sha256_of_file(fpath) == meta["sha256"], f"Hash mismatch for {part_name}"

    with open(TEST_DATASET_PATH, "r", encoding="utf-8") as f:
        test_tasks = json.load(f)
    assert len(test_tasks) == 180

    # Ensure 30 tasks per intent
    counts = {}
    for t in test_tasks:
        counts[t["intent"]] = counts.get(t["intent"], 0) + 1
    for intent in manifest["intents"]:
        assert counts[intent] == 30


def test_classification_checkpoints_and_results_artifacts():
    """Verify presence of Variant A and Variant B checkpoints, as well as evaluation metrics."""
    assert VARIANT_A_CKPT.exists()
    assert VARIANT_B_CKPT.exists()
    assert RESULTS_JSON_PATH.exists()

    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
        res = json.load(f)

    comp = res["comparative_summary"]
    assert "variant_a_frozen_backbone" in comp
    assert "variant_b_finetuned_100pct" in comp
    assert "variant_b_finetuned_50pct" in comp
    assert "variant_b_finetuned_25pct" in comp
    assert "keyword_baseline" in comp
    assert "governed_deterministic_router" in comp
    assert "governed_neuro_symbolic_hybrid" in comp

    # Verify that Variant B outperforms Keyword Baseline on held-out test
    assert comp["variant_b_finetuned_100pct"]["intent_accuracy"] > comp["keyword_baseline"]["intent_accuracy"]

    # Verify positive data scaling: 100% > 50% > 25%
    acc_100 = comp["variant_b_finetuned_100pct"]["intent_accuracy"]
    acc_50 = comp["variant_b_finetuned_50pct"]["intent_accuracy"]
    acc_25 = comp["variant_b_finetuned_25pct"]["intent_accuracy"]
    assert acc_100 > acc_50 > acc_25, f"Expected monotonic data scaling, got {acc_100} > {acc_50} > {acc_25}"

    # Verify that Governed Hybrid has <= 1.0% unsafe execution rate
    assert comp["governed_neuro_symbolic_hybrid"]["unsafe_execution_rate"] <= 0.01

    # Verify that latency is under 10 ms on CPU
    assert comp["variant_b_finetuned_100pct"]["avg_latency_ms"] < 10.0


def test_classifier_head_pure_forward_pass():
    """Verify single-pass inference through ChakrMicro with classification head."""
    config = ModelConfig(vocab_size=4096, d_model=192, n_layers=2, n_heads=6, max_seq_len=64)
    model = ChakrMicro(config)
    head = torch.nn.Linear(192, 6)

    dummy_input = torch.tensor([[1, 42, 108, 999]], dtype=torch.long)
    _, h = model(dummy_input, return_hidden_states=True)
    pooled = h[:, -1, :]
    logits = head(pooled)
    assert logits.shape == (1, 6)
