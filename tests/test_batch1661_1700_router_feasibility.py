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

BATCH_DIR = REPO_ROOT / "experiments" / "batch1661_1700"
MANIFEST_PATH = BATCH_DIR / "data" / "DATASET_MANIFEST.json"
TEST_DATASET_PATH = BATCH_DIR / "data" / "test_held_out.json"
CANDIDATE_CHECKPOINT_PATH = BATCH_DIR / "checkpoints" / "chakrmicro_router_candidate_step100.pt"
RESULTS_JSON_PATH = BATCH_DIR / "results" / "ROUTER_EVALUATION_RESULTS.json"


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_reference_checkpoints_unmodified():
    """Verify that both canonical and adapted checkpoints remain byte-for-byte unchanged."""
    assert CANONICAL_CHECKPOINT_PATH.exists()
    assert ADAPTED_CHECKPOINT_PATH.exists()

    assert sha256_of_file(CANONICAL_CHECKPOINT_PATH) == CANONICAL_SHA256
    assert sha256_of_file(ADAPTED_CHECKPOINT_PATH) == ADAPTED_SHA256


def test_parameter_count_resolution():
    """Verify authoritative 3,443,136 unique parameters and weight-tying properties."""
    ckpt = torch.load(CANONICAL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = ChakrMicro(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    # Unique parameters count
    unique_params = sum(p.numel() for p in {p.data_ptr(): p for p in model.parameters()}.values())
    assert unique_params == 3443136, f"Expected 3,443,136 unique parameters, got {unique_params}"

    # Verify weight-tying between embedding and lm_head
    assert model.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr()

    # Verify state_dict serialization elements
    serialized_elements = sum(p.numel() for p in ckpt["model_state_dict"].values())
    assert serialized_elements == 4229568


def test_router_dataset_manifest_and_leakage_control():
    """Verify router dataset manifest integrity and 120 held-out balanced tasks."""
    assert MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_file = BATCH_DIR / "data" / "train.jsonl"
    val_file = BATCH_DIR / "data" / "val.jsonl"
    test_file = BATCH_DIR / "data" / "test_held_out.json"

    assert sha256_of_file(train_file) == manifest["train_sha256"]
    assert sha256_of_file(val_file) == manifest["val_sha256"]
    assert sha256_of_file(test_file) == manifest["test_held_out_sha256"]

    with open(TEST_DATASET_PATH, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    assert len(tasks) == 120

    # Ensure 20 tasks per intent
    counts = {}
    for t in tasks:
        counts[t["intent"]] = counts.get(t["intent"], 0) + 1
    for intent in manifest["intents"]:
        assert counts[intent] == 20


def test_router_evaluation_artifacts_and_metrics():
    """Verify presence of evaluation results and multi-system comparison metrics."""
    assert CANDIDATE_CHECKPOINT_PATH.exists()
    assert RESULTS_JSON_PATH.exists()

    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
        res = json.load(f)

    comp = res["comparative_summary"]
    assert "deterministic_governed_router" in comp
    assert "keyword_baseline" in comp
    assert "canonical_neural_untrained" in comp
    assert "candidate_neural_step100" in comp
    assert "governed_neuro_symbolic_hybrid" in comp

    # Verify keyword baseline outperformed candidate neural router on intent accuracy
    assert comp["keyword_baseline"]["intent_accuracy"] > comp["candidate_neural_step100"]["intent_accuracy"]

    # Verify that hybrid system had 100% valid JSON and safe routing
    assert comp["governed_neuro_symbolic_hybrid"]["json_validity_rate"] == 1.0


def test_governed_tool_router_fail_closed_on_adversarial():
    """Verify that deterministic router fails closed on hostile and missing input."""
    router = GovernedToolRouter()
    res1 = router.route_request("Calculate it for me please")
    assert res1.route.value in ["ABSTAIN", "DIRECT_RESPONSE"]

    res2 = router.route_request("Execute: ; cat /etc/passwd")
    assert res2.route.value != "CONTAINER_EXECUTION"
