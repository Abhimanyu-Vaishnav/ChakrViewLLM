import pytest
import json
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent

def test_canonical_artifact_immutability():
    expected = {
        "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt": "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee",
        "data/experiments/vocab_4096/vocab.json": "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a",
        "data/experiments/vocab_4096/merges.json": "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7",
        "chakrview/runtime/session.py": "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
    }
    for rel_path, exp_hash in expected.items():
        p = REPO_ROOT / rel_path
        assert p.exists(), f"Missing file: {p}"
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == exp_hash, f"Hash mismatch on {rel_path}: {h} vs {exp_hash}"

def test_dataset_split_leakage_and_coverage():
    manifest_file = EXP_DIR / "DATASET_MANIFEST.json"
    assert manifest_file.exists()
    
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["splits"]["train"]["count"] == 180
    assert manifest["splits"]["val"]["count"] == 60
    assert manifest["splits"]["test"]["count"] == 60
    
    train_prompts = set()
    with open(EXP_DIR / "data" / "train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            train_prompts.add(json.loads(line)["prompt"])
            
    val_prompts = set()
    with open(EXP_DIR / "data" / "val.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            val_prompts.add(json.loads(line)["prompt"])
            
    test_prompts = set()
    with open(EXP_DIR / "data" / "test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            test_prompts.add(json.loads(line)["prompt"])
            
    assert len(train_prompts.intersection(val_prompts)) == 0
    assert len(train_prompts.intersection(test_prompts)) == 0
    assert len(val_prompts.intersection(test_prompts)) == 0

def test_checkpoint_and_history_reproducibility():
    best_ckpt = EXP_DIR / "checkpoints" / "candidate_best.pt"
    assert best_ckpt.exists()
    
    history_file = EXP_DIR / "checkpoints" / "training_history.json"
    assert history_file.exists()
    
    history = json.loads(history_file.read_text(encoding="utf-8"))
    assert len(history) == 6  # 20, 40, 60, 80, 100, 120 steps
    assert history[0]["train_loss"] > history[-1]["train_loss"]
    assert history[-1]["drift_pct"] <= 4.0  # within drift budget

def test_evaluation_reproducibility():
    baseline_res_file = EXP_DIR / "BASELINE_RESULTS.json"
    candidate_res_file = EXP_DIR / "CANDIDATE_RESULTS.json"
    per_example_file = EXP_DIR / "PER_EXAMPLE_RESULTS.jsonl"
    
    assert baseline_res_file.exists()
    assert candidate_res_file.exists()
    assert per_example_file.exists()
    
    b_res = json.loads(baseline_res_file.read_text(encoding="utf-8"))
    c_res = json.loads(candidate_res_file.read_text(encoding="utf-8"))
    
    assert b_res["total_examples"] == 60
    assert c_res["total_examples"] == 60
    assert c_res["relevant_pct"] > b_res["relevant_pct"]  # Candidate improved over baseline

def test_self_correction_traces():
    audit_file = EXP_DIR / "SELF_CORRECTION_AUDIT.jsonl"
    assert audit_file.exists()
    
    traces = []
    with open(audit_file, "r", encoding="utf-8") as f:
        for line in f:
            traces.append(json.loads(line))
            
    assert len(traces) == 13
    accepted = [t for t in traces if t["status"] == "ACCEPTED_FOR_TRAINING"]
    assert len(accepted) == 3
    for a in accepted:
        assert a["is_verified"] is True
        assert a["verification_method"] == "PYTHON_AST_PARSER"
