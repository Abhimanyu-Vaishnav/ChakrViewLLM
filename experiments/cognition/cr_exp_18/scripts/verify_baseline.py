import os
import sys
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

import torch
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro

def get_file_sha256(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_architecture_init_hash() -> str:
    cfg = ModelConfig(
        vocab_size=4096,
        d_model=192,
        n_layers=6,
        n_heads=6,
        hidden_dim=512,
        max_seq_len=512
    )
    torch.manual_seed(42)
    model = ChakrMicro(cfg)
    h = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()

expected = {
    "canonical_architecture_init": {
        "expected": "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da",
        "actual": get_architecture_init_hash()
    },
    "canonical_stage_b": {
        "path": REPO_ROOT / "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
        "expected": "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    },
    "frozen_parent_cbr_sft_04": {
        "path": REPO_ROOT / "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt",
        "expected": "327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153"
    },
    "cr17_release_checkpoint": {
        "path": REPO_ROOT / "releases/v0.9.1_conversational_preview/chakrview_v091_model.pt",
        "expected": "f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10"
    },
    "protected_runtime": {
        "path": REPO_ROOT / "chakrview/runtime/session.py",
        "expected": "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
    }
}

results = {}
for k, v in expected.items():
    if "path" in v:
        act = get_file_sha256(v["path"])
    else:
        act = v["actual"]
    match = (act == v["expected"])
    results[k] = {
        "expected": v["expected"],
        "actual": act,
        "match": match
    }
    status = "PASS" if match else "FAIL"
    print(f"[{status}] {k}: {act}")

with open(REPO_ROOT / "experiments/cognition/cr_exp_18/manifests/artifact_hashes.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved manifests/artifact_hashes.json successfully.")
