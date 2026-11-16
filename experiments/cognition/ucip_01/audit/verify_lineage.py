import os
import sys
import json
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from chakrview.brain.model import ChakrMicro, ModelConfig

def get_architecture_init_hash():
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

def get_file_sha256(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

artifacts = {
    "canonical_architecture_init": {
        "expected": "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da",
        "actual": get_architecture_init_hash()
    },
    "canonical_stage_b": {
        "path": "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
        "expected": "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    },
    "frozen_cr16_benchmark": {
        "path": "experiments/cognition/cr_exp_16_algorithmic_pretraining/benchmark_data/cr16_synthetic_benchmark_1000.json",
        "expected": "0b61e73d57d903f34dc789736ecd9957c5633c1018fc8c8264bd1e1cb73cac97"
    },
    "protected_runtime_session": {
        "path": "chakrview/runtime/session.py",
        "expected": "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
    },
    "parent_cbr_sft_04": {
        "path": "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt",
        "expected": "327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153"
    },
    "v0.9.1_release": {
        "path": "releases/v0.9.1_conversational_preview/chakrview_v091_model.pt",
        "expected": "f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10"
    },
    "v0.10_release": {
        "path": "releases/v0.10_conversational_model/chakrview_v010_model.pt",
        "expected": "9815f235996934c9c8de4772a2ecd98db1b3b9f5f4d70e5796d3530e4a6fe2ba"
    },
    "v0.10.1_release": {
        "path": "releases/v0.10.1_conversational_model/chakrview_v0101_model.pt",
        "expected": "548b20a73d3cb3a662bcbfe74ab4f108d7bfdc4c60e56abd76049b260fa16a3b"
    },
    "v0.11_release": {
        "path": "releases/v0.11_conversational_model/chakrview_v011_model.pt",
        "expected": "5ae98b369b08373df1222364d1b6165493b34287baf297ba32320c16b76f2754"
    }
}

lineage_results = {}
all_passed = True

for key, item in artifacts.items():
    if key == "canonical_architecture_init":
        actual = item["actual"]
    else:
        actual = get_file_sha256(REPO_ROOT / item["path"])
    
    passed = (actual == item["expected"])
    if not passed:
        all_passed = False
    
    lineage_results[key] = {
        "path": item.get("path", "in_memory_seeded_init"),
        "expected_sha256": item["expected"],
        "actual_sha256": actual,
        "status": "PASS" if passed else "FAIL"
    }
    print(f"[{'PASS' if passed else 'FAIL'}] {key}: {actual}")

out_manifest = REPO_ROOT / "experiments/cognition/ucip_01/manifests/artifact_lineage.json"
with open(out_manifest, "w", encoding="utf-8") as f:
    json.dump({"verification_status": "ALL_PASS" if all_passed else "FAIL", "artifacts": lineage_results}, f, indent=2)

print("\nAll lineage checks pass:", all_passed)
