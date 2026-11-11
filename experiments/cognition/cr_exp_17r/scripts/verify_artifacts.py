import os
import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path("d:/Project/ChakrView")

def get_sha256(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

expected_hashes = {
    "release_checkpoint": {
        "path": REPO_ROOT / "releases/v0.9.1_conversational_preview/chakrview_v091_model.pt",
        "expected": "f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10"
    },
    "best_checkpoint": {
        "path": REPO_ROOT / "experiments/conversation/cr_exp_17/checkpoints/chakrview_v091_conversational_best.pt",
        "expected": "f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10"
    },
    "cr17_benchmark": {
        "path": REPO_ROOT / "experiments/conversation/cr_exp_17/CONVERSATION_BENCHMARK_V1.json",
        "expected": "d2472e0b2a72f2555bea68511ac7373c5f2983f785bb9a7d60da4dd9ac2f1a5f"
    },
    "canonical_stage_b": {
        "path": REPO_ROOT / "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
        "expected": "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
    },
    "frozen_parent_cbr_sft_04": {
        "path": REPO_ROOT / "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt",
        "expected": "327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153"
    },
    "cr16a_arm_d": {
        "path": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/checkpoints/arm_d_generalization/arm_d_generalization_best.pt",
        "expected": "9d609a770c748f9321d06713093620da0fe6caabc04fea2401b37403d33de1a5"
    },
    "protected_session_py": {
        "path": REPO_ROOT / "chakrview/runtime/session.py",
        "expected": "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
    },
    "dataset_train": {
        "path": REPO_ROOT / "experiments/conversation/cr_exp_17/data/conversation_train_1080.jsonl",
        "expected": "91ca4c4ca4dbe4d59fca76865d4ebf07297e6e58dfda5f69c5e2ef975ea8e24c"
    },
    "dataset_val": {
        "path": REPO_ROOT / "experiments/conversation/cr_exp_17/data/conversation_val_120.jsonl",
        "expected": "c4960ce0f9f3bfb5883ea57cb5a909b6d8591f24d770c675fb621375b48db9d0"
    },
    "dataset_manifest": {
        "path": REPO_ROOT / "experiments/conversation/cr_exp_17/data/conversation_corpus_manifest.json",
        "expected": "78ce7ebef371cceea9f002d2bf94aa5df8d9bbf38283a216dbbb746ca72a6b25"
    },
    "release_config": {
        "path": REPO_ROOT / "releases/v0.9.1_conversational_preview/config.json",
        "expected": "7f9dbcfbc0b8755bfaea7a2b9e6727271fa9fbdeaf4777995e8697fa2312d9b6"
    }
}

audit_results = {}
all_pass = True

for key, item in expected_hashes.items():
    actual_hash = get_sha256(item["path"])
    matched = (actual_hash == item["expected"])
    audit_results[key] = {
        "path": str(item["path"].relative_to(REPO_ROOT)),
        "expected_sha256": item["expected"],
        "actual_sha256": actual_hash,
        "match": matched
    }
    if not matched:
        all_pass = False

with open("experiments/cognition/cr_exp_17r/manifests/artifact_hashes.json", "w") as f:
    json.dump(audit_results, f, indent=2)

print(f"Artifact integrity check complete. All matched: {all_pass}")
for k, v in audit_results.items():
    status = "PASS" if v["match"] else "FAIL"
    print(f"[{status}] {k}: {v['actual_sha256']}")
