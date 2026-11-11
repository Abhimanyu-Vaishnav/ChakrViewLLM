"""
Audit A: Checkpoint Lineage & Integrity Verification Script
Computes exact SHA-256 digests for all 7 target checkpoints/artifacts,
inspects metadata, compares parameter counts and weight differences.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import compute_model_hash, EXPECTED_WEIGHT_HASH
from chakrview.cognition.release_manifest_verifier import EXPECTED_TOKENIZER_HASHES

EXPECTED = {
    "parent": {
        "path": "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt",
        "expected_sha256": "327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153"
    },
    "candidate_pilot": {
        "path": "experiments/cognition/autonomous_12h/conversational_pilot/checkpoints/conversational_pilot_best_val.pt",
        "expected_sha256": "ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd"
    },
    "control_arm_best_val": {
        "path": "experiments/cognition/cr_exp_06_variable_binding/control_arm/checkpoints/control_arm_best_val.pt",
        "expected_sha256": "729d5812c1ec02d217ef8cc342453a008ab3046a550e627b16b341c4a458c21f"
    },
    "control_arm_final_step_150": {
        "path": "experiments/cognition/cr_exp_06_variable_binding/control_arm/checkpoints/control_arm_final_step_150.pt",
        "expected_sha256": "d67e943906149d18637f200a7eb5cdc89340b989d17c9cd5668df94a17ba3184"
    },
    "treatment_arm_best_val": {
        "path": "experiments/cognition/cr_exp_06_variable_binding/treatment_arm/checkpoints/treatment_arm_best_val.pt",
        "expected_sha256": "2d8cea93ba8dbe06e1c714d5563ad89b18600f7a4f766fbb7cfc563c947a9d90"
    },
    "treatment_arm_final_step_150": {
        "path": "experiments/cognition/cr_exp_06_variable_binding/treatment_arm/checkpoints/treatment_arm_final_step_150.pt",
        "expected_sha256": "dac7517e2532d42d684237216bd00ddb57991050eaacac77380d783c13723b33"
    },
    "tokenizer_vocab": {
        "path": "data/experiments/vocab_4096/vocab.json",
        "expected_sha256": EXPECTED_TOKENIZER_HASHES["vocab.json"]
    },
    "tokenizer_merges": {
        "path": "data/experiments/vocab_4096/merges.json",
        "expected_sha256": EXPECTED_TOKENIZER_HASHES["merges.json"]
    },
    "tokenizer_config": {
        "path": "data/experiments/vocab_4096/config.json",
        "expected_sha256": EXPECTED_TOKENIZER_HASHES["config.json"]
    },
    "runtime_session": {
        "path": "chakrview/runtime/session.py",
        "expected_sha256": "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
    }
}

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== Audit A: Checkpoint Lineage & Integrity Verification ===")
    results = {}

    for k, info in EXPECTED.items():
        p = REPO_ROOT / info["path"]
        exists = p.exists()
        h = sha256_file(p) if exists else None
        match = (h == info["expected_sha256"])
        size_bytes = p.stat().st_size if exists else 0
        results[k] = {
            "path": str(p),
            "exists": exists,
            "size_bytes": size_bytes,
            "sha256": h,
            "expected_sha256": info["expected_sha256"],
            "match": match
        }
        print(f"  {k:28s}: {'MATCH' if match else 'MISMATCH'} ({str(h)[:16]}...)")

    # Canonical baseline
    torch.manual_seed(42)
    canonical_model = ChakrMicro(ModelConfig())
    canonical_model.eval()
    canon_hash = compute_model_hash(canonical_model)
    canon_match = (canon_hash == EXPECTED_WEIGHT_HASH)
    results["canonical_baseline"] = {
        "sha256": canon_hash,
        "expected_sha256": EXPECTED_WEIGHT_HASH,
        "match": canon_match
    }
    print(f"  canonical_baseline          : {'MATCH' if canon_match else 'MISMATCH'} ({canon_hash[:16]}...)")

    # Load and inspect checkpoint metadata
    ckpt_ctrl = torch.load(REPO_ROOT / EXPECTED["control_arm_best_val"]["path"], map_location="cpu", weights_only=False)
    ckpt_treat = torch.load(REPO_ROOT / EXPECTED["treatment_arm_best_val"]["path"], map_location="cpu", weights_only=False)
    ckpt_pilot = torch.load(REPO_ROOT / EXPECTED["candidate_pilot"]["path"], map_location="cpu", weights_only=False)

    metadata_audit = {
        "control_arm_metadata": {
            "step": ckpt_ctrl.get("step"),
            "val_loss": ckpt_ctrl.get("val_loss"),
            "train_loss": ckpt_ctrl.get("train_loss"),
            "config": ckpt_ctrl.get("config")
        },
        "treatment_arm_metadata": {
            "step": ckpt_treat.get("step"),
            "val_loss": ckpt_treat.get("val_loss"),
            "train_loss": ckpt_treat.get("train_loss"),
            "config": ckpt_treat.get("config")
        },
        "initialization_parent_match": (
            ckpt_ctrl.get("config", {}).get("parent_sha256") == EXPECTED["candidate_pilot"]["expected_sha256"]
            and ckpt_treat.get("config", {}).get("parent_sha256") == EXPECTED["candidate_pilot"]["expected_sha256"]
        ),
        "total_model_parameters": sum(p.numel() for p in canonical_model.parameters())
    }

    # Weight difference between control and treatment candidates
    l2_diff = 0.0
    for name, p_ctrl in ckpt_ctrl["model_state_dict"].items():
        p_treat = ckpt_treat["model_state_dict"][name]
        l2_diff += (p_ctrl - p_treat).pow(2).sum().item()
    l2_diff = l2_diff ** 0.5
    metadata_audit["l2_distance_control_vs_treatment"] = round(l2_diff, 6)

    # Weight distance from pilot initialization
    l2_ctrl_from_pilot = 0.0
    l2_treat_from_pilot = 0.0
    for name, p_init in ckpt_pilot["model_state_dict"].items():
        p_ctrl = ckpt_ctrl["model_state_dict"][name]
        p_treat = ckpt_treat["model_state_dict"][name]
        l2_ctrl_from_pilot += (p_ctrl - p_init).pow(2).sum().item()
        l2_treat_from_pilot += (p_treat - p_init).pow(2).sum().item()

    metadata_audit["l2_distance_control_from_pilot"] = round(l2_ctrl_from_pilot ** 0.5, 6)
    metadata_audit["l2_distance_treatment_from_pilot"] = round(l2_treat_from_pilot ** 0.5, 6)

    output = {
        "artifact_hashes": results,
        "metadata_and_lineage": metadata_audit,
        "lineage_audit_passed": all(v["match"] for v in results.values()) and metadata_audit["initialization_parent_match"]
    }

    out_file = REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/audit/checkpoint_lineage_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nAudit A results written to {out_file}")
    print(f"Lineage verification overall: {'PASSED' if output['lineage_audit_passed'] else 'FAILED'}")

if __name__ == "__main__":
    main()

