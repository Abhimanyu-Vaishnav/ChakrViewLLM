import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

ARTIFACTS = {
    "cbr01_seed42": {
        "path": "data/experiments/cbr01_seed42/checkpoints/best_val_checkpoint.pt",
        "expected_sha": "c57ec9d30c5e3152d194fc6b63ca5270170a4c2847fb60ab3523ee4e92a8385a"
    },
    "parent_sft_step200": {
        "path": "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt",
        "expected_sha": "327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153"
    },
    "conversational_pilot": {
        "path": "experiments/cognition/autonomous_12h/conversational_pilot/checkpoints/conversational_pilot_best_val.pt",
        "expected_sha": "ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd"
    },
    "cr06_arm_a": {
        "path": "experiments/cognition/cr_exp_06_variable_binding/checkpoints/arm_a/cr_exp_06_arm_a_best_val.pt",
        "expected_sha": "b2f6ef859ffec448202d0cf3f41249fa6f9efc6fb9f338b70f08e5623087224d"
    },
    "cr06_arm_b": {
        "path": "experiments/cognition/cr_exp_06_variable_binding/checkpoints/arm_b/cr_exp_06_arm_b_best_val.pt",
        "expected_sha": "34720e3f88d7455d6fcbbaab32c024d2d47da15c71b05ff94aa44b7d0c8d1973"
    },
    "cr08_arm_a": {
        "path": "experiments/cognition/cr_exp_08_contextual_assignment/checkpoints/arm_a/cr_exp_08_arm_a_best_val.pt",
        "expected_sha": "0b435ff5e0b6d21389823f0fbba17c5bfaeeb501fc55e5b87ccafec3b9c2b3ad"
    },
    "cr08_arm_b": {
        "path": "experiments/cognition/cr_exp_08_contextual_assignment/checkpoints/arm_b/cr_exp_08_arm_b_best_val.pt",
        "expected_sha": "98bd76f99d43dedd10c31a1d290c95e452eba299f511ecf72e5ecc4075fb0956"
    },
    "cr08_arm_c": {
        "path": "experiments/cognition/cr_exp_08_contextual_assignment/checkpoints/arm_c/cr_exp_08_arm_c_best_val.pt",
        "expected_sha": "a3b04c86ec0b8e217dfa74d284fb7fb5cf2873130d20dcf7569e5d4a1329c4ba"
    },
    "cr09_baseline_copy": {
        "path": "experiments/cognition/cr_exp_09_autonomous_mission/checkpoints/baseline/cr_exp_08_arm_b_best_val_copy.pt",
        "expected_sha": "98bd76f99d43dedd10c31a1d290c95e452eba299f511ecf72e5ecc4075fb0956"
    },
    "cr09_candidate_1": {
        "path": "experiments/cognition/cr_exp_09_autonomous_mission/checkpoints/candidate_1/cr_exp_09_candidate_1_best_val.pt",
        "expected_sha": "3134918e7784654bdf1356828ce9f835cb4b71ff4b62db5606dcfceb3e8e7a0d"
    },
    "cr09_candidate_2": {
        "path": "experiments/cognition/cr_exp_09_autonomous_mission/checkpoints/candidate_2/cr_exp_09_candidate_2_best_val.pt",
        "expected_sha": "8e3a2469eb0702ca7452d7ee8b0304a0e949ffec151b75c879d74e0d8fa09995"
    },
    "cr09_candidate_3": {
        "path": "experiments/cognition/cr_exp_09_autonomous_mission/checkpoints/candidate_3/cr_exp_09_candidate_3_best_val.pt",
        "expected_sha": "de871613aa28264014f22e227beea8976c359dca6f7026ed4221703ac7a3e592"
    },
    "vocab_json": {
        "path": "data/experiments/vocab_4096/vocab.json",
        "expected_sha": "bf2d4c064e6f0b6e1ee7425176b668d2f16a19f20108ff4ceaa0be5c94d65012"
    },
    "merges_txt": {
        "path": "data/experiments/vocab_4096/merges.txt",
        "expected_sha": "7f09f196ebcf7951a87610fd67807b58532fa7dd4ec9ebae7aa97d394d03e911"
    },
    "session_runtime": {
        "path": "chakrview/runtime/session.py",
        "expected_sha": "a1ec68d18408a0d0d8bbec287e076634c00be05e55e8248ab41b0728c738e4a9"
    }
}

def verify():
    results = {}
    all_matched = True
    for key, spec in ARTIFACTS.items():
        p = REPO_ROOT / spec["path"]
        if not p.exists():
            results[key] = {"exists": False, "status": "MISSING", "path": str(spec["path"])}
            all_matched = False
            continue
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        matched = (actual == spec["expected_sha"])
        if not matched:
            all_matched = False
        results[key] = {
            "exists": True,
            "status": "VERIFIED" if matched else "MISMATCH",
            "path": str(spec["path"]),
            "expected_sha": spec["expected_sha"],
            "actual_sha": actual,
            "size_bytes": p.stat().st_size
        }
        print(f"[{'PASS' if matched else 'FAIL'}] {key:20s}: {results[key]['status']} ({actual[:16]}...)")
    
    out_path = REPO_ROOT / "experiments/cognition/cr_exp_10_readiness/checkpoint_lineage_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"all_matched": all_matched, "artifacts": results}, f, indent=2)
    print(f"\nAudit completed. Saved to {out_path}. All verified: {all_matched}")

if __name__ == "__main__":
    verify()
