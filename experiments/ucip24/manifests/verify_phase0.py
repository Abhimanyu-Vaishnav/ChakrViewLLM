"""
UCIP-24 Phase 0 Verification Script:
Verifies Git HEAD, branch, protected runtime SHA-256, Step-800 checkpoint SHA-256,
UCIP-20 checkpoint SHA-256, and all UCIP-23 dataset/manifest SHA-256 hashes against
the recorded manifests.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

EXPECTED_HEAD = "8594858fad781d6ad724d3c39c2d5a857f02cfce"
EXPECTED_RUNTIME_SHA256 = "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
EXPECTED_STEP800_SHA256 = "8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943"
EXPECTED_UCIP20_SHA256 = "0b4b745e923188c6320c8c7290031bff665786c4f4c1c71a74484bcaec6b1a27"


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_phase0():
    results = {}

    # 1. Git HEAD
    res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    actual_head = res.stdout.strip()
    results["git_head"] = {
        "expected": EXPECTED_HEAD,
        "actual": actual_head,
        "match": actual_head == EXPECTED_HEAD
    }

    # 2. Branch
    res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True)
    actual_branch = res.stdout.strip()
    results["git_branch"] = {
        "expected": "exp/cbr-01-conversational-breakthrough",
        "actual": actual_branch,
        "match": actual_branch == "exp/cbr-01-conversational-breakthrough"
    }

    # 3. Protected runtime
    runtime_path = "chakrview/runtime/session.py"
    actual_runtime_sha = compute_sha256(runtime_path)
    results["protected_runtime"] = {
        "path": runtime_path,
        "expected_sha256": EXPECTED_RUNTIME_SHA256,
        "actual_sha256": actual_runtime_sha,
        "match": actual_runtime_sha == EXPECTED_RUNTIME_SHA256
    }

    # 4. Checkpoints
    step800_path = "experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt"
    actual_step800_sha = compute_sha256(step800_path)
    results["step800_checkpoint"] = {
        "path": step800_path,
        "expected_sha256": EXPECTED_STEP800_SHA256,
        "actual_sha256": actual_step800_sha,
        "match": actual_step800_sha == EXPECTED_STEP800_SHA256
    }

    ucip20_path = "experiments/ucip20/checkpoints/checkpoint_best.pt"
    actual_ucip20_sha = compute_sha256(ucip20_path)
    results["ucip20_best_checkpoint"] = {
        "path": ucip20_path,
        "expected_sha256": EXPECTED_UCIP20_SHA256,
        "actual_sha256": actual_ucip20_sha,
        "match": actual_ucip20_sha == EXPECTED_UCIP20_SHA256
    }

    # 5. Output manifest verification of UCIP-23 files
    out_manifest_path = "experiments/ucip23/manifests/output_manifest.json"
    with open(out_manifest_path, "r", encoding="utf-8") as f:
        out_manifest = json.load(f)

    dataset_verifications = {}
    for path, meta in out_manifest["artifacts"]["datasets"].items():
        actual_sha = compute_sha256(path)
        expected_sha = meta["sha256"]
        dataset_verifications[path] = {
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "match": actual_sha == expected_sha
        }
    results["ucip23_datasets"] = dataset_verifications

    manifest_verifications = {}
    for path, meta in out_manifest["artifacts"]["manifests"].items():
        actual_sha = compute_sha256(path)
        expected_sha = meta["sha256"]
        manifest_verifications[path] = {
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "match": actual_sha == expected_sha
        }
    results["ucip23_manifests"] = manifest_verifications

    all_matched = (
        results["git_head"]["match"]
        and results["git_branch"]["match"]
        and results["protected_runtime"]["match"]
        and results["step800_checkpoint"]["match"]
        and results["ucip20_best_checkpoint"]["match"]
        and all(v["match"] for v in dataset_verifications.values())
        and all(v["match"] for v in manifest_verifications.values())
    )

    results["all_matched"] = all_matched

    out_file = "experiments/ucip24/manifests/phase0_verification.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Phase 0 Verification Complete. All matched: {all_matched}")
    assert all_matched, "Phase 0 verification failed! Check discrepancies."


if __name__ == "__main__":
    verify_phase0()
