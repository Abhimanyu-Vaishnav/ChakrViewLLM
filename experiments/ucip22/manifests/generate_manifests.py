"""
UCIP-22 Phase 0 Lineage and Manifest Generator.
Verifies all protected runtime invariants, checkpoint hashes, and dataset manifests,
writing an immutable manifest to experiments/ucip22/manifests/input_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_phase_0(manifest_dir: Path) -> Dict[str, Any]:
    artifacts = {
        "protected_runtime": {
            "path": ROOT / "chakrview/runtime/session.py",
            "expected_sha256": "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67",
        },
        "baseline_step800": {
            "path": ROOT / "experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt",
            "expected_sha256": "8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943",
        },
        "ucip20_best": {
            "path": ROOT / "experiments/ucip20/checkpoints/checkpoint_best.pt",
            "expected_sha256": "0b4b745e923188c6320c8c7290031bff665786c4f4c1c71a74484bcaec6b1a27",
        },
        "bind_train_data": {
            "path": ROOT / "data/sft/ucip20_bind/bind_train_1600.jsonl",
            "expected_sha256": "4b376b0fbf46eac6060ec7d881f6de73be19087c9717847e91eb5877b4972cd6",
        },
        "bind_val_data": {
            "path": ROOT / "data/sft/ucip20_bind/bind_val_200.jsonl",
            "expected_sha256": "3de0e86fb34b0a1b814b8485432574b6df9760a485e59d53f2c09e9e433cc432",
        },
        "bind_dev_data": {
            "path": ROOT / "data/sft/ucip20_bind/bind_dev_200.jsonl",
            "expected_sha256": "698b21208f9f1484e2112eee08f1c9c6ead3d7e9f004abb9b3c6d1d01c52fa62",
        },
        "bind_master_data": {
            "path": ROOT / "data/sft/ucip20_bind/bind_master_2000.jsonl",
            "expected_sha256": "ec7cfab1262158891e8704a7f11740e5f9a8bb81362dba9bc9f734b04bc7e1e9",
        },
    }

    manifest = {"timestamp": "2026-11-21T13:22:00+05:30", "artifacts": {}, "all_verified": True}
    for name, item in artifacts.items():
        p = item["path"]
        actual_hash = sha256_file(p)
        exp = item["expected_sha256"]
        matched = (actual_hash == exp)
        manifest["artifacts"][name] = {
            "path": str(p),
            "expected_sha256": exp,
            "actual_sha256": actual_hash,
            "status": "MATCH" if matched else "MISMATCH",
        }
        if not matched:
            manifest["all_verified"] = False
            raise RuntimeError(f"Hash mismatch on {name}!")

    out_file = manifest_dir / "input_manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("UCIP-22 Phase 0 Verification Complete: All artifacts verified bit-exact.")
    return manifest


if __name__ == "__main__":
    manifest_dir = ROOT / "experiments/ucip22/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    run_phase_0(manifest_dir)
