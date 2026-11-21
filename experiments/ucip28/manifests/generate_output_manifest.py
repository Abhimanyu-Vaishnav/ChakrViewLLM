"""
Generates the cryptographic output manifest for UCIP-28.
"""

from __future__ import annotations
import glob
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def compute_sha256(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest():
    search_dirs = [
        "experiments/ucip28/layer_ablation/*",
        "experiments/ucip28/matched_controls/*",
        "experiments/ucip28/ffn_channels/*",
        "experiments/ucip28/context_attribution/*",
        "experiments/ucip28/activation_patching/*",
        "experiments/ucip28/manifests/*",
        "experiments/ucip28/reports/*",
        "experiments/ucip28/tests/*"
    ]
    all_files = []
    for pattern in search_dirs:
        for p in glob.glob(pattern):
            path_obj = Path(p)
            if path_obj.is_file() and not p.endswith("output_manifest.json"):
                all_files.append(str(path_obj).replace("\\", "/"))

    all_files.sort()
    manifest = {
        "manifest_version": "ucip-28-v1",
        "total_files": len(all_files),
        "files": {f: compute_sha256(f) for f in all_files}
    }

    out_path = "experiments/ucip28/manifests/output_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written with {len(all_files)} files.")


if __name__ == "__main__":
    generate_manifest()
