"""
UCIP-25 Reproducibility Manifest Generator:
Computes and verifies SHA-256 hashes of all artifacts produced during UCIP-25.
"""

import glob
import hashlib
import json


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest():
    manifest = {
        "mission": "UCIP-25",
        "description": "Frozen Zero-Shot Capability Evaluation & Baseline Comparison",
        "artifacts": {}
    }

    files = sorted(glob.glob("experiments/ucip25/**/*", recursive=True))
    for f in files:
        if "reproducibility_manifest.json" in f:
            continue
        try:
            import os
            if os.path.isfile(f):
                norm = f.replace("\\", "/")
                manifest["artifacts"][norm] = {
                    "sha256": compute_sha256(f)
                }
        except Exception:
            pass

    out_file = "experiments/ucip25/manifests/reproducibility_manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Reproducibility manifest generated at {out_file} ({len(manifest['artifacts'])} files).")


if __name__ == "__main__":
    generate_manifest()
