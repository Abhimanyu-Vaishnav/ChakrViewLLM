"""
UCIP-26 Output Manifest Generator:
Records SHA-256 digests of all UCIP-26 audit outputs, scripts, and tests.
"""

import glob
import hashlib
import json
import os


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest():
    manifest = {
        "mission": "UCIP-26",
        "description": "Generation Boundary Forensics & Evaluation Integrity Audit",
        "artifacts": {}
    }

    files = sorted(glob.glob("experiments/ucip26/**/*", recursive=True))
    for f in files:
        if "output_manifest.json" in f:
            continue
        if os.path.isfile(f):
            norm = f.replace("\\", "/")
            manifest["artifacts"][norm] = {
                "sha256": compute_sha256(f)
            }

    out_file = "experiments/ucip26/manifests/output_manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"UCIP-26 Output manifest generated at {out_file} ({len(manifest['artifacts'])} files).")


if __name__ == "__main__":
    generate_manifest()
