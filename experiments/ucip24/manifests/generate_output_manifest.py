"""
UCIP-24 Output Manifest Generator:
Records SHA-256 digests of all UCIP-24 audit scripts, test suites, and audit outputs.
"""

import hashlib
import json
import glob
from pathlib import Path


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest():
    manifest = {
        "mission": "UCIP-24",
        "description": "Independent Benchmark Forensic Audit & Evaluation Readiness",
        "artifacts": {
            "audits": {},
            "tests": {},
            "manifests": {}
        }
    }

    for fpath in sorted(glob.glob("experiments/ucip24/**/*.json", recursive=True)):
        if "output_manifest.json" in fpath:
            continue
        norm = fpath.replace("\\", "/")
        manifest["artifacts"]["audits" if "audits" in norm else "manifests"][norm] = {
            "sha256": compute_sha256(fpath)
        }

    for fpath in sorted(glob.glob("experiments/ucip24/**/*.py", recursive=True)):
        norm = fpath.replace("\\", "/")
        category = "tests" if "tests" in norm else ("audits" if "audits" in norm else "manifests")
        manifest["artifacts"][category][norm] = {
            "sha256": compute_sha256(fpath)
        }

    out_file = "experiments/ucip24/manifests/output_manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"UCIP-24 Output manifest generated at {out_file}")


if __name__ == "__main__":
    generate_manifest()
