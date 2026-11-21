"""
Generate output manifest recording SHA-256 digests and record counts for all
UCIP-23 benchmark artifacts: datasets, manifests, schemas, validators, baselines, and tests.
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


def generate_output_manifest():
    manifest = {
        "benchmark": "UCIP-23 Compositional Binding Benchmark",
        "version": "1.0.0",
        "artifacts": {
            "datasets": {},
            "manifests": {},
            "schemas": {},
            "validators": {},
            "tests": {},
            "baselines": {}
        },
        "summary": {}
    }

    # Datasets
    total_records = 0
    for fpath in sorted(glob.glob("experiments/ucip23/datasets/task_*.jsonl")):
        norm_path = fpath.replace("\\", "/")
        with open(fpath, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        count = len(lines)
        total_records += count
        manifest["artifacts"]["datasets"][norm_path] = {
            "records": count,
            "sha256": compute_sha256(fpath)
        }

    # Manifests
    for fpath in sorted(glob.glob("experiments/ucip23/manifests/*.json")):
        norm_path = fpath.replace("\\", "/")
        manifest["artifacts"]["manifests"][norm_path] = {
            "sha256": compute_sha256(fpath)
        }

    # Schemas
    schema_path = "experiments/ucip22/schemas/benchmark_record_schema.json"
    if Path(schema_path).exists():
        manifest["artifacts"]["schemas"][schema_path] = {
            "sha256": compute_sha256(schema_path)
        }

    # Validators
    for vpath in [
        "experiments/ucip22/validators/benchmark_validator.py",
        "experiments/ucip23/validators/extended_validator.py"
    ]:
        if Path(vpath).exists():
            manifest["artifacts"]["validators"][vpath] = {
                "sha256": compute_sha256(vpath)
            }

    # Tests
    for tpath in [
        "experiments/ucip22/tests/test_benchmark_validator.py",
        "experiments/ucip23/tests/test_adversarial_validator.py"
    ]:
        if Path(tpath).exists():
            manifest["artifacts"]["tests"][tpath] = {
                "sha256": compute_sha256(tpath)
            }

    # Baselines
    base_path = "experiments/ucip23/baselines/baseline_specs.json"
    if Path(base_path).exists():
        manifest["artifacts"]["baselines"][base_path] = {
            "sha256": compute_sha256(base_path)
        }

    manifest["summary"] = {
        "total_task_files": len(manifest["artifacts"]["datasets"]),
        "total_benchmark_records": total_records,
        "expected_records": 1000,
        "integrity_verified": total_records == 1000
    }

    out_file = "experiments/ucip23/manifests/output_manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Output manifest generated successfully at {out_file}.")
    print(f"Total benchmark records: {total_records}")


if __name__ == "__main__":
    generate_output_manifest()
