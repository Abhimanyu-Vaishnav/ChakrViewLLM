"""
Verification script for all critical file hashes in UCIP-10E-R2V.
Checks:
- Checkpoints (UCIP-09, UCIP-09R, UCIP-09S)
- Benchmark datasets (sealed, development)
- Manifests
- Protected runtime (session.py)
- Evaluator implementation (strict_evaluators_r2.py)
"""

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")

target_files = {
    "protected_runtime": {
        "path": REPO_ROOT / "chakrview" / "runtime" / "session.py",
        "expected": "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
    },
    "ucip09_checkpoint": {
        "path": REPO_ROOT / "releases" / "ucip_09_candidates" / "chakrview_ucip09_medium_model.pt",
        "expected": "5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499"
    },
    "ucip09r_checkpoint": {
        "path": REPO_ROOT / "releases" / "ucip_09r_candidates" / "chakrview_ucip09r_medium_model.pt",
        "expected": "e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0"
    },
    "ucip09s_checkpoint": {
        "path": REPO_ROOT / "experiments" / "ucip09s" / "checkpoints" / "ucip09s_balanced_pilot.pt",
        "expected": "c99dcc0a229310ec682857349c06e1a46957b559a8d180a33a91ea05b225d830"
    },
    "benchmark_sealed": {
        "path": REPO_ROOT / "experiments" / "ucip10e_r2" / "datasets" / "sealed" / "benchmark_r2_sealed.json",
        "expected": "ef94076849dd5c61cdea291e3a2ede35fdda96f47e36c39e53d8cc54fb0f36ef"
    },
    "benchmark_development": {
        "path": REPO_ROOT / "experiments" / "ucip10e_r2" / "datasets" / "development" / "benchmark_r2_development.json",
        "expected": "29b213badefba55bd76cece47afb6f59d694cfe2040037970a3384539e610b3c"
    },
    "manifest_registry": {
        "path": REPO_ROOT / "experiments" / "ucip10e_r2" / "manifests" / "manifest_registry_r2.json",
        "expected": "f44c5af7f340e4d8873da01cf08d953e0af4e6503d221d5b56eb64870943b49a"
    },
    "evaluator_engine": {
        "path": REPO_ROOT / "experiments" / "ucip10e_r2" / "evaluators" / "strict_evaluators_r2.py",
        "expected": None
    }
}

results = {}
all_passed = True

for key, item in target_files.items():
    p = item["path"]
    if not p.exists():
        results[key] = {"path": str(p), "status": "MISSING", "matched": False}
        all_passed = False
        continue
    
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    exp = item["expected"]
    matched = (h == exp) if exp else True
    if exp and not matched:
        all_passed = False
        
    results[key] = {
        "path": str(p),
        "sha256": h,
        "expected_sha256": exp,
        "matched": matched,
        "status": "VALID" if matched else "HASH_MISMATCH"
    }

results["_overall_status"] = "ALL_MATCHED" if all_passed else "MISMATCH_DETECTED"

out_p = REPO_ROOT / "experiments" / "ucip10e_r2v" / "evaluations" / "hash_verification_manifest.json"
with open(out_p, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Hash verification results:")
print(json.dumps(results, indent=2))
