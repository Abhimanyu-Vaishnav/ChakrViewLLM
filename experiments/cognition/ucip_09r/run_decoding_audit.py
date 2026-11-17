import json
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
DIAG_FILE = REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "pipeline_diagnostics_raw.json"
BENCH_FILE = REPO_ROOT / "experiments" / "cognition" / "ucip_08" / "sealed_owner_reality_100.json"

import sys
sys.path.insert(0, str(REPO_ROOT))
from experiments.cognition.ucip_08.evaluator_owner_reality_100 import evaluate_owner_reality_probe

data = json.load(open(DIAG_FILE, encoding="utf-8"))["ucip09_medium"] if "ucip09_medium" in json.load(open(DIAG_FILE, encoding="utf-8")) else json.load(open(DIAG_FILE, encoding="utf-8"))["ucip08_medium"]
probes = json.load(open(BENCH_FILE, encoding="utf-8"))["probes"]
probe_map = {p["id"]: p for p in probes}

print("=== CONTROLLED DECODING EVALUATION (UCIP-09R PHASE E) ===")
# Testing Arm A (Raw Greedy), Arm B (RepPen 1.05), Arm C (RepPen 1.15), Arm D (NoRepeat 3), Arm E (RepPen 1.05 + NoRepeat 3)
for arm_name, results in json.load(open(DIAG_FILE, encoding="utf-8"))["ucip08_medium"]["arms"].items():
    passed = 0
    for r in results:
        p = probe_map[r["id"]]
        if evaluate_owner_reality_probe(p, r["generated_text"])["passed"]:
            passed += 1
    print(f"{arm_name:<25}: {passed:>2}/100 ({passed:.1f}%)")
