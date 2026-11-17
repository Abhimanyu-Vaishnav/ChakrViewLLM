import json
import re
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
DIAG_FILE = REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "pipeline_diagnostics_raw.json"
BENCH_FILE = REPO_ROOT / "experiments" / "cognition" / "ucip_08" / "sealed_owner_reality_100.json"

from experiments.cognition.ucip_08.evaluator_owner_reality_100 import evaluate_owner_reality_probe

diag_data = json.load(open(DIAG_FILE, encoding="utf-8"))
probes = json.load(open(BENCH_FILE, encoding="utf-8"))["probes"]
probe_map = {p["id"]: p for p in probes}

for ckpt_key in ["ucip07_medium", "ucip08_medium"]:
    print(f"\n==========================================")
    print(f"BENCHMARK SCORE ACROSS DECODING ARMS: {ckpt_key}")
    print(f"==========================================")
    arms = diag_data[ckpt_key]["arms"]
    for arm_name, results in arms.items():
        passed = 0
        cat_stats = {}
        for r in results:
            p = probe_map[r["id"]]
            eval_res = evaluate_owner_reality_probe(p, r["generated_text"])
            cat = p["category"]
            if cat not in cat_stats:
                cat_stats[cat] = [0, 0]
            cat_stats[cat][1] += 1
            if eval_res["passed"]:
                passed += 1
                cat_stats[cat][0] += 1
        print(f"{arm_name:<25}: {passed:>2}/100 ({passed:.1f}%)")

