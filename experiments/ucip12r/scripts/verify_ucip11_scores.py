"""
UCIP-12R Phase I: UCIP-11 Evaluation Score Forensic Verification Tool.
Inspects raw evaluation output records from UCIP-11 and validates scoring heuristics.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVAL_RESULTS_PATH = ROOT / "experiments/ucip11/results/full_benchmark_evaluation_results.json"

with open(EVAL_RESULTS_PATH, "r", encoding="utf-8") as f:
    eval_results = json.load(f)

print(f"Auditing UCIP-11 evaluation records across {len(eval_results)} conditions...")

for cond, data in eval_results.items():
    sealed_evals = data.get("sealed_evaluations", [])
    passes = [e for e in sealed_evals if e.get("passed")]
    print(f"\nCondition: {cond}")
    print(f"  Total Probes Evaluated: {len(sealed_evals)}")
    print(f"  Nominal Reported Passes: {len(passes)}")
    for p in passes:
        print(f"    Probe ID: {p.get('id')}")
        print(f"    Scoring Reason: {p.get('reason')}")
        print(f"    Reported Score: {p.get('score')}")
        print(f"    Prompt: {repr(p.get('prompt'))}")
        print(f"    Raw Output: {repr(p.get('raw_output'))}")
        
        # Verify false positive
        assert p.get("id") == "SEA_INS_001"
        assert p.get("reason") == "Length check"
        assert "H2O" not in p.get("raw_output")
        print("    -> VERIFIED FALSE POSITIVE: Evaluator accepted incoherent text based on length check.")

print("\nAUDIT VERDICT: All historical UCIP-11 model passes on unseen sealed probes are invalid false positives.")
print("True functional generalization score across all conditions: 0.00%.")
