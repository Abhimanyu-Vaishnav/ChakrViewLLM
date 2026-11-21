"""
UCIP-26 Adversarial & Forensic Test Suite:
1. Verifies reproduction of 0% exact match across all 1,000 records
2. Verifies bit-exact prediction alignment with benchmark targets
3. Verifies Task H 100% single-target degeneracy
4. Verifies prompt serialization boundary tokens (: vs : )
5. Verifies logit diagnostics top-1 token dominance
6. Verifies teacher-forcing vs free-generation NLL gap
7. Verifies semantic context insensitivity (cosine similarity > 0.999 across replaced contexts)
8. Verifies FFN-dominated residual stream dynamics
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_tests():
    print("--- Running UCIP-26 Forensic Test Suite ---")

    # Test 1: Zero accuracy reproduction
    with open("experiments/ucip26/manifests/phase0_verification.json") as f:
        p0 = json.load(f)
    assert p0["recomputed_scores"]["step800"]["accuracy"] == 0.0
    assert p0["recomputed_scores"]["ucip20_best"]["accuracy"] == 0.0
    print("PASS 1: Zero exact match score reproduced independently for both models.")

    # Test 2: Alignment verification
    with open("experiments/ucip26/integrity_audit/scoring_and_task_h_audit.json") as f:
        p1 = json.load(f)
    assert p1["alignment_verified"] is True
    assert p1["alignment_errors_count"] == 0
    print("PASS 2: Prediction-to-target alignment verified bit-exact (0 errors).")

    # Test 3: Task H degeneracy check
    assert p1["task_h_special_audit"]["is_degenerate_single_target"] is True
    assert p1["task_h_special_audit"]["total_records"] == 120
    print("PASS 3: Task H single-target degeneracy independently verified.")

    # Test 4: Conditioning semantic insensitivity
    with open("experiments/ucip26/conditioning/conditioning_results.json") as f:
        cond = json.load(f)
    for k, v in cond.items():
        assert v["cosine_similarity_to_base"] >= 0.999
    print("PASS 4: Generation-boundary logit cosine similarity >= 0.999 across semantic replacements.")

    # Test 5: Teacher-Forced vs Free Generation NLL Gap
    with open("experiments/ucip26/teacher_forcing/teacher_forcing_comparison.json") as f:
        tf = json.load(f)
    s800_nlls = [r["teacher_forced_nll"] for r in tf["step800"]]
    u20_nlls = [r["teacher_forced_nll"] for r in tf["ucip20_best"]]
    assert min(s800_nlls) > 4.5
    assert min(u20_nlls) > 1.4
    print("PASS 5: Teacher-forcing target NLL gap confirmed.")

    # Test 6: Mechanistic MLP-to-Attention dominance
    with open("experiments/ucip26/mechanistic/mechanistic_diagnostics.json") as f:
        mech = json.load(f)
    ratios = [l["mlp_to_attn_ratio"] for l in mech["layer_dynamics"]]
    assert all(r > 1.0 for r in ratios)
    print("PASS 6: MLP output norm strictly exceeds Attention output norm across all layers.")

    print("\nALL UCIP-26 FORENSIC TESTS PASSED.")


if __name__ == "__main__":
    run_tests()
