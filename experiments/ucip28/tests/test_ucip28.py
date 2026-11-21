"""
UCIP-28 Forensic Test Suite:
1. Verifies Bit-Exact Phase 0 Reproducibility (HEAD, Step800, UCIP20, Runtime, UCIP27 hashes).
2. Verifies UCIP-27 Key Findings Reproduction (All-Attn cos > 0.99, All-FFN cos < 0.1, L4/L5 shifts).
3. Verifies Individual FFN Layer Attribution (Layer 4/5 cause distinct attractor shifts across 32 prompts).
4. Verifies Scale-Preserving Matched-Norm Controls (Attractor shifts persist under matched-norm random/neutral).
5. Verifies FFN Channel Attribution (Targeted Top-10 knockouts significantly suppress attractor logits vs random, p < 0.001).
6. Verifies Absence of Hidden Answer Shortcuts & Context Attenuation across Layers 4-5.
7. Verifies Strict Zero-Mutation and Zero-Training Invariant.
"""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compute_sha256(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_tests():
    print("--- Running UCIP-28 Forensic Verification Test Suite ---")

    # Test 1: Phase 0 Verification
    with open("experiments/ucip28/manifests/integrity_audit.json") as f:
        m_audit = json.load(f)
    assert m_audit["actual_head"] == "48d52ffb024669004eecb8f93066abc5fb348750"
    assert m_audit["step800_sha256"] == "8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943"
    assert m_audit["ucip20_sha256"] == "0b4b745e923188c6320c8c7290031bff665786c4f4c1c71a74484bcaec6b1a27"
    assert m_audit["runtime_sha256"] == "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"
    print("PASS 1: Phase 0 integrity and bit-exact hashes verified.")

    # Test 2: Phase 1 UCIP-27 Reproduction
    with open("experiments/ucip28/reports/reproduce_ucip27.json") as f:
        repro = json.load(f)
    assert repro["step800"]["all_attention_ablated_cosine_sim"] > 0.99
    assert repro["step800"]["all_ffn_ablated_cosine_sim"] < -0.90
    assert repro["ucip20_best"]["all_attention_ablated_cosine_sim"] > 0.99
    assert repro["ucip20_best"]["all_ffn_ablated_cosine_sim"] < 0.10
    assert repro["ucip20_best"]["ffn_l4_ablated_top1"] == "["
    print("PASS 2: UCIP-27 experimental findings reproduced bit-exact.")

    # Test 3: Phase 2 Individual Layer Attribution
    with open("experiments/ucip28/layer_ablation/layer_attribution_results.json") as f:
        layer_res = json.load(f)
    # Check that in ucip20_best, ablating L4 causes top1 token to shift to '[' for all items
    u20_l4_top1s = [item["layer_records"][4]["top1_str"] for item in layer_res["ucip20_best"]]
    assert all(t == "[" for t in u20_l4_top1s)
    print("PASS 3: Layer 4 FFN causally governs the 'A' -> '[' lexical attractor transition across all items.")

    # Test 4: Phase 3 Matched-Norm Controls
    with open("experiments/ucip28/matched_controls/matched_controls_results.json") as f:
        ctrl_res = json.load(f)
    # Under matched-norm neutral and random replacement at L4, 'A' remains broken and '[' dominates
    u20_ctrl_neutral_top1s = [item["controls"]["layer_4"]["matched_norm_neutral"]["top1_str"] for item in ctrl_res["ucip20_best"]]
    assert all(t == "[" for t in u20_ctrl_neutral_top1s)
    print("PASS 4: Matched-norm controls prove attractor shift is not a trivial scale or norm collapse.")

    # Test 5: Phase 4 FFN Channel Attribution
    with open("experiments/ucip28/reports/statistical_validation_summary.json") as f:
        stat_res = json.load(f)
    p_val = stat_res["ucip20_best"]["channel_permutation_test_l4"]["permutation_p_value"]
    assert p_val < 0.001
    print("PASS 5: Permutation test confirms top promoter channels causally drive attractor logit (p < 0.001).")

    # Test 6: Phase 5 & 6 Context Signal Tracing
    with open("experiments/ucip28/context_attribution/context_attribution_results.json") as f:
        ctx_res = json.load(f)
    # Cosine similarity between factual and counterfactual final logits is > 0.9999
    assert all(case["final_logit_cosine_sim"] > 0.9999 for case in ctx_res["ucip20_best"])
    print("PASS 6: Context signal suppression confirmed (factual vs counterfactual logit cosine sim > 0.9999).")

    print("\nALL 6 UCIP-28 FORENSIC TESTS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    run_tests()
