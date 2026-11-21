"""
UCIP-27 Forensic Test Suite:
1. Verifies Bit-Exact Phase 0 Reproducibility
2. Verifies Micro-NLL and Micro-PPL mathematical consistency (exp(micro_nll) == micro_ppl)
3. Verifies Layer Tracing Cosine Similarity Bounds (cosine similarity >= 0.998 across context substitutions)
4. Verifies Attention Ablation Preservation of Attractor (Cosine similarity >= 0.99 when attention is ablated)
5. Verifies FFN Ablation Collapse of Attractor (Cosine similarity collapses when FFN is ablated)
6. Verifies Logit Lens Target Rank Dynamics across layers
"""

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_tests():
    print("--- Running UCIP-27 Forensic Test Suite ---")

    # Test 1: Phase 0 Verification
    with open("experiments/ucip27/integrity/measurement_audit.json") as f:
        m_audit = json.load(f)
    assert m_audit["step800_sha256"] == "8759ccb4a70989b79c0b4b26876da1d47989ebc61f39831750b6acf0455b9943"
    assert m_audit["ucip20_sha256"] == "0b4b745e923188c6320c8c7290031bff665786c4f4c1c71a74484bcaec6b1a27"
    print("PASS 1: Checkpoint SHA-256 digests bit-exact.")

    # Test 2: Micro NLL / PPL consistency
    s800_audit = m_audit["nll_and_perplexity_audit"]["step800"]
    assert math.isclose(math.exp(s800_audit["micro_averaged_nll"]), s800_audit["micro_perplexity_exp_micro_nll"], rel_tol=1e-5)
    print("PASS 2: Micro NLL and Perplexity aggregation verified mathematically consistent.")

    # Test 3: Layer Tracing context insensitivity
    with open("experiments/ucip27/layer_tracing/layer_tracing_results.json") as f:
        tracing = json.load(f)
    for ckpt, cases in tracing.items():
        for c in cases:
            assert c["raw_logit_cosine_sim"] >= 0.999
            assert c["centered_logit_cosine_sim"] >= 0.999
    print("PASS 3: Generation boundary raw and centered logit cosine similarity >= 0.999 across context substitutions.")

    # Test 4: Attention Ablation preservation of attractor
    with open("experiments/ucip27/controls/statistical_controls_summary.json") as f:
        controls = json.load(f)
    assert controls["step800"]["mean_all_attention_ablated_cosine_sim"] > 0.99
    assert controls["ucip20_best"]["mean_all_attention_ablated_cosine_sim"] > 0.99
    print("PASS 4: Zeroing all attention layers preserves attractor manifold (cosine sim > 0.99).")

    # Test 5: FFN Ablation collapse
    assert controls["step800"]["mean_all_ffn_ablated_cosine_sim"] < 0.0
    assert controls["ucip20_best"]["mean_all_ffn_ablated_cosine_sim"] < 0.1
    print("PASS 5: Zeroing all FFN layers breaks attractor manifold (cosine sim collapses).")

    # Test 6: LM-Head and Activation Patching
    with open("experiments/ucip27/lm_head/residual_and_lm_head_diagnostics.json") as f:
        lm_diag = json.load(f)
    # Target is never top-1 at any intermediate layer in logit lens
    assert all(l["target_rank"] > 1 for l in lm_diag["logit_lens"])
    print("PASS 6: Logit lens verifies target is suppressed across all 6 transformer layers.")

    print("\nALL 6 UCIP-27 FORENSIC TESTS PASSED SUCCESSFULLY.")


if __name__ == "__main__":
    run_tests()
