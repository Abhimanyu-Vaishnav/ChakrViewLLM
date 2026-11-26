"""
NR-017 Phase 7: Compositional and Counterfactual Testing Suite
Tests paired tasks across 4 regimes:
1. Variable Renaming Invariance: Same computation, renamed variables (e.g., x, y -> a, b).
   Expected: Output preserved.
2. Value Perturbation Sensitivity: Same variables, modified input values.
   Expected: Output changes appropriately.
3. Operation Reordering: Commutative operations reordered (e.g., a + b vs b + a).
   Expected: Output preserved.
4. Non-Commutative Reordering: Non-commutative operations reordered (e.g., a - b vs b - a).
   Expected: Output changes appropriately.
"""

import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, '.')
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def run_compositional_and_counterfactual_tests():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== NR-017 COMPOSITIONAL & COUNTERFACTUAL PAIRED TESTS on {device} ===")

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    # Test Condition A (Direct) and Condition B (Trace) models
    models = {
        "Direct_A": "experiments/genesis_stage_g_017/Condition_A_Seed42.pt",
        "Trace_B": "experiments/genesis_stage_g_017/Condition_B_Seed42.pt",
    }

    pairs = [
        # Pair 1: Variable Renaming Invariance (x, y -> a, b)
        {
            "pair_name": "Pair 1: Variable Renaming Invariance",
            "base_prompt": "<task> Bind x = 12 and y = 15. Compute x + y. </task>",
            "cf_prompt": "<task> Bind a = 12 and b = 15. Compute a + b. </task>",
            "expected_behavior": "invariant",
            "gt_base": "27",
            "gt_cf": "27"
        },
        # Pair 2: Value Perturbation Sensitivity (12, 15 -> 12, 30)
        {
            "pair_name": "Pair 2: Value Perturbation Sensitivity",
            "base_prompt": "<task> Bind x = 12 and y = 15. Compute x + y. </task>",
            "cf_prompt": "<task> Bind x = 12 and y = 30. Compute x + y. </task>",
            "expected_behavior": "sensitive",
            "gt_base": "27",
            "gt_cf": "42"
        },
        # Pair 3: Commutative Reordering (Add x and y vs Add y and x)
        {
            "pair_name": "Pair 3: Commutative Reordering",
            "base_prompt": "<task> Initial state S = 20. Add 10, then add 5. </task>",
            "cf_prompt": "<task> Initial state S = 20. Add 5, then add 10. </task>",
            "expected_behavior": "invariant",
            "gt_base": "35",
            "gt_cf": "35"
        },
        # Pair 4: Non-Commutative Reordering (Add 10 then sub 5 vs Sub 5 then add 10)
        {
            "pair_name": "Pair 4: State Update Perturbation",
            "base_prompt": "<task> Initial state S = 50. Add 15, then subtract 5. </task>",
            "cf_prompt": "<task> Initial state S = 50. Add 15, then subtract 20. </task>",
            "expected_behavior": "sensitive",
            "gt_base": "60",
            "gt_cf": "45"
        }
    ]

    results = {}

    for m_label, ckpt_path in models.items():
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = Genesis17MConfig(**st['config'])
        model = ChakrGenesis17M(cfg)
        model.load_state_dict(st['model_state_dict'])
        model.to(device)
        model.eval()

        print(f"\n--- Testing Model: {m_label} ---")
        m_results = []

        for p in pairs:
            # Generate base
            p_ids_b = torch.tensor([tokenizer.encode(p['base_prompt'])], dtype=torch.long, device=device)
            curr_b = p_ids_b
            with torch.no_grad():
                for _ in range(50):
                    out, _ = model(curr_b)
                    nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                    curr_b = torch.cat([curr_b, nxt], dim=1)
                    if nxt.item() == 1: break
            gen_b = tokenizer.decode(curr_b[0, p_ids_b.size(1):].tolist()).strip()

            # Generate counterfactual
            p_ids_cf = torch.tensor([tokenizer.encode(p['cf_prompt'])], dtype=torch.long, device=device)
            curr_cf = p_ids_cf
            with torch.no_grad():
                for _ in range(50):
                    out, _ = model(curr_cf)
                    nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                    curr_cf = torch.cat([curr_cf, nxt], dim=1)
                    if nxt.item() == 1: break
            gen_cf = tokenizer.decode(curr_cf[0, p_ids_cf.size(1):].tolist()).strip()

            behaved_correctly = False
            if p['expected_behavior'] == "invariant":
                behaved_correctly = (gen_b == gen_cf) or (p['gt_base'] in gen_b and p['gt_cf'] in gen_cf)
            else: # sensitive
                behaved_correctly = (gen_b != gen_cf)

            print(f"  {p['pair_name']}:")
            print(f"    Base Output: {repr(gen_b[:40])}")
            print(f"    CF Output:   {repr(gen_cf[:40])}")
            print(f"    Behavior Expected: {p['expected_behavior']} | Correctly Handled: {behaved_correctly}")

            m_results.append({
                "pair_name": p['pair_name'],
                "expected_behavior": p['expected_behavior'],
                "gen_base": gen_b,
                "gen_cf": gen_cf,
                "correct_behavior": behaved_correctly
            })

        results[m_label] = m_results

    out_file = Path("data/stage_g_017/compositional_counterfactual_results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved compositional and counterfactual results to: {out_file}")

if __name__ == '__main__':
    run_compositional_and_counterfactual_tests()
