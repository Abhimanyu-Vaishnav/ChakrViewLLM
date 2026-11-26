"""
NR-017 Phase 7: Compositional and Counterfactual Paired Invariance Tests
Tests whether models change output appropriately when computations change,
and preserve output when only irrelevant surface details change.

Evaluates 5 paired conditions across Direct A, Explicit B, Compact C:
Pair 1: Variable Renaming Invariance (x, y -> a, b with same values) -> Target: Invariant output
Pair 2: Value Mutation Sensitivity (x=10, y=20 vs x=10, y=50) -> Target: Distinct output
Pair 3: Operation Commutativity (add a, b vs add b, a) -> Target: Invariant output
Pair 4: Sequence Length Extension (2 steps vs 4 steps) -> Target: Tracking depth
Pair 5: Representation Invariance (words vs symbols) -> Target: Invariant answer
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

    pairs = [
        # Pair 1: Variable Renaming Invariance
        {
            "pair_name": "Variable Renaming Invariance",
            "task_base": "<task> Bind x = 12 and y = 15. Compute x + y. </task>",
            "task_variant": "<task> Bind a = 12 and b = 15. Compute a + b. </task>",
            "expected_relation": "invariant",
            "base_gt": "27",
            "variant_gt": "27"
        },
        # Pair 2: Value Mutation Sensitivity
        {
            "pair_name": "Value Mutation Sensitivity",
            "task_base": "<task> Initial state S = 20. Add 10, then subtract 5. </task>",
            "task_variant": "<task> Initial state S = 20. Add 10, then subtract 15. </task>",
            "expected_relation": "sensitive",
            "base_gt": "25",
            "variant_gt": "15"
        },
        # Pair 3: Commutativity Invariance
        {
            "pair_name": "Commutativity Invariance",
            "task_base": "<task> Compute 14 * 6. </task>",
            "task_variant": "<task> Compute 6 * 14. </task>",
            "expected_relation": "invariant",
            "base_gt": "84",
            "variant_gt": "84"
        },
        # Pair 4: Representation Invariance
        {
            "pair_name": "Representation Invariance",
            "task_base": "<task> Take word 'tensor'. Step 1: uppercase. Step 2: reverse. </task>",
            "task_variant": "<task> Take word 'TENSOR'. Reverse the characters. </task>",
            "expected_relation": "invariant",
            "base_gt": "ROSNET",
            "variant_gt": "ROSNET"
        }
    ]

    models = {
        "Direct_A": "Condition_A_Seed42.pt",
        "Explicit_B": "Condition_B_Seed42.pt",
        "Compact_C": "Condition_C_Seed42.pt"
    }

    results = {}

    for m_name, ckpt_name in models.items():
        st = torch.load(f"experiments/genesis_stage_g_017/{ckpt_name}", map_location="cpu", weights_only=False)
        cfg = Genesis17MConfig(**st['config'])
        model = ChakrGenesis17M(cfg)
        model.load_state_dict(st['model_state_dict'])
        model.to(device)
        model.eval()

        m_results = []
        for p in pairs:
            # Generate base
            p1 = torch.tensor([tokenizer.encode(p['task_base'])], dtype=torch.long, device=device)
            c1 = p1
            with torch.no_grad():
                for _ in range(40):
                    out, _ = model(c1)
                    nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                    c1 = torch.cat([c1, nxt], dim=1)
                    if nxt.item() == 1: break
            out_base = tokenizer.decode(c1[0, p1.size(1):].tolist()).strip()

            # Generate variant
            p2 = torch.tensor([tokenizer.encode(p['task_variant'])], dtype=torch.long, device=device)
            c2 = p2
            with torch.no_grad():
                for _ in range(40):
                    out, _ = model(c2)
                    nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                    c2 = torch.cat([c2, nxt], dim=1)
                    if nxt.item() == 1: break
            out_var = tokenizer.decode(c2[0, p2.size(1):].tolist()).strip()

            is_invariant = (out_base == out_var)
            expected_inv = (p['expected_relation'] == 'invariant')
            satisfied = (is_invariant == expected_inv)

            m_results.append({
                "pair": p['pair_name'],
                "expected_relation": p['expected_relation'],
                "out_base": out_base,
                "out_variant": out_var,
                "is_invariant": is_invariant,
                "satisfied": satisfied
            })

            print(f"[{m_name}] {p['pair_name']}: Expected {p['expected_relation']} | Base: {repr(out_base[:25])} | Var: {repr(out_var[:25])} | Satisfied: {satisfied}")

        results[m_name] = m_results

    out_file = Path("data/stage_g_017/compositional_counterfactual_results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nCompositional and counterfactual results saved to: {out_file}")

if __name__ == '__main__':
    run_compositional_and_counterfactual_tests()
