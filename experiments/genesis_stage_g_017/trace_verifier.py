"""
NR-017 Phase 5: Independent Trace Verifier & Execution Replay Engine
Parses generated state traces and verifies:
1. Correct formatting (tags intact)
2. Complete trace sequence
3. Valid initial state transition
4. Valid intermediate transitions
5. Valid final state
6. Correct final answer
7. Entire trace verified correct

Also tests the verifier itself with deliberately corrupted traces:
- Corrupted arithmetic
- Missing transitions
- Reordered states
- Invalid operations
"""

import re
import json
import torch
from pathlib import Path

class IndependentTraceVerifier:
    def __init__(self):
        pass

    def verify_trace_b(self, generated_text: str, expected_states: list, expected_answer: str) -> dict:
        """Verifies Condition B (Explicit State Trace)."""
        res = {
            "format_valid": False,
            "complete_trace": False,
            "valid_first_transition": False,
            "valid_intermediate_transitions": False,
            "valid_final_state": False,
            "correct_final_answer": False,
            "entire_trace_correct": False
        }

        # Check format
        has_states = ("<state_1>" in generated_text and "</state_1>" in generated_text)
        has_answer = ("<answer>" in generated_text and "</answer>" in generated_text)
        if not (has_states and has_answer):
            return res
        res["format_valid"] = True

        # Extract answer
        ans_match = re.search(r"<answer>(.*?)</answer>", generated_text)
        if ans_match:
            ans_val = ans_match.group(1).strip()
            if ans_val == expected_answer or expected_answer in ans_val:
                res["correct_final_answer"] = True

        # Extract states
        state_matches = re.findall(r"<state_\d+>(.*?)</state_\d+>", generated_text)
        extracted_states = [s.strip() for s in state_matches]

        if len(extracted_states) == len(expected_states):
            res["complete_trace"] = True

        if extracted_states and len(expected_states) > 0:
            if extracted_states[0] == expected_states[0]:
                res["valid_first_transition"] = True

        if extracted_states and len(expected_states) > 0:
            if extracted_states[-1] == expected_states[-1]:
                res["valid_final_state"] = True

        if len(extracted_states) == len(expected_states):
            all_inter = all(e == a for e, a in zip(extracted_states, expected_states))
            if all_inter:
                res["valid_intermediate_transitions"] = True

        if res["valid_intermediate_transitions"] and res["correct_final_answer"]:
            res["entire_trace_correct"] = True

        return res

    def self_test(self):
        """Self-test verifier with corrupted, missing, and reordered traces."""
        exp_states = ["x=10", "y=20", "sum=30"]
        exp_ans = "30"

        # 1. Perfect trace
        t_perfect = "<state_1> x=10 </state_1> <state_2> y=20 </state_2> <state_3> sum=30 </state_3> <answer> 30 </answer>"
        r1 = self.verify_trace_b(t_perfect, exp_states, exp_ans)
        assert r1["entire_trace_correct"] == True, "Self-test 1 failed!"

        # 2. Corrupted arithmetic state
        t_corrupt = "<state_1> x=10 </state_1> <state_2> y=20 </state_2> <state_3> sum=999 </state_3> <answer> 30 </answer>"
        r2 = self.verify_trace_b(t_corrupt, exp_states, exp_ans)
        assert r2["entire_trace_correct"] == False, "Self-test 2 failed!"
        assert r2["correct_final_answer"] == True, "Self-test 2 answer extraction failed!"

        # 3. Missing transition
        t_missing = "<state_1> x=10 </state_1> <state_2> sum=30 </state_2> <answer> 30 </answer>"
        r3 = self.verify_trace_b(t_missing, exp_states, exp_ans)
        assert r3["entire_trace_correct"] == False, "Self-test 3 failed!"

        # 4. Reordered transitions
        t_reordered = "<state_1> y=20 </state_1> <state_2> x=10 </state_2> <state_3> sum=30 </state_3> <answer> 30 </answer>"
        r4 = self.verify_trace_b(t_reordered, exp_states, exp_ans)
        assert r4["valid_first_transition"] == False, "Self-test 4 failed!"

        return True

def run_trace_verification_suite():
    print("=== NR-017 INDEPENDENT TRACE VERIFICATION SUITE ===")
    verifier = IndependentTraceVerifier()
    self_test_passed = verifier.self_test()
    print(f"Verifier Self-Test on Corrupted/Missing/Reordered Traces: {self_test_passed} (PASSED)")

    # Load Condition B (Explicit State Trace) evaluation details
    # We inspect the generated traces from Condition_B_Seed42
    import sys
    sys.path.insert(0, '.')
    from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
    from chakrview.tokenizer.serialization import load_tokenizer_artifacts

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    ckpt_path = "experiments/genesis_stage_g_017/Condition_B_Seed42.pt"
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Genesis17MConfig(**st['config'])
    model = ChakrGenesis17M(cfg)
    model.load_state_dict(st['model_state_dict'])
    model.to(device)
    model.eval()

    with open("data/stage_g_017/stage_g_sealed_benchmark_400.json", 'r', encoding='utf-8') as f:
        bench_tasks = json.load(f)

    metrics = {
        "format_valid": 0,
        "complete_trace": 0,
        "valid_first_transition": 0,
        "valid_intermediate_transitions": 0,
        "valid_final_state": 0,
        "correct_final_answer": 0,
        "entire_trace_correct": 0,
        "total_tasks": len(bench_tasks)
    }

    results = []

    for idx, t in enumerate(bench_tasks):
        p_ids = torch.tensor([tokenizer.encode(t['prompt'])], dtype=torch.long, device=device)
        curr = p_ids
        with torch.no_grad():
            for _ in range(80):
                out, _ = model(curr)
                nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                curr = torch.cat([curr, nxt], dim=1)
                if nxt.item() == 1:
                    break
        gen_text = tokenizer.decode(curr[0, p_ids.size(1):].tolist()).strip()

        v_res = verifier.verify_trace_b(gen_text, t['states'], t['ground_truth_answer'])
        for k in metrics:
            if k != "total_tasks" and v_res.get(k, False):
                metrics[k] += 1

        results.append({
            "task_id": t['task_id'],
            "is_ood": t['is_ood'],
            "gen_text": gen_text,
            "verification": v_res
        })

    print("\n--- Trace Verification Results (Condition B on 400 Tasks) ---")
    total = metrics["total_tasks"]
    for k, v in metrics.items():
        if k != "total_tasks":
            print(f"  {k:35s}: {v:3d}/{total} ({v/total*100:5.1f}%)")

    out_file = Path("data/stage_g_017/trace_verification_results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": metrics,
            "details": results
        }, f, indent=2)

    print(f"\nTrace verification results written to: {out_file}")

if __name__ == '__main__':
    run_trace_verification_suite()
