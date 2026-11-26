"""
NR-017 Phase 6: Detailed Family and Difficulty Breakdown
Analyzes the 400-task sealed benchmark results across all 10 families and 3 difficulty levels
for Condition A, Condition B, and Condition C.
"""

import json
from pathlib import Path

def analyze_generalization_breakdown():
    with open("data/stage_g_017/training_comparison_results.json", 'r') as f:
        comp_res = json.load(f)

    with open("data/stage_g_017/stage_g_sealed_benchmark_400.json", 'r') as f:
        bench_tasks = json.load(f)

    task_map = {t['task_id']: t for t in bench_tasks}

    # Evaluate per family and per difficulty for Seed 42 runs
    runs = {
        "Direct_A": "Condition_A_Seed42.pt",
        "Explicit_B": "Condition_B_Seed42.pt",
        "Compact_C": "Condition_C_Seed42.pt"
    }

    # We evaluate directly using inference on the tasks
    import torch
    import sys
    sys.path.insert(0, '.')
    from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
    from chakrview.tokenizer.serialization import load_tokenizer_artifacts

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    stats = {}
    for r_name, ckpt_name in runs.items():
        st = torch.load(f"experiments/genesis_stage_g_017/{ckpt_name}", map_location="cpu", weights_only=False)
        cfg = Genesis17MConfig(**st['config'])
        model = ChakrGenesis17M(cfg)
        model.load_state_dict(st['model_state_dict'])
        model.to(device)
        model.eval()

        family_acc = {}
        diff_acc = {1: [0, 0], 2: [0, 0], 3: [0, 0]}

        cond_key = "target_a" if "Direct" in r_name else ("target_b" if "Explicit" in r_name else "target_c")
        max_g = 40 if "Direct" in r_name else (100 if "Explicit" in r_name else 60)

        for t in bench_tasks:
            fam = t['family']
            diff = t['difficulty']
            if fam not in family_acc:
                family_acc[fam] = [0, 0] # passed, total

            p_ids = torch.tensor([tokenizer.encode(t['prompt'])], dtype=torch.long, device=device)
            curr = p_ids
            with torch.no_grad():
                for _ in range(max_g):
                    out, _ = model(curr)
                    nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                    curr = torch.cat([curr, nxt], dim=1)
                    if nxt.item() == 1: break

            gen_text = tokenizer.decode(curr[0, p_ids.size(1):].tolist()).strip()
            gt = t['ground_truth_answer']

            if cond_key == "target_a":
                extracted = gen_text.split("<answer>")[1].split("</answer>")[0].strip() if ("<answer>" in gen_text and "</answer>" in gen_text) else gen_text.replace("<answer>", "").replace("</answer>", "").strip()
            elif cond_key == "target_b":
                extracted = gen_text.split("<answer>")[1].split("</answer>")[0].strip() if ("<answer>" in gen_text and "</answer>" in gen_text) else gen_text.strip()
            else:
                extracted = gen_text.split("answer=")[1].replace("</trace>", "").strip() if "answer=" in gen_text else gen_text.strip()

            passed = (extracted == gt) or (gt in extracted)

            family_acc[fam][1] += 1
            diff_acc[diff][1] += 1
            if passed:
                family_acc[fam][0] += 1
                diff_acc[diff][0] += 1

        stats[r_name] = {
            "family": family_acc,
            "difficulty": diff_acc
        }

    print("\n=== FAMILY-WISE ACCURACY BREAKDOWN (400 TASKS) ===")
    families = sorted(list(bench_tasks[0].keys()))
    all_fams = sorted(list(stats["Direct_A"]["family"].keys()))
    print(f"{'Task Family':32s} | {'Direct A':12s} | {'Explicit B':12s} | {'Compact C':12s}")
    print("-" * 75)
    for f in all_fams:
        a_p, a_tot = stats["Direct_A"]["family"][f]
        b_p, b_tot = stats["Explicit_B"]["family"][f]
        c_p, c_tot = stats["Compact_C"]["family"][f]
        print(f"{f:32s} | {a_p/a_tot*100:4.1f}% ({a_p:2d}/{a_tot}) | {b_p/b_tot*100:4.1f}% ({b_p:2d}/{b_tot}) | {c_p/c_tot*100:4.1f}% ({c_p:2d}/{c_tot})")

    print("\n=== DIFFICULTY-WISE ACCURACY BREAKDOWN ===")
    for d in [1, 2, 3]:
        a_p, a_tot = stats["Direct_A"]["difficulty"][d]
        b_p, b_tot = stats["Explicit_B"]["difficulty"][d]
        c_p, c_tot = stats["Compact_C"]["difficulty"][d]
        print(f"Level {d} Tasks ({a_tot} tasks): Direct A = {a_p/a_tot*100:4.1f}% | Explicit B = {b_p/b_tot*100:4.1f}% | Compact C = {c_p/c_tot*100:4.1f}%")

    out_file = Path("data/stage_g_017/detailed_generalization_breakdown.json")
    with open(out_file, 'w', encoding='utf-8') as fp:
        json.dump(stats, fp, indent=2)
    print(f"\nDetailed breakdown saved to: {out_file}")

if __name__ == '__main__':
    analyze_generalization_breakdown()
