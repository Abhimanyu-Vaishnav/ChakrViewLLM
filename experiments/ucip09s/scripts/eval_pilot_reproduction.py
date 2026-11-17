import sys
import re
import json
import torch
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.runtime.sampling import apply_repetition_penalty, apply_no_repeat_ngram
from experiments.ucip09s.scripts.run_owner_reproduction import evaluate_reproduction_probe

def main():
    manifest_path = REPO_ROOT / "experiments" / "ucip09s" / "manifests" / "owner_reproduction_suite.json"
    probes = json.load(open(manifest_path, encoding="utf-8"))
    
    ckpt_path = REPO_ROOT / "experiments" / "ucip09s" / "checkpoints" / "ucip09s_balanced_pilot.pt"
    runner = DirectNeuralRunner(str(ckpt_path))
    tok = runner.tokenizer
    model = runner.model
    model.eval()

    total_passed = 0
    cat_stats = {}
    results = []

    print("\n=======================================================")
    print("Running Owner Reproduction Test Suite: UCIP-09S Condition C (Balanced Pilot)")
    print("=======================================================")

    for p in probes:
        cat = p["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "passed": 0}
        cat_stats[cat]["total"] += 1

        prompt_str = f"User: {p['input']}\nAssistant:"
        p_tokens = tok.encode(prompt_str, add_bos=True, add_eos=False)
        curr = list(p_tokens)
        gen = []

        with torch.no_grad():
            for step in range(48):
                t_in = torch.tensor([curr], dtype=torch.long)
                if t_in.shape[1] > 512:
                    t_in = t_in[:, -512:]
                logits = model(t_in)[0, -1, :]

                if gen:
                    logits = apply_repetition_penalty(logits, gen, 1.05)
                if len(curr) >= 3:
                    logits = apply_no_repeat_ngram(logits, curr, 3)

                nxt = int(torch.argmax(logits).item())
                if nxt == 1:
                    break
                gen.append(nxt)
                curr.append(nxt)

        out_text = tok.decode(gen, errors="replace").strip()
        eval_res = evaluate_reproduction_probe(p, out_text)
        if eval_res["passed"]:
            total_passed += 1
            cat_stats[cat]["passed"] += 1

        print(f"[{'PASS' if eval_res['passed'] else 'FAIL'}] [{cat}] {p['input']}")
        print(f"  -> Generated: {repr(out_text)}")
        print(f"  -> Reason:    {eval_res['reason']}\n")

        results.append({
            "id": p["id"],
            "category": cat,
            "input": p["input"],
            "generated": out_text,
            "passed": eval_res["passed"],
            "reason": eval_res["reason"]
        })

    accuracy = (total_passed / len(probes)) * 100
    print(f"Total Pilot Passed: {total_passed}/{len(probes)} ({accuracy:.1f}%)")
    for c, d in cat_stats.items():
        pct = (d["passed"] / d["total"]) * 100
        print(f"  {c:22s}: {d['passed']}/{d['total']} ({pct:.1f}%)")

    out_payload = {
        "UCIP-09S_Pilot": {
            "total_passed": total_passed,
            "total_probes": len(probes),
            "accuracy": accuracy,
            "category_breakdown": cat_stats,
            "evaluations": results
        }
    }
    out_path = REPO_ROOT / "experiments" / "ucip09s" / "evaluations" / "pilot_reproduction_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved pilot reproduction results to: {out_path}")

if __name__ == "__main__":
    main()
