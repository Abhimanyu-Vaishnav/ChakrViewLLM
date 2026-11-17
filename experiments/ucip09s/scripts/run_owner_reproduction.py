import sys
import re
import json
import torch
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.runtime.sampling import apply_repetition_penalty, apply_no_repeat_ngram

MANIFEST_PATH = REPO_ROOT / "experiments" / "ucip09s" / "manifests" / "owner_reproduction_suite.json"

CHECKPOINTS = {
    "UCIP-09": REPO_ROOT / "releases" / "ucip_09_candidates" / "chakrview_ucip09_medium_model.pt",
    "UCIP-09R": REPO_ROOT / "releases" / "ucip_09r_candidates" / "chakrview_ucip09r_medium_model.pt"
}

def is_repetitive(text: str) -> bool:
    words = [w.strip() for w in text.split() if w.strip()]
    if len(words) < 4:
        return False
    for i in range(len(words) - 2):
        if words[i].lower() == words[i+1].lower() == words[i+2].lower():
            return True
    return False

def evaluate_reproduction_probe(probe, generated):
    cleaned = generated.strip()
    norm_out = cleaned.lower()
    cat = probe["category"]

    if not cleaned:
        return {"passed": False, "reason": "EMPTY_OUTPUT"}
    if is_repetitive(cleaned):
        return {"passed": False, "reason": "REPETITIVE_DEGENERATION"}

    # Arithmetic
    if cat == "Arithmetic":
        tgt = probe["target_number"]
        found_nums = [int(m) for m in re.findall(r"\b\d+\b", cleaned)]
        passed = (tgt in found_nums)
        return {"passed": passed, "reason": f"Target: {tgt}, Found: {found_nums}"}

    # Exact Copying
    if cat == "Exact copying":
        tgt = probe["target_text"].strip().lower()
        t_clean = re.sub(r"^[^\w]+|[^\w]+$", "", tgt)
        g_clean = re.sub(r"^[^\w]+|[^\w]+$", "", norm_out)
        passed = (t_clean == g_clean) or (tgt in norm_out and len(cleaned.split()) <= len(tgt.split()) + 2)
        return {"passed": passed, "reason": f"Target: '{tgt}' vs '{cleaned}'"}

    # Instruction Following
    if cat == "Instruction following":
        if "forbidden" in probe:
            for f in probe["forbidden"]:
                if re.search(rf"\b{f.lower()}\b", norm_out):
                    return {"passed": False, "reason": f"Forbidden word used: {f}"}
        exp_kws = probe.get("expected_keywords", [])
        hits = [k for k in exp_kws if re.search(rf"\b{k.lower()}\b", norm_out)]
        passed = len(hits) >= 1
        return {"passed": passed, "reason": f"Hits: {hits}"}

    # Hindi / Hinglish / English Conversation
    exp_kws = probe.get("expected_keywords", [])
    if exp_kws:
        hits = [k for k in exp_kws if k.lower() in norm_out]
        passed = len(hits) >= 1 and len(cleaned.split()) >= 3
        return {"passed": passed, "reason": f"Keywords: {hits}"}

    # Default rubric check
    passed = len(cleaned.split()) >= 3 and not ("?" in cleaned and len(cleaned.split()) < 5)
    return {"passed": passed, "reason": f"Length and structure check"}

def run_reproduction_study():
    probes = json.load(open(MANIFEST_PATH, encoding="utf-8"))
    all_results = {}

    for name, path in CHECKPOINTS.items():
        print(f"\n==============================================")
        print(f"Running Owner Reproduction Test Suite: {name}")
        print(f"==============================================")
        runner = DirectNeuralRunner(path)
        tok = runner.tokenizer
        model = runner.model
        model.eval()

        results = []
        cat_stats = {}
        total_passed = 0

        for p in probes:
            cat = p["category"]
            if cat not in cat_stats:
                cat_stats[cat] = {"total": 0, "passed": 0}
            cat_stats[cat]["total"] += 1

            prompt_str = f"User: {p['input']}\nAssistant:"
            p_tokens = tok.encode(prompt_str, add_bos=True, add_eos=False)
            curr = list(p_tokens)
            gen = []

            # Exact inference path: rep_penalty=1.05, no_repeat_ngram=3
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

        print(f"Total Score ({name}): {total_passed}/{len(probes)} ({(total_passed/len(probes)*100):.1f}%)")
        for c, s in cat_stats.items():
            print(f"  {c:<22}: {s['passed']}/{s['total']} ({(s['passed']/s['total']*100):.1f}%)")

        all_results[name] = {
            "total_passed": total_passed,
            "total_probes": len(probes),
            "accuracy": total_passed / len(probes) * 100.0,
            "category_breakdown": cat_stats,
            "evaluations": results
        }

    out_file = REPO_ROOT / "experiments" / "ucip09s" / "evaluations" / "baseline_reproduction_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved reproduction results to: {out_file}")

if __name__ == "__main__":
    run_reproduction_study()
