"""
evaluate_cr20.py - Comprehensive Multilingual & Multi-Turn Evaluation Suite (CR-EXP-20)

Evaluates:
- Baseline (Conversational Parent cbr_sft_04)
- Candidate Checkpoints (e.g. cr20_candidate_seed2026_best.pt, cr20_candidate_seed3030_best.pt)

Benchmarks:
1. english_100.json
2. hindi_100.json
3. roman_hindi_100.json
4. hinglish_100.json
5. context_200.json (1, 2, 3, 5, 8 turns)
6. cross_language_100.json
"""

import os
import sys
import json
import time
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

import torch
from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.runtime.sampling import SamplingConfig
from chakrview.chat import ConversationalChatCLI

BASE_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_20"
BENCH_DIR = BASE_DIR / "benchmark"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PARENT_CKPT = REPO_ROOT / "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt"

def evaluate_benchmark_file(runner, bench_file: Path, cfg: SamplingConfig):
    tok = runner.tokenizer
    with open(bench_file, "r", encoding="utf-8") as f:
        probes = json.load(f)

    hits = 0
    total = len(probes)
    details = []

    for p in probes:
        pid = p["id"]
        prompt = p["prompt"]
        exp = p["expected"]
        target = p.get("target", "")

        toks = tok.encode(prompt, add_bos=True, add_eos=False)
        curr = list(toks)
        gen = []
        with torch.no_grad():
            for step in range(40):
                t_in = torch.tensor([curr], dtype=torch.long)
                logits = runner.model(t_in)[0, -1, :]
                tok_id = runner.sampler.sample(logits, generated_tokens=gen, config=cfg, step=step)
                gen.append(tok_id)
                curr.append(tok_id)
                if tok_id == 1:
                    break

        resp = tok.decode(gen, errors="replace").strip()
        hit = any(e.lower() in resp.lower() for e in exp)
        if hit:
            hits += 1

        details.append({
            "id": pid,
            "prompt": prompt,
            "target": target,
            "expected": exp,
            "response": resp,
            "hit": hit
        })

    return {
        "benchmark": bench_file.name,
        "total": total,
        "hits": hits,
        "accuracy": hits / max(1, total),
        "details": details
    }

def evaluate_context_benchmark(runner, bench_file: Path, cfg: SamplingConfig):
    tok = runner.tokenizer
    with open(bench_file, "r", encoding="utf-8") as f:
        probes = json.load(f)

    by_horizon = {}
    details = []
    total_hits = 0

    for p in probes:
        pid = p["id"]
        h = p.get("turns_horizon", 1)
        prompt = p["prompt"]
        exp = p["expected"]
        target = p.get("target", "")

        if h not in by_horizon:
            by_horizon[h] = {"total": 0, "hits": 0}

        toks = tok.encode(prompt, add_bos=True, add_eos=False)
        curr = list(toks)
        gen = []
        with torch.no_grad():
            for step in range(40):
                t_in = torch.tensor([curr], dtype=torch.long)
                logits = runner.model(t_in)[0, -1, :]
                tok_id = runner.sampler.sample(logits, generated_tokens=gen, config=cfg, step=step)
                gen.append(tok_id)
                curr.append(tok_id)
                if tok_id == 1:
                    break

        resp = tok.decode(gen, errors="replace").strip()
        hit = any(e.lower() in resp.lower() for e in exp)

        by_horizon[h]["total"] += 1
        if hit:
            by_horizon[h]["hits"] += 1
            total_hits += 1

        details.append({
            "id": pid,
            "turns_horizon": h,
            "expected": exp,
            "response": resp,
            "hit": hit
        })

    return {
        "benchmark": bench_file.name,
        "total": len(probes),
        "total_hits": total_hits,
        "accuracy": total_hits / max(1, len(probes)),
        "by_horizon": {
            k: {
                "total": v["total"],
                "hits": v["hits"],
                "acc": v["hits"] / max(1, v["total"])
            }
            for k, v in sorted(by_horizon.items())
        },
        "details": details
    }

def main():
    benchmarks_single = [
        BENCH_DIR / "english_100.json",
        BENCH_DIR / "hindi_100.json",
        BENCH_DIR / "roman_hindi_100.json",
        BENCH_DIR / "hinglish_100.json",
        BENCH_DIR / "cross_language_100.json"
    ]
    context_bench = BENCH_DIR / "context_200.json"

    candidates = [
        {"name": "parent_baseline", "path": PARENT_CKPT},
        {"name": "cr20_candidate_seed2026", "path": BASE_DIR / "checkpoints" / "cr20_candidate_seed2026_best.pt"},
    ]

    seed3030_pt = BASE_DIR / "checkpoints" / "cr20_candidate_seed3030_best.pt"
    if seed3030_pt.exists():
        candidates.append({"name": "cr20_candidate_seed3030", "path": seed3030_pt})

    cfg = SamplingConfig(temperature=0.0, repetition_penalty=1.15, no_repeat_ngram_size=3)
    all_results = {}

    for cand in candidates:
        name = cand["name"]
        ckpt_path = cand["path"]
        print(f"\n=======================================================")
        print(f"  EVALUATING: {name}")
        print(f"  Checkpoint: {ckpt_path.name}")
        print(f"=======================================================")
        t0 = time.time()
        runner = DirectNeuralRunner(ckpt_path)

        cand_summary = {}

        # 1. Evaluate single-turn multilingual benchmarks
        for bfile in benchmarks_single:
            res = evaluate_benchmark_file(runner, bfile, cfg)
            cand_summary[bfile.stem] = {
                "hits": res["hits"],
                "total": res["total"],
                "acc": res["accuracy"]
            }
            print(f"  {bfile.stem:20s}: {res['hits']:3d}/{res['total']} ({res['accuracy']*100:.1f}%)")

        # 2. Evaluate context_200.json
        ctx_res = evaluate_context_benchmark(runner, context_bench, cfg)
        cand_summary["context_200"] = {
            "total_hits": ctx_res["total_hits"],
            "total": ctx_res["total"],
            "accuracy": ctx_res["accuracy"],
            "by_horizon": ctx_res["by_horizon"]
        }
        print(f"  {'context_200':20s}: {ctx_res['total_hits']:3d}/{ctx_res['total']} ({ctx_res['accuracy']*100:.1f}%)")
        for h, d in ctx_res["by_horizon"].items():
            print(f"    Horizon {h} turns: {d['hits']:2d}/{d['total']} ({d['acc']*100:.1f}%)")

        elapsed = time.time() - t0
        cand_summary["elapsed_sec"] = round(elapsed, 2)
        all_results[name] = cand_summary
        print(f"Candidate {name} evaluation finished in {elapsed:.1f}s")

    with open(RESULTS_DIR / "cr20_evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved all evaluation results to results/cr20_evaluation_results.json")

if __name__ == "__main__":
    main()

