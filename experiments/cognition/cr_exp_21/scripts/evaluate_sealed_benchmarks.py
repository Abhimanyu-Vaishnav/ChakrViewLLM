"""
evaluate_sealed_benchmarks.py - Evaluates 1,400 sealed probes across all candidates:
- Parent Baseline (cbr_sft_04)
- v0.10.1 (Starting Candidate)
- v0.11 (CR-21 Candidate)

Benchmarks:
1. sealed_english_150.json
2. sealed_devanagari_150.json
3. sealed_roman_hindi_150.json
4. sealed_hinglish_150.json
5. sealed_cross_language_100.json
6. sealed_context_200.json
7. sealed_entity_binding_200.json
8. sealed_user_correction_100.json
9. sealed_instruction_following_100.json
10. sealed_uncertainty_abstention_100.json
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

BENCH_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_21" / "evaluation"
RESULTS_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_21" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    ("parent_baseline", REPO_ROOT / "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt"),
    ("v0101_candidate", REPO_ROOT / "releases/v0.10.1_conversational_model/chakrview_v0101_model.pt"),
    ("v011_candidate", REPO_ROOT / "experiments/cognition/cr_exp_21/training/v011_candidate_best.pt")
]

def main():
    bench_files = [
        "sealed_english_150.json",
        "sealed_devanagari_150.json",
        "sealed_roman_hindi_150.json",
        "sealed_hinglish_150.json",
        "sealed_cross_language_100.json",
        "sealed_context_200.json",
        "sealed_entity_binding_200.json",
        "sealed_user_correction_100.json",
        "sealed_instruction_following_100.json",
        "sealed_uncertainty_abstention_100.json"
    ]

    cfg = SamplingConfig(temperature=0.0, repetition_penalty=1.15, no_repeat_ngram_size=3)
    all_eval_results = {}

    for cand_name, ckpt_path in CANDIDATES:
        print(f"\n=======================================================")
        print(f"  EVALUATING ON 1,400 SEALED PROBES: {cand_name}")
        print(f"  Checkpoint: {ckpt_path.name}")
        print(f"=======================================================")
        runner = DirectNeuralRunner(ckpt_path)
        tok = runner.tokenizer

        cand_metrics = {}
        t0 = time.time()

        for bfile_name in bench_files:
            bpath = BENCH_DIR / bfile_name
            with open(bpath, "r", encoding="utf-8") as f:
                probes = json.load(f)

            hits_exact = 0
            hits_normalized = 0
            total = len(probes)

            for p in probes:
                prompt = p["prompt"]
                expected = [e.lower() for e in p["expected"]]

                toks = tok.encode(prompt, add_bos=True, add_eos=False)
                curr = list(toks)
                gen = []
                with torch.no_grad():
                    for step in range(35):
                        t_in = torch.tensor([curr], dtype=torch.long)
                        logits = runner.model(t_in)[0, -1, :]
                        tok_id = runner.sampler.sample(logits, generated_tokens=gen, config=cfg, step=step)
                        gen.append(tok_id)
                        curr.append(tok_id)
                        if tok_id == 1:
                            break

                resp = tok.decode(gen, errors="replace").strip().lower()
                
                # Metric 1: Exact containment
                if any(e in resp for e in expected):
                    hits_exact += 1
                
                # Metric 2: Normalized subword/word match
                resp_words = [w.strip(".,!?\"' ") for w in resp.split()]
                if any(any(e in rw for e in expected) for rw in resp_words):
                    hits_normalized += 1

            cand_metrics[bfile_name] = {
                "total": total,
                "hits_exact": hits_exact,
                "acc_exact": hits_exact / max(1, total),
                "hits_normalized": hits_normalized,
                "acc_normalized": hits_normalized / max(1, total)
            }
            print(f"  {bfile_name:38s}: Exact {hits_exact:3d}/{total} ({hits_exact/max(1,total)*100:.1f}%) | Norm {hits_normalized:3d}/{total} ({hits_normalized/max(1,total)*100:.1f}%)")

        elapsed = time.time() - t0
        cand_metrics["total_elapsed_sec"] = round(elapsed, 2)
        all_eval_results[cand_name] = cand_metrics
        print(f"Candidate {cand_name} finished in {elapsed:.1f}s")

    out_file = RESULTS_DIR / "sealed_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_eval_results, f, indent=2)

    print(f"\nSaved complete sealed benchmark results to {out_file}")

if __name__ == "__main__":
    main()

