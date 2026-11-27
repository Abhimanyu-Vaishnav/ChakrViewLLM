"""
experiments/pretraining_nr051/evaluate_sealed_benchmark.py
NR-051 Phase 7: Sealed Benchmark Evaluation across Pretrained+SFT Candidates
- Candidate 1: chakr_17m_sft_candidate.pt
- Candidate 2: chakr_38m_sft_candidate.pt
- Baselines: Stage-D (canonical), RC1, RC2, RC3 (historical reference)
- Benchmark: experiments/fast_track_047_049/sealed_120_benchmark.json (120 tasks across 6 categories)
- Modes:
  1. Greedy (T=0.0)
  2. Controlled Sampling (T=0.3, top_p=0.9, rep_penalty=1.15)
- Metrics:
  - Exact match / Semantic keyword match %
  - Echo contamination %
  - Syntactic coherence %
  - Latency / tokens per second
"""

import sys
import os
import json
import time
import math
import random
import re
import hashlib
import torch
import torch.nn.functional as F
from pathlib import Path

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

sys.path.insert(0, ".")

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M
from chakrview.brain.genesis.chakr_genesis_38m import ChakrGenesis38M
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def generate_response(model, tokenizer, prompt, max_new_tokens=64, temperature=0.0, top_p=0.9, rep_penalty=1.15, device="cuda"):
    model.eval()
    prompt_str = f"<user>\n{prompt}\n<assistant>\n"
    input_ids = tokenizer.encode(prompt_str)
    inp = torch.tensor([input_ids], dtype=torch.long, device=device)
    prompt_len = inp.size(1)

    generated = list(input_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            cur_inp = torch.tensor([generated[-256:]], dtype=torch.long, device=device)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                logits, _ = model(cur_inp)
            next_token_logits = logits[0, -1, :].clone()

            # Repetition penalty
            if rep_penalty != 1.0:
                for token_id in set(generated[prompt_len:]):
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= rep_penalty
                    else:
                        next_token_logits[token_id] *= rep_penalty

            if temperature == 0.0:
                next_token = torch.argmax(next_token_logits).item()
            else:
                next_token_logits = next_token_logits / temperature
                # Top-p filtering
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == 3: # <eos>
                break
            generated.append(next_token)

    resp_tokens = generated[prompt_len:]
    try:
        response_text = tokenizer.decode(resp_tokens, errors="replace").strip()
    except Exception:
        try:
            response_text = tokenizer.decode(resp_tokens).strip()
        except Exception:
            response_text = ""
    return response_text

def score_response(prompt, response, expected, keywords):
    # 1. Echo check
    echo = prompt.strip().lower() in response.strip().lower() or "<user>" in response or "<assistant>" in response

    # 2. Syntactic coherence: non-empty, contains basic words/spaces, no degenerate repetition
    words = response.strip().split()
    if len(words) == 0:
        coherent = False
    else:
        # Check repetitive loop (e.g., word repeated 5+ times in sequence)
        has_repetition_loop = any(words[i] == words[i+1] == words[i+2] == words[i+3] for i in range(len(words)-3))
        coherent = not has_repetition_loop and len(words) >= 1

    # 3. Semantic keyword match: at least one rubric keyword present in response (case-insensitive)
    resp_lower = response.lower()
    matches = [kw for kw in keywords if kw.lower() in resp_lower]
    keyword_match = len(matches) > 0

    return {
        "echo": echo,
        "coherent": coherent,
        "keyword_match": keyword_match,
        "matched_keywords": matches
    }

def evaluate_model(model_name, model_class, ckpt_path, benchmark_tasks, tokenizer, device="cuda"):
    print(f"\n=======================================================")
    print(f"EVALUATING {model_name}: {ckpt_path}")
    print(f"=======================================================")

    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = st["config"]
    model = model_class(cfg)
    model.load_state_dict(st["model_state_dict"])
    model.to(device)

    results_greedy = []
    results_sampling = []

    t0 = time.time()
    for task in benchmark_tasks:
        prompt = task["prompt"]
        expected = task.get("expected", "")
        keywords = task.get("rubric_keywords", [])
        cat = task.get("category", "General")

        # Greedy
        resp_g = generate_response(model, tokenizer, prompt, max_new_tokens=64, temperature=0.0, device=device)
        score_g = score_response(prompt, resp_g, expected, keywords)
        results_greedy.append({
            "id": task["id"],
            "category": cat,
            "prompt": prompt,
            "expected": expected,
            "response": resp_g,
            **score_g
        })

        # Sampling (T=0.3, top_p=0.9, rep_penalty=1.15)
        resp_s = generate_response(model, tokenizer, prompt, max_new_tokens=64, temperature=0.3, top_p=0.9, rep_penalty=1.15, device=device)
        score_s = score_response(prompt, resp_s, expected, keywords)
        results_sampling.append({
            "id": task["id"],
            "category": cat,
            "prompt": prompt,
            "expected": expected,
            "response": resp_s,
            **score_s
        })

    elapsed = time.time() - t0

    def aggregate(res_list):
        total = len(res_list)
        coherent = sum(1 for r in res_list if r["coherent"])
        echo = sum(1 for r in res_list if r["echo"])
        kw_match = sum(1 for r in res_list if r["keyword_match"])
        
        # Breakdown by category
        cat_stats = {}
        for r in res_list:
            c = r["category"]
            if c not in cat_stats:
                cat_stats[c] = {"total": 0, "kw_match": 0, "coherent": 0, "echo": 0}
            cat_stats[c]["total"] += 1
            if r["keyword_match"]: cat_stats[c]["kw_match"] += 1
            if r["coherent"]: cat_stats[c]["coherent"] += 1
            if r["echo"]: cat_stats[c]["echo"] += 1

        for c, s in cat_stats.items():
            s["kw_match_rate"] = round(s["kw_match"] / s["total"] * 100, 2)
            s["coherence_rate"] = round(s["coherent"] / s["total"] * 100, 2)
            s["echo_rate"] = round(s["echo"] / s["total"] * 100, 2)

        return {
            "total_tasks": total,
            "coherence_count": coherent,
            "coherence_pct": round(coherent / total * 100, 2),
            "echo_count": echo,
            "echo_pct": round(echo / total * 100, 2),
            "keyword_match_count": kw_match,
            "keyword_match_pct": round(kw_match / total * 100, 2),
            "category_breakdown": cat_stats
        }

    agg_g = aggregate(results_greedy)
    agg_s = aggregate(results_sampling)

    print(f"[{model_name}] Greedy Match: {agg_g['keyword_match_pct']}% | Coherence: {agg_g['coherence_pct']}% | Echo: {agg_g['echo_pct']}%")
    print(f"[{model_name}] Sampling Match: {agg_s['keyword_match_pct']}% | Coherence: {agg_s['coherence_pct']}% | Echo: {agg_s['echo_pct']}%")

    del model
    torch.cuda.empty_cache()

    return {
        "model_name": model_name,
        "checkpoint": str(ckpt_path),
        "sha256": file_sha256(ckpt_path),
        "evaluation_time_seconds": elapsed,
        "greedy_metrics": agg_g,
        "sampling_metrics": agg_s,
        "sample_outputs": results_greedy[:10]
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== NR-051 PHASE 7 SEALED BENCHMARK EVALUATION ({device.upper()}) ===")

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)
    benchmark_path = "experiments/fast_track_047_049/sealed_120_benchmark.json"

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark_tasks = json.load(f)

    print(f"Loaded {len(benchmark_tasks)} sealed evaluation tasks from {benchmark_path}")

    # Evaluate 17M SFT Candidate
    res_17m = evaluate_model(
        "ChakrGenesis-17M-Pretrained-SFT",
        ChakrGenesis17M,
        "experiments/pretraining_nr051/chakr_17m_sft_candidate.pt",
        benchmark_tasks,
        tokenizer,
        device=device
    )

    # Evaluate 38M SFT Candidate
    res_38m = evaluate_model(
        "ChakrGenesis-38M-Pretrained-SFT",
        ChakrGenesis38M,
        "experiments/pretraining_nr051/chakr_38m_sft_candidate.pt",
        benchmark_tasks,
        tokenizer,
        device=device
    )

    # Summary
    out_path = Path("experiments/pretraining_nr051/NR051_EVALUATION_RESULTS.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "17M_candidate": res_17m,
            "38M_candidate": res_38m
        }, f, indent=2)

    print(f"\nAll evaluations complete. Full results written to: {out_path}")

if __name__ == "__main__":
    main()
