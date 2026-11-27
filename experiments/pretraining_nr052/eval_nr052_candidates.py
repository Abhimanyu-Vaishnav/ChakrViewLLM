"""
experiments/pretraining_nr052/eval_nr052_candidates.py

Phase 7: Independent Semantic Evaluation on Sealed 120-Prompt Benchmark
Evaluates NR-052 candidates against historical baselines:
- NR-052 17M SFT candidate (chakr_17m_nr052_sft.pt)
- NR-052 38M SFT candidate (chakr_38m_nr052_sft.pt)
- Compares directly against NR-051 (17M & 38M) and RC3 baselines
- Applies identical decoding (T=0.3, top_p=0.9, rep_pen=1.15 and greedy T=0.0)
- Applies transparent 4-point rubric (0 to 3) across 8 functional categories
- Measures Syntactic Coherence, Prompt Echo, Semantic Keyword Hits, and Per-Category Capability
- Outputs experiments/pretraining_nr052/NR052_BENCHMARK_RESULTS.json
"""

import sys
import os
import json
import time
import random
import torch
import torch.nn.functional as F
from pathlib import Path

random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

sys.path.insert(0, ".")

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.brain.genesis.chakr_genesis_38m import ChakrGenesis38M, Genesis38MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def load_17m(ckpt_path, device="cuda"):
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_cfg = st.get("config", st.get("cfg"))
    cfg = raw_cfg if isinstance(raw_cfg, Genesis17MConfig) else Genesis17MConfig(**raw_cfg)
    model = ChakrGenesis17M(cfg)
    model.load_state_dict(st["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def load_38m(ckpt_path, device="cuda"):
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_cfg = st.get("config", st.get("cfg"))
    cfg = raw_cfg if isinstance(raw_cfg, Genesis38MConfig) else Genesis38MConfig(**raw_cfg)
    model = ChakrGenesis38M(cfg)
    model.load_state_dict(st["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def generate_response(model, tokenizer, prompt, max_new_tokens=48, temperature=0.0, top_p=1.0, repetition_penalty=1.0, device="cuda"):
    formatted = f"<user>\n{prompt}\n<assistant>\n"
    tokens = tokenizer.encode(formatted)
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_inp = input_ids[:, -256:] if input_ids.shape[1] > 256 else input_ids
            logits, _ = model(curr_inp)
            next_logits = logits[0, -1, :].clone()
            
            if repetition_penalty != 1.0:
                for token_id in set(generated):
                    if next_logits[token_id] > 0:
                        next_logits[token_id] /= repetition_penalty
                    else:
                        next_logits[token_id] *= repetition_penalty
                        
            if temperature == 0.0:
                next_token = torch.argmax(next_logits).item()
            else:
                next_logits = next_logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_logits[indices_to_remove] = -float("Inf")
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
            generated.append(next_token)
            
            if next_token in [1, 2]:
                break
            if len(generated) >= 4 and generated[-4:] == [63, 4082, 118, 65]:
                break
            try:
                dec = tokenizer.decode(generated, errors="ignore")
                if "<user>" in dec or "<assistant>" in dec:
                    break
            except Exception:
                pass
                
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            
    try:
        raw_text = tokenizer.decode(generated, errors="replace")
    except Exception:
        raw_text = ""
        
    clean_text = raw_text.replace("<eos>", "").replace("<user>", "").replace("<assistant>", "").strip()
    return clean_text

def score_response(response, item):
    if not response or len(response.split()) < 2:
        return 0, False, False, False
        
    words = response.split()
    has_echo = ("user:" in response.lower() or "assistant:" in response.lower() or item["prompt"].lower() in response.lower())
    has_rep = len(words) > 6 and len(set(words)) / len(words) < 0.4
    is_coherent = not has_rep and len(words) >= 2
    
    resp_lower = response.lower()
    kw_hits = sum(1 for kw in item["rubric_keywords"] if kw.lower() in resp_lower)
    
    if kw_hits >= 2 or (len(item["rubric_keywords"]) == 1 and kw_hits == 1):
        if len(words) >= 6 and not has_echo:
            score = 3
        else:
            score = 2
    elif kw_hits == 1:
        score = 1
    else:
        topical = any(w in resp_lower for w in item["expected"].lower().split() if len(w) > 4)
        score = 1 if topical else 0
        
    return score, is_coherent, has_echo, has_rep

def evaluate_model(model, model_name, tokenizer, benchmark, device="cuda"):
    print(f"\nEvaluating {model_name} on Sealed 120 Benchmark...", flush=True)
    categories = sorted(list(set(item["category"] for item in benchmark)))
    per_cat = {c: {"total": 0, "score_sum": 0, "scores": [0,0,0,0], "coherent": 0, "echo": 0} for c in categories}
    results = []
    
    start_time = time.time()
    for idx, item in enumerate(benchmark):
        p = item["prompt"]
        cat = item["category"]
        
        # Sampling T=0.3
        torch.manual_seed(42)
        resp_samp = generate_response(model, tokenizer, p, max_new_tokens=48, temperature=0.3, top_p=0.9, repetition_penalty=1.15, device=device)
        score_s, coh_s, echo_s, rep_s = score_response(resp_samp, item)
        
        # Greedy T=0.0
        torch.manual_seed(42)
        resp_greedy = generate_response(model, tokenizer, p, max_new_tokens=48, temperature=0.0, top_p=1.0, repetition_penalty=1.0, device=device)
        score_g, coh_g, echo_g, rep_g = score_response(resp_greedy, item)
        
        cat_stat = per_cat[cat]
        cat_stat["total"] += 1
        cat_stat["score_sum"] += score_s
        cat_stat["scores"][score_s] += 1
        if coh_s: cat_stat["coherent"] += 1
        if echo_s: cat_stat["echo"] += 1
        
        results.append({
            "id": item["id"],
            "category": cat,
            "prompt": p,
            "expected": item["expected"],
            "keywords": item["rubric_keywords"],
            "sampling_T03": {
                "response": resp_samp,
                "score": score_s,
                "coherent": coh_s,
                "prompt_echo": echo_s
            },
            "greedy": {
                "response": resp_greedy,
                "score": score_g,
                "coherent": coh_g,
                "prompt_echo": echo_g
            }
        })
        
        if (idx + 1) % 30 == 0:
            print(f"  [{model_name}] Evaluated {idx+1}/{len(benchmark)} prompts...", flush=True)
            
    total_time = time.time() - start_time
    total_items = len(benchmark)
    total_score = sum(c["score_sum"] for c in per_cat.values())
    max_possible = total_items * 3
    mean_score = total_score / total_items
    pct_score = (total_score / max_possible) * 100
    total_coherent = sum(c["coherent"] for c in per_cat.values())
    total_echo = sum(c["echo"] for c in per_cat.values())
    score_dist = [sum(c["scores"][s] for c in per_cat.values()) for s in range(4)]
    
    summary = {
        "model_name": model_name,
        "eval_time_sec": round(total_time, 2),
        "total_prompts": total_items,
        "mean_score_out_of_3": round(mean_score, 3),
        "percentage_capability": round(pct_score, 1),
        "score_distribution": {f"score_{s}": score_dist[s] for s in range(4)},
        "coherence_rate_pct": round(total_coherent / total_items * 100, 1),
        "prompt_echo_rate_pct": round(total_echo / total_items * 100, 1),
        "per_category": per_cat
    }
    
    print(f"[{model_name}] Complete: Mean = {mean_score:.3f}/3.0 ({pct_score:.1f}%), Coherence = {summary['coherence_rate_pct']}%, Echo = {summary['prompt_echo_rate_pct']}%", flush=True)
    return summary, results

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== PHASE 7: NR-052 SEALED BENCHMARK EVALUATION ({device.upper()}) ===", flush=True)

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    with open("experiments/fast_track_047_049/sealed_120_benchmark.json", "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # Load historical results for baseline comparisons
    with open("experiments/pretraining_nr051/NR051_BENCHMARK_RESULTS.json", "r", encoding="utf-8") as f:
        nr051_data = json.load(f)

    rc3_summary = nr051_data["summary_comparison"]["RC3_17M_NoPretrain"]
    nr051_17m_summary = nr051_data["summary_comparison"]["NR051_17M_Pretrained_SFT"]
    nr051_38m_summary = nr051_data["summary_comparison"]["NR051_38M_Pretrained_SFT"]

    # 1. Evaluate NR-052 17M SFT candidate
    model_17m = load_17m("experiments/pretraining_nr052/chakr_17m_nr052_sft.pt", device=device)
    summary_17m, details_17m = evaluate_model(model_17m, "ChakrGenesis-17M-NR052-SFT", tokenizer, benchmark, device=device)
    del model_17m
    torch.cuda.empty_cache()

    # 2. Evaluate NR-052 38M SFT candidate
    model_38m = load_38m("experiments/pretraining_nr052/chakr_38m_nr052_sft.pt", device=device)
    summary_38m, details_38m = evaluate_model(model_38m, "ChakrGenesis-38M-NR052-SFT", tokenizer, benchmark, device=device)
    del model_38m
    torch.cuda.empty_cache()

    out_comparison = {
        "benchmark_meta": {
            "file": "experiments/fast_track_047_049/sealed_120_benchmark.json",
            "total_prompts": len(benchmark),
            "decoding": "temperature=0.3, top_p=0.9, repetition_penalty=1.15"
        },
        "summary_comparison": {
            "RC3_Historical_Baseline": rc3_summary,
            "NR051_17M_SFT": nr051_17m_summary,
            "NR051_38M_SFT": nr051_38m_summary,
            "NR052_17M_SFT": summary_17m,
            "NR052_38M_SFT": summary_38m
        },
        "deltas": {
            "38M_vs_17M_NR052": round(summary_38m["mean_score_out_of_3"] - summary_17m["mean_score_out_of_3"], 3),
            "NR052_vs_NR051_17M": round(summary_17m["mean_score_out_of_3"] - nr051_17m_summary["mean_score_out_of_3"], 3),
            "NR052_vs_NR051_38M": round(summary_38m["mean_score_out_of_3"] - nr051_38m_summary["mean_score_out_of_3"], 3)
        },
        "per_category_comparison": {
            "NR052_17M": summary_17m["per_category"],
            "NR052_38M": summary_38m["per_category"]
        },
        "detailed_completions": {
            "17M": details_17m,
            "38M": details_38m
        }
    }

    out_path = Path("experiments/pretraining_nr052/NR052_BENCHMARK_RESULTS.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_comparison, f, indent=2, ensure_ascii=False)

    print(f"\nBenchmark Evaluation complete! Results saved to {out_path}")
    print("\n================ FINAL COMPARISON SUMMARY ================")
    print(f"RC3 (Historical Baseline):   Mean = {rc3_summary['mean_score_out_of_3']:.3f}/3.0 ({rc3_summary['percentage_capability']}%)")
    print(f"NR-051 17M SFT:             Mean = {nr051_17m_summary['mean_score_out_of_3']:.3f}/3.0 ({nr051_17m_summary['percentage_capability']}%)")
    print(f"NR-051 38M SFT:             Mean = {nr051_38m_summary['mean_score_out_of_3']:.3f}/3.0 ({nr051_38m_summary['percentage_capability']}%)")
    print(f"NR-052 17M SFT:             Mean = {summary_17m['mean_score_out_of_3']:.3f}/3.0 ({summary_17m['percentage_capability']}%)")
    print(f"NR-052 38M SFT:             Mean = {summary_38m['mean_score_out_of_3']:.3f}/3.0 ({summary_38m['percentage_capability']}%)")
    print("==========================================================")

if __name__ == "__main__":
    main()
