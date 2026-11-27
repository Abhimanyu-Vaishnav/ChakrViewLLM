"""
experiments/fast_track_047_049/eval_rc3_sealed120.py
Evaluates the new RC3 candidate model against the exact same sealed 120-prompt benchmark:
- Compares RC2 vs RC3 under identical decoding conditions (sampling_T03 and greedy)
- 4-point Rubric scoring across all 8 functional categories
- Measures category improvements, regressions, and capability deltas
"""

import sys
import os
import json
import time
import math
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
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def load_model(ckpt_path, device="cuda"):
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_cfg = st["config"]
    cfg = raw_cfg if isinstance(raw_cfg, Genesis17MConfig) else Genesis17MConfig(**raw_cfg)
    model = ChakrGenesis17M(cfg)
    sd = st.get("model_state_dict", st.get("model", st))
    model.load_state_dict(sd)
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

def run_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== EVALUATING RC3 ON SEALED 120 BENCHMARK ({device.upper()}) ===")
    
    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)
    
    with open("experiments/fast_track_047_049/sealed_120_benchmark.json", "r", encoding="utf-8") as f:
        benchmark = json.load(f)
        
    model_rc3 = load_model("experiments/fast_track_047_049/chakr_1_0_rc3_best.pt", device=device)
    
    # Load previous RC2 benchmark results
    with open("experiments/fast_track_047_049/NR047_SEALED_EVALUATION_RESULTS.json", "r", encoding="utf-8") as f:
        prior_res = json.load(f)
        
    categories = sorted(list(set(item["category"] for item in benchmark)))
    rc3_per_cat = {c: {"total": 0, "score_sum": 0, "scores": [0,0,0,0], "coherent": 0, "echo": 0} for c in categories}
    
    eval_results = []
    start_time = time.time()
    
    for idx, item in enumerate(benchmark):
        p = item["prompt"]
        cat = item["category"]
        
        # Controlled sampling: T=0.3, p=0.9, rep_pen=1.15
        torch.manual_seed(42)
        resp_samp = generate_response(model_rc3, tokenizer, p, max_new_tokens=48, temperature=0.3, top_p=0.9, repetition_penalty=1.15, device=device)
        score_s, coh_s, echo_s, rep_s = score_response(resp_samp, item)
        
        # Greedy: T=0.0
        torch.manual_seed(42)
        resp_greedy = generate_response(model_rc3, tokenizer, p, max_new_tokens=48, temperature=0.0, top_p=1.0, repetition_penalty=1.0, device=device)
        score_g, coh_g, echo_g, rep_g = score_response(resp_greedy, item)
        
        cat_stat = rc3_per_cat[cat]
        cat_stat["total"] += 1
        cat_stat["score_sum"] += score_s
        cat_stat["scores"][score_s] += 1
        if coh_s: cat_stat["coherent"] += 1
        if echo_s: cat_stat["echo"] += 1
        
        eval_results.append({
            "id": item["id"],
            "category": cat,
            "prompt": p,
            "expected": item["expected"],
            "keywords": item["rubric_keywords"],
            "rc3_sampling_T03": {
                "response": resp_samp,
                "score": score_s,
                "coherent": coh_s,
                "prompt_echo": echo_s
            },
            "rc3_greedy": {
                "response": resp_greedy,
                "score": score_g,
                "coherent": coh_g,
                "prompt_echo": echo_g
            }
        })
        
        if (idx + 1) % 20 == 0:
            print(f"Evaluated {idx+1}/{len(benchmark)} prompts...")
            
    total_time = time.time() - start_time
    total_items = len(benchmark)
    total_score = sum(c["score_sum"] for c in rc3_per_cat.values())
    max_possible = total_items * 3
    mean_score = total_score / total_items
    pct_score = (total_score / max_possible) * 100
    total_coherent = sum(c["coherent"] for c in rc3_per_cat.values())
    total_echo = sum(c["echo"] for c in rc3_per_cat.values())
    score_dist = [sum(c["scores"][s] for c in rc3_per_cat.values()) for s in range(4)]
    
    rc3_summary = {
        "total_prompts": total_items,
        "mean_score_out_of_3": round(mean_score, 3),
        "percentage_capability": round(pct_score, 1),
        "score_distribution": {f"score_{s}": score_dist[s] for s in range(4)},
        "coherence_rate_pct": round(total_coherent / total_items * 100, 1),
        "prompt_echo_rate_pct": round(total_echo / total_items * 100, 1)
    }
    
    print("\n=== RC3 SUMMARY ON SEALED 120 BENCHMARK ===")
    print(f"Mean Score: {mean_score:.3f} / 3.0 ({pct_score:.1f}%)")
    print(f"Score Breakdown (0/1/2/3): {score_dist}")
    print(f"Coherence: {total_coherent / total_items * 100:.1f}% | Prompt Echo: {total_echo / total_items * 100:.1f}%")
    
    # Save combined comparison
    comparison_output = {
        "comparison_summary": {
            "RC1": prior_res["summary"]["RC1"],
            "RC2": prior_res["summary"]["RC2"],
            "RC3": rc3_summary
        },
        "per_category_comparison": {
            "RC2": prior_res["per_category"]["RC2"],
            "RC3": rc3_per_cat
        },
        "rc3_details": eval_results
    }
    
    out_file = Path("experiments/fast_track_047_049/NR049_RC3_EVALUATION_RESULTS.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison_output, f, indent=2, ensure_ascii=False)
        
    print(f"Comparison saved to: {out_file}")

if __name__ == "__main__":
    run_evaluation()
