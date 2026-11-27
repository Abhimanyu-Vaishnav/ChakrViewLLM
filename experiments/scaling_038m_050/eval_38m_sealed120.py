"""
experiments/scaling_038m_050/eval_38m_sealed120.py
Evaluates ChakrGenesis-38M against the exact same sealed 120-prompt benchmark:
- Evaluates greedy decoding (T=0.0) and controlled sampling (T=0.3, p=0.9, rep_pen=1.15)
- Applies identical 4-point semantic rubric
- Compares performance directly against RC3 and RC2
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

from chakrview.brain.genesis.chakr_genesis_38m import ChakrGenesis38M, Genesis38MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def load_38m_model(ckpt_path, device="cuda"):
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_cfg = st["config"]
    cfg = raw_cfg if isinstance(raw_cfg, Genesis38MConfig) else Genesis38MConfig(**raw_cfg)
    model = ChakrGenesis38M(cfg)
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
    print(f"=== EVALUATING CHAKRGENESIS-38M ON SEALED 120 BENCHMARK ({device.upper()}) ===")
    
    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)
    
    with open("experiments/fast_track_047_049/sealed_120_benchmark.json", "r", encoding="utf-8") as f:
        benchmark = json.load(f)
        
    model_38m = load_38m_model("experiments/scaling_038m_050/chakr_genesis_38m_best.pt", device=device)
    
    # Load prior comparison results (RC1, RC2, RC3)
    with open("experiments/fast_track_047_049/NR049_RC3_EVALUATION_RESULTS.json", "r", encoding="utf-8") as f:
        prior_comp = json.load(f)
        
    categories = sorted(list(set(item["category"] for item in benchmark)))
    m38_per_cat = {c: {"total": 0, "score_sum": 0, "scores": [0,0,0,0], "coherent": 0, "echo": 0} for c in categories}
    
    eval_results = []
    start_time = time.time()
    
    for idx, item in enumerate(benchmark):
        p = item["prompt"]
        cat = item["category"]
        
        # Controlled sampling: T=0.3, p=0.9, rep_pen=1.15
        torch.manual_seed(42)
        resp_samp = generate_response(model_38m, tokenizer, p, max_new_tokens=48, temperature=0.3, top_p=0.9, repetition_penalty=1.15, device=device)
        score_s, coh_s, echo_s, rep_s = score_response(resp_samp, item)
        
        # Greedy: T=0.0
        torch.manual_seed(42)
        resp_greedy = generate_response(model_38m, tokenizer, p, max_new_tokens=48, temperature=0.0, top_p=1.0, repetition_penalty=1.0, device=device)
        score_g, coh_g, echo_g, rep_g = score_response(resp_greedy, item)
        
        cat_stat = m38_per_cat[cat]
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
            "m38_sampling_T03": {
                "response": resp_samp,
                "score": score_s,
                "coherent": coh_s,
                "prompt_echo": echo_s
            },
            "m38_greedy": {
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
    total_score = sum(c["score_sum"] for c in m38_per_cat.values())
    max_possible = total_items * 3
    mean_score = total_score / total_items
    pct_score = (total_score / max_possible) * 100
    total_coherent = sum(c["coherent"] for c in m38_per_cat.values())
    total_echo = sum(c["echo"] for c in m38_per_cat.values())
    score_dist = [sum(c["scores"][s] for c in m38_per_cat.values()) for s in range(4)]
    
    m38_summary = {
        "model": "ChakrGenesis-38M",
        "unique_parameters": 37499392,
        "total_prompts": total_items,
        "mean_score_out_of_3": round(mean_score, 3),
        "percentage_capability": round(pct_score, 1),
        "score_distribution": {f"score_{s}": score_dist[s] for s in range(4)},
        "coherence_rate_pct": round(total_coherent / total_items * 100, 1),
        "prompt_echo_rate_pct": round(total_echo / total_items * 100, 1)
    }
    
    print("\n=== CHAKRGENESIS-38M BENCHMARK SUMMARY ===")
    print(f"Mean Score: {mean_score:.3f} / 3.0 ({pct_score:.1f}%)")
    print(f"Score Breakdown (0/1/2/3): {score_dist}")
    print(f"Coherence: {total_coherent / total_items * 100:.1f}% | Prompt Echo: {total_echo / total_items * 100:.1f}%")
    
    # Combined output comparing RC2, RC3, and 38M
    output = {
        "comparison_summary": {
            "RC1": prior_comp["comparison_summary"]["RC1"],
            "RC2": prior_comp["comparison_summary"]["RC2"],
            "RC3": prior_comp["comparison_summary"]["RC3"],
            "ChakrGenesis-38M": m38_summary
        },
        "per_category_comparison": {
            "RC2": prior_comp["per_category_comparison"]["RC2"],
            "RC3": prior_comp["per_category_comparison"]["RC3"],
            "ChakrGenesis-38M": m38_per_cat
        },
        "details_38m": eval_results
    }
    
    out_file = Path("experiments/scaling_038m_050/NR050_38M_EVALUATION_RESULTS.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        
    print(f"Scaling comparison written to {out_file}")

if __name__ == "__main__":
    run_evaluation()
