"""
experiments/fast_track_047_049/eval_rc1_rc2_sealed120.py
Runs fast batch evaluation of RC1 and RC2 on the sealed 120-prompt benchmark.
Evaluates:
- Greedy decoding (T=0.0)
- Controlled sampling (T=0.3, top_p=0.9, rep_pen=1.15)
- Pure neural output
- 4-point Rubric scoring:
    0: unrelated / nonsensical
    1: partially relevant but incorrect / incomplete
    2: mostly correct and relevant
    3: correct, relevant, and adequately explained
- Syntactic coherence, repetition rate, prompt echo rate
- Preserves raw outputs for human-readable audit
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
            
            if next_token in [1, 2]: # EOS / PAD
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
    """
    Rubric:
    0: unrelated or nonsensical
    1: partially relevant but incorrect or incomplete
    2: mostly correct and relevant
    3: correct, relevant and adequately explained
    """
    if not response or len(response.split()) < 2:
        return 0, False, False, False # score, coherent, echo, repetitive
        
    words = response.split()
    # Check prompt echo
    has_echo = ("user:" in response.lower() or "assistant:" in response.lower() or item["prompt"].lower() in response.lower())
    
    # Check repetitive stutter
    has_rep = len(words) > 6 and len(set(words)) / len(words) < 0.4
    
    # Check syntactic coherence (reasonable word lengths, no severe stutter)
    is_coherent = not has_rep and len(words) >= 2
    
    # Semantic scoring against expected and keywords
    resp_lower = response.lower()
    kw_hits = sum(1 for kw in item["rubric_keywords"] if kw.lower() in resp_lower)
    
    if kw_hits >= 2 or (len(item["rubric_keywords"]) == 1 and kw_hits == 1):
        # Strong match
        if len(words) >= 6 and not has_echo:
            score = 3
        else:
            score = 2
    elif kw_hits == 1:
        score = 1
    else:
        # Check if response has any partial topical overlap
        topical = any(w in resp_lower for w in item["expected"].lower().split() if len(w) > 4)
        score = 1 if topical else 0
        
    return score, is_coherent, has_echo, has_rep

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== EVALUATING RC1 VS RC2 ON SEALED 120 BENCHMARK ({device.upper()}) ===")
    
    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)
    
    with open("experiments/fast_track_047_049/sealed_120_benchmark.json", "r", encoding="utf-8") as f:
        benchmark = json.load(f)
        
    model_rc1 = load_model("release/v1.0.0-rc1/chakr_1_0_rc1_best.pt", device=device)
    model_rc2 = load_model("experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt", device=device)
    
    models = {
        "RC1": model_rc1,
        "RC2": model_rc2
    }
    
    # Evaluation settings: Greedy (T=0.0) and Controlled Sampling (T=0.3, p=0.9, rep_pen=1.15)
    eval_modes = [
        {"name": "greedy", "temp": 0.0, "top_p": 1.0, "rep_pen": 1.0},
        {"name": "sampling_T03", "temp": 0.3, "top_p": 0.9, "rep_pen": 1.15}
    ]
    
    results = {
        "metadata": {
            "num_prompts": len(benchmark),
            "device": device,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        },
        "summary": {},
        "per_category": {},
        "raw_evaluations": []
    }
    
    # Initialize category trackers
    categories = sorted(list(set(item["category"] for item in benchmark)))
    for m_name in models.keys():
        results["summary"][m_name] = {}
        results["per_category"][m_name] = {c: {"total": 0, "score_sum": 0, "scores": [0,0,0,0], "coherent": 0, "echo": 0} for c in categories}
        
    start_time = time.time()
    
    for idx, item in enumerate(benchmark):
        p = item["prompt"]
        cat = item["category"]
        
        prompt_record = {
            "id": item["id"],
            "category": cat,
            "prompt": p,
            "expected": item["expected"],
            "keywords": item["rubric_keywords"],
            "models": {}
        }
        
        for m_name, model in models.items():
            prompt_record["models"][m_name] = {}
            for mode in eval_modes:
                mode_name = mode["name"]
                torch.manual_seed(42)
                resp = generate_response(
                    model, tokenizer, p,
                    max_new_tokens=48,
                    temperature=mode["temp"],
                    top_p=mode["top_p"],
                    repetition_penalty=mode["rep_pen"],
                    device=device
                )
                score, coherent, echo, rep = score_response(resp, item)
                
                prompt_record["models"][m_name][mode_name] = {
                    "response": resp,
                    "score": score,
                    "coherent": coherent,
                    "prompt_echo": echo,
                    "repetitive": rep
                }
                
                # Update category stats for sampling_T03 (primary controlled mode)
                if mode_name == "sampling_T03":
                    cat_stat = results["per_category"][m_name][cat]
                    cat_stat["total"] += 1
                    cat_stat["score_sum"] += score
                    cat_stat["scores"][score] += 1
                    if coherent: cat_stat["coherent"] += 1
                    if echo: cat_stat["echo"] += 1
                    
        results["raw_evaluations"].append(prompt_record)
        if (idx + 1) % 20 == 0:
            print(f"Evaluated {idx+1}/{len(benchmark)} prompts...")
            
    total_time = time.time() - start_time
    print(f"Benchmark completed in {total_time:.1f} seconds.")
    
    # Calculate overall summary metrics
    for m_name in models.keys():
        total_items = len(benchmark)
        total_score = sum(c["score_sum"] for c in results["per_category"][m_name].values())
        max_possible = total_items * 3
        mean_score = total_score / total_items
        pct_score = (total_score / max_possible) * 100
        total_coherent = sum(c["coherent"] for c in results["per_category"][m_name].values())
        total_echo = sum(c["echo"] for c in results["per_category"][m_name].values())
        score_dist = [sum(c["scores"][s] for c in results["per_category"][m_name].values()) for s in range(4)]
        
        results["summary"][m_name] = {
            "total_prompts": total_items,
            "mean_score_out_of_3": round(mean_score, 3),
            "percentage_capability": round(pct_score, 1),
            "score_distribution": {f"score_{s}": score_dist[s] for s in range(4)},
            "coherence_rate_pct": round(total_coherent / total_items * 100, 1),
            "prompt_echo_rate_pct": round(total_echo / total_items * 100, 1)
        }
        
        print(f"\n[{m_name} Overall Performance]")
        print(f"  Mean Score: {mean_score:.2f} / 3.0 ({pct_score:.1f}%)")
        print(f"  Score Breakdown (0/1/2/3): {score_dist}")
        print(f"  Coherence: {total_coherent / total_items * 100:.1f}% | Prompt Echo: {total_echo / total_items * 100:.1f}%")
        
    out_file = Path("experiments/fast_track_047_049/NR047_SEALED_EVALUATION_RESULTS.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nResults written to: {out_file}")

if __name__ == "__main__":
    run_benchmark()
