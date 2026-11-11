"""
Phase B Evaluator: Evaluate Candidate Checkpoints on CONVERSATION_BENCHMARK_V1.json
Evaluates:
1. Canonical Stage B
2. CR-16A Arm D
3. Conversational SFT Parent (CBR-SFT-04 step 200)
4. Conversational Pilot Best (autonomous 12h)
5. CBR-SFT-07 Step 50
6. CBR-SFT-08 Step 60
Saves metrics and raw responses to CR17_BASELINE_EVALUATION.md and baseline_eval_results.json.
"""

import sys
import json
import hashlib
from pathlib import Path
import torch

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ModelConfig, ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
tok, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

BENCHMARK_PATH = REPO_ROOT / "experiments" / "conversation" / "cr_exp_17" / "CONVERSATION_BENCHMARK_V1.json"
with open(BENCHMARK_PATH, encoding="utf-8") as f:
    benchmark = json.load(f)

checkpoints = {
    "Stage B (Canonical Step 10k)": REPO_ROOT / "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
    "CR-16A Arm D (Copy Parent)": REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/checkpoints/arm_d_generalization/arm_d_generalization_best.pt",
    "Conversational SFT Parent (Step 200)": REPO_ROOT / "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt",
    "Conversational Pilot Best": REPO_ROOT / "experiments/cognition/autonomous_12h/conversational_pilot/checkpoints/conversational_pilot_best_val.pt",
    "CBR-SFT-07 (Step 50)": REPO_ROOT / "experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_50.pt",
    "CBR-SFT-08 (Step 60)": REPO_ROOT / "experiments/cbr01_sft/runs/cbr_sft_08/checkpoints/sft_checkpoint_step_60.pt"
}

def load_model(ckpt_path):
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, max_seq_len=512, tie_embeddings=True)
    m = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m

def generate_calibrated(model, prompt_str, max_tokens=35, rep_pen=1.15, no_repeat=3):
    input_ids = tok.encode(prompt_str)
    curr_ids = list(input_ids)
    gen_ids = []
    
    stop_phrases = ["\nUser:", "User:", "\nAssistant:", "Assistant:"]
    
    with torch.no_grad():
        for step in range(max_tokens):
            if len(curr_ids) >= 510:
                break
            x = torch.tensor([curr_ids], dtype=torch.long)
            logits = model(x)[0, -1, :]
            
            # Repetition penalty
            if rep_pen > 1.0:
                for g in set(gen_ids):
                    if logits[g] < 0:
                        logits[g] *= rep_pen
                    else:
                        logits[g] /= rep_pen
                        
            # No repeat n-gram
            if no_repeat > 0 and len(gen_ids) >= no_repeat:
                ngrams = {}
                for i in range(len(gen_ids) - no_repeat + 1):
                    prev = tuple(gen_ids[i:i + no_repeat - 1])
                    nxt = gen_ids[i + no_repeat - 1]
                    ngrams.setdefault(prev, set()).add(nxt)
                curr_pref = tuple(gen_ids[-(no_repeat - 1):])
                if curr_pref in ngrams:
                    for banned in ngrams[curr_pref]:
                        logits[banned] = -float("inf")
                        
            nxt = int(torch.argmax(logits).item())
            if nxt == 1: # EOS
                break
            curr_ids.append(nxt)
            gen_ids.append(nxt)
            
            text = tok.decode(gen_ids, errors="replace")
            should_stop = False
            for sp in stop_phrases:
                if sp in text:
                    should_stop = True
                    break
            if should_stop:
                break
                
    response = tok.decode(gen_ids, errors="replace")
    for sp in stop_phrases:
        if sp in response:
            response = response.split(sp)[0]
    return response.strip()

all_eval_results = {}

for name, path in checkpoints.items():
    print(f"\n==========================================")
    print(f"Evaluating: {name}")
    print(f"Path: {path}")
    print(f"==========================================")
    
    model = load_model(path)
    cat_stats = {}
    records = []
    
    total_keyword_hits = 0
    total_non_empty = 0
    total_no_repetition = 0
    
    for probe in benchmark:
        cat = probe["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "hits": 0, "non_empty": 0, "no_rep": 0}
            
        cat_stats[cat]["total"] += 1
        resp = generate_calibrated(model, probe["prompt"])
        
        # Keyword hit check (case-insensitive substring match of any expected keyword)
        kw_hit = False
        resp_lower = resp.lower()
        for kw in probe["expected_keywords"]:
            if kw.lower() in resp_lower:
                kw_hit = True
                break
                
        is_non_empty = len(resp.split()) >= probe.get("min_tokens", 1)
        
        # Repetition check (any word repeated > 3 times)
        words = resp.lower().split()
        word_counts = {}
        has_rep = False
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
            if word_counts[w] > 3 and len(w) > 3:
                has_rep = True
                break
        is_clean = not has_rep
        
        if kw_hit:
            cat_stats[cat]["hits"] += 1
            total_keyword_hits += 1
        if is_non_empty:
            cat_stats[cat]["non_empty"] += 1
            total_non_empty += 1
        if is_clean:
            cat_stats[cat]["no_rep"] += 1
            total_no_repetition += 1
            
        records.append({
            "id": probe["id"],
            "category": cat,
            "prompt": probe["prompt"],
            "expected_keywords": probe["expected_keywords"],
            "response": resp,
            "keyword_hit": kw_hit,
            "non_empty": is_non_empty,
            "clean_repetition": is_clean
        })
        
    summary = {
        "checkpoint_name": name,
        "checkpoint_path": str(path),
        "total_probes": len(benchmark),
        "keyword_hits": total_keyword_hits,
        "keyword_hit_rate": total_keyword_hits / len(benchmark),
        "non_empty_responses": total_non_empty,
        "clean_repetition_rate": total_no_repetition / len(benchmark),
        "category_breakdown": cat_stats
    }
    
    print(f"Results for {name}:")
    print(f"  Keyword Hits: {total_keyword_hits}/{len(benchmark)} ({total_keyword_hits/len(benchmark):.1%})")
    print(f"  Non-empty:    {total_non_empty}/{len(benchmark)} ({total_non_empty/len(benchmark):.1%})")
    print(f"  Clean:        {total_no_repetition}/{len(benchmark)} ({total_no_repetition/len(benchmark):.1%})")
    
    all_eval_results[name] = {
        "summary": summary,
        "records": records
    }

out_json = REPO_ROOT / "experiments/conversation/cr_exp_17/baseline_eval_results.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(all_eval_results, f, indent=2)

print(f"\nAll baseline evaluations saved to: {out_json}")
