import json
import hashlib
import time
from pathlib import Path
import sys

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

import torch
from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.runtime.sampling import SamplingConfig

BENCHMARK_PATH = REPO_ROOT / "experiments/cognition/cr_exp_17r/benchmark/CR17R_CONVERSATION_100.json"
PARENT_CKPT = REPO_ROOT / "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt"
RELEASE_CKPT = REPO_ROOT / "releases/v0.9.1_conversational_preview/chakrview_v091_model.pt"

RESULTS_DIR = REPO_ROOT / "experiments/cognition/cr_exp_17r/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_checkpoint(ckpt_path, model_name, decoding_cfg):
    print(f"\n--- Evaluating {model_name} on CR17R 100-probe benchmark ---")
    runner = DirectNeuralRunner(ckpt_path)
    tok = runner.tokenizer
    
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        bench_data = json.load(f)
        
    records = []
    category_stats = {}
    total_refusal_sinks = 0
    total_repetition_loops = 0
    total_keyword_hits = 0
    
    for item in bench_data:
        pid = item["probe_id"]
        cat = item["category"]
        prompt = item["formatted_prompt"]
        expected_req = item["required_info"]
        
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "hits": 0, "refusals": 0}
        category_stats[cat]["total"] += 1
        
        # Encode
        toks = tok.encode(prompt, add_bos=True, add_eos=False)
        curr = list(toks)
        gen = []
        terminated_normally = False
        
        start_t = time.time()
        with torch.no_grad():
            for step in range(48):
                t_in = torch.tensor([curr], dtype=torch.long)
                logits = runner.model(t_in)[0, -1, :]
                tok_id = runner.sampler.sample(logits, generated_tokens=gen, config=decoding_cfg, step=step)
                gen.append(tok_id)
                curr.append(tok_id)
                if tok_id == 1:
                    terminated_normally = True
                    break
        elapsed = time.time() - start_t
        
        resp_text = tok.decode(gen, errors="replace").strip()
        
        # Check refusal sink
        refusal_phrases = ["not provided in the context", "provided context does not", "not in the context", "no context provided"]
        is_refusal = any(rp in resp_text.lower() for rp in refusal_phrases)
        if is_refusal:
            total_refusal_sinks += 1
            category_stats[cat]["refusals"] += 1
            
        # Check repetition loop (e.g. repeated 3-gram or repeated word)
        words = resp_text.split()
        rep_loop = False
        if len(words) >= 6:
            for w_idx in range(len(words) - 3):
                trigram = " ".join(words[w_idx:w_idx+3])
                if resp_text.count(trigram) >= 3:
                    rep_loop = True
                    break
        if rep_loop:
            total_repetition_loops += 1
            
        # Check required info keyword match
        hit = any(req.lower() in resp_text.lower() for req in expected_req)
        if hit:
            total_keyword_hits += 1
            category_stats[cat]["hits"] += 1
            
        records.append({
            "probe_id": pid,
            "category": cat,
            "prompt": prompt,
            "response": resp_text,
            "token_count": len(gen),
            "terminated_normally": terminated_normally,
            "elapsed_sec": round(elapsed, 4),
            "is_refusal_sink": is_refusal,
            "is_repetition_loop": rep_loop,
            "keyword_hit": hit,
            "required_info": expected_req
        })
        
    summary = {
        "model_name": model_name,
        "checkpoint_path": str(ckpt_path),
        "total_probes": len(bench_data),
        "keyword_hits": total_keyword_hits,
        "keyword_hit_rate": round(total_keyword_hits / len(bench_data), 4),
        "refusal_sink_count": total_refusal_sinks,
        "refusal_sink_rate": round(total_refusal_sinks / len(bench_data), 4),
        "repetition_loop_count": total_repetition_loops,
        "repetition_loop_rate": round(total_repetition_loops / len(bench_data), 4),
        "category_breakdown": category_stats,
        "records": records
    }
    return summary

def main():
    # Deterministic calibrated decoding matching release defaults
    cfg_release = SamplingConfig(
        temperature=0.0,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3
    )
    
    # 1. Evaluate Parent
    parent_res = evaluate_checkpoint(PARENT_CKPT, "Conversational Parent (CBR-SFT-04)", cfg_release)
    with open(RESULTS_DIR / "parent_results.json", "w", encoding="utf-8") as f:
        json.dump(parent_res, f, indent=2)
    print(f"Parent Results saved: Hits={parent_res['keyword_hits']}/100, Refusals={parent_res['refusal_sink_count']}/100")
    
    # 2. Evaluate Release Candidate
    cand_res = evaluate_checkpoint(RELEASE_CKPT, "ChakrView v0.9.1 (Release Preview)", cfg_release)
    with open(RESULTS_DIR / "candidate_results.json", "w", encoding="utf-8") as f:
        json.dump(cand_res, f, indent=2)
    print(f"Candidate Results saved: Hits={cand_res['keyword_hits']}/100, Refusals={cand_res['refusal_sink_count']}/100")
    
    # 3. Save combined raw generations
    with open(RESULTS_DIR / "raw_generations.jsonl", "w", encoding="utf-8") as f:
        for p, c in zip(parent_res["records"], cand_res["records"]):
            f.write(json.dumps({
                "probe_id": p["probe_id"],
                "category": p["category"],
                "prompt": p["prompt"],
                "parent_response": p["response"],
                "candidate_response": c["response"],
                "parent_refusal": p["is_refusal_sink"],
                "candidate_refusal": c["is_refusal_sink"],
                "parent_hit": p["keyword_hit"],
                "candidate_hit": c["keyword_hit"],
                "required_info": p["required_info"]
            }, ensure_ascii=False) + "\n")
            
    print("Combined raw_generations.jsonl saved successfully.")

if __name__ == "__main__":
    main()
