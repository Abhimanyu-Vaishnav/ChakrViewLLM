import os
import sys
import json
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.brain.genesis.chakr_genesis_38m import ChakrGenesis38M, Genesis38MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.forensics_nr055.robust_evaluator import score_item

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running Benchmark V3 independent capability evaluation on {device.upper()}...")

tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")

with open("data/evaluation/benchmark_v3/benchmark_v3_300.json", "r", encoding="utf-8") as f:
    benchmark_v3 = json.load(f)

candidates = {
    "RC3_Baseline": {
        "path": "experiments/fast_track_047_049/chakr_1_0_rc3_best.pt",
        "model_cls": ChakrGenesis17M,
        "config_cls": Genesis17MConfig
    },
    "NR051_17M_SFT": {
        "path": "experiments/pretraining_nr051/chakr_17m_sft_candidate.pt",
        "model_cls": ChakrGenesis17M,
        "config_cls": Genesis17MConfig
    },
    "NR051_38M_SFT": {
        "path": "experiments/pretraining_nr051/chakr_38m_sft_candidate.pt",
        "model_cls": ChakrGenesis38M,
        "config_cls": Genesis38MConfig
    },
    "NR052_17M_SFT": {
        "path": "experiments/pretraining_nr052/chakr_17m_nr052_sft.pt",
        "model_cls": ChakrGenesis17M,
        "config_cls": Genesis17MConfig
    },
    "NR052_38M_SFT": {
        "path": "experiments/pretraining_nr052/chakr_38m_nr052_sft.pt",
        "model_cls": ChakrGenesis38M,
        "config_cls": Genesis38MConfig
    }
}

evaluation_results = {}

for cand_name, meta in candidates.items():
    print(f"\n---> Evaluating {cand_name} on Benchmark V3 (300 items)...")
    ckpt = torch.load(meta["path"], map_location=device, weights_only=False)
    cfg = ckpt.get("config", meta["config_cls"]())
    if isinstance(cfg, dict):
        cfg = meta["config_cls"](**cfg)
        
    model = meta["model_cls"](cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    
    cand_scores = []
    category_scores = {}
    test_type_scores = {}
    score_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    raw_generations = []
    scored_records = []
    
    t0 = time.time()
    for idx, item in enumerate(benchmark_v3):
        p_text = item["prompt"]
        formatted = f"<user>\n{p_text}\n<assistant>\n"
        tokens = [0] + tokenizer.encode(formatted)
        input_ids = torch.tensor([tokens], device=device, dtype=torch.long)
        
        # Greedy decoding up to 32 tokens
        generated = input_ids.clone()
        with torch.no_grad():
            for _ in range(32):
                out = model(generated)
                logits = out[0] if isinstance(out, tuple) else out
                nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, nxt], dim=1)
                if nxt.item() in (1, 2):  # EOS
                    break
                    
        resp_tokens = generated[0, len(tokens):].tolist()
        raw_response = tokenizer.decode(resp_tokens, errors="replace").strip()
        
        # Score response
        eval_dict = score_item(raw_response, item)
        score = eval_dict["score"]
        
        cand_scores.append(score)
        score_dist[score] += 1
        
        cat = item["category"]
        tt = item["test_type"]
        category_scores.setdefault(cat, []).append(score)
        test_type_scores.setdefault(tt, []).append(score)
        
        raw_generations.append({
            "prompt_id": item["id"],
            "prompt": p_text,
            "expected_answer": item["expected_answer"],
            "generated_response": raw_response
        })
        
        scored_records.append({
            "prompt_id": item["id"],
            "category": cat,
            "test_type": tt,
            "expected_answer": item["expected_answer"],
            "generated_response": raw_response,
            "score": score,
            "verdict": eval_dict["verdict"]
        })
        
    elapsed = time.time() - t0
    raw_mean = float(np.mean(cand_scores))
    norm_pct = float(raw_mean / 3.0 * 100)
    
    # Save raw generations file
    raw_gen_path = f"experiments/forensics_nr056/raw_generations/{cand_name}_raw_generations.json"
    with open(raw_gen_path, "w", encoding="utf-8") as f:
        json.dump(raw_generations, f, indent=2, ensure_ascii=False)
        
    evaluation_results[cand_name] = {
        "candidate": cand_name,
        "total_prompts": len(benchmark_v3),
        "raw_sum": int(sum(cand_scores)),
        "raw_mean": raw_mean,
        "normalized_pct": norm_pct,
        "score_distribution": score_dist,
        "category_scores": {k: {"mean": float(np.mean(v)), "norm_pct": float(np.mean(v)/3*100), "pts": int(sum(v))} for k, v in category_scores.items()},
        "test_type_scores": {k: {"mean": float(np.mean(v)), "norm_pct": float(np.mean(v)/3*100), "pts": int(sum(v))} for k, v in test_type_scores.items()},
        "elapsed_seconds": elapsed,
        "scored_records": scored_records
    }
    print(f"[{cand_name}] Evaluated 300 items in {elapsed:.1f}s | Mean={raw_mean:.4f}/3.0 ({norm_pct:.2f}%) | ScoreDist={score_dist}")

# Statistical Summary on Benchmark V3
np.random.seed(42)
B = 10000
v3_boot = {}
for cand_name, data in evaluation_results.items():
    s = np.array([r["score"] for r in data["scored_records"]])
    means = [np.mean(np.random.choice(s, size=len(s), replace=True)) for _ in range(B)]
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    v3_boot[cand_name] = {
        "mean": float(np.mean(s)),
        "ci_95_raw": [float(ci_low), float(ci_high)],
        "ci_95_norm_pct": [float(ci_low/3*100), float(ci_high/3*100)]
    }

# Save final benchmark v3 results
final_out = {
    "benchmark_version": "3.0.0",
    "total_prompts": len(benchmark_v3),
    "candidates": evaluation_results,
    "bootstrap_analysis": v3_boot
}

with open("experiments/forensics_nr056/benchmark_v3_results.json", "w", encoding="utf-8") as f:
    json.dump(final_out, f, indent=2, ensure_ascii=False)

print("\nBENCHMARK V3 EVALUATION COMPLETE. Results saved.")
