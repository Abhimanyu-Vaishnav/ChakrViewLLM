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
print(f"Running blind evaluation on {device.upper()}...")

tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")

with open("data/evaluation/benchmark_v2/benchmark_v2_100.json", "r", encoding="utf-8") as f:
    benchmark = json.load(f)

# Candidate configurations
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
blind_records = []

for cand_name, meta in candidates.items():
    print(f"\n---> Evaluating {cand_name}...")
    ckpt = torch.load(meta["path"], map_location=device, weights_only=False)
    cfg = ckpt.get("config", meta["config_cls"]())
    if isinstance(cfg, dict):
        cfg = meta["config_cls"](**cfg)
        
    model = meta["model_cls"](cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    
    cand_scores = []
    category_scores = {}
    language_scores = {}
    difficulty_scores = {}
    score_dist = {0: 0, 1: 0, 2: 0, 3: 0}
    prompt_records = []
    
    t0 = time.time()
    for idx, item in enumerate(benchmark):
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
        
        # Score response with robust rubric
        eval_dict = score_item(raw_response, item)
        score = eval_dict["score"]
        
        cand_scores.append(score)
        score_dist[score] += 1
        
        cat = item["category"]
        lang = item["language"]
        diff = item["difficulty"]
        
        category_scores.setdefault(cat, []).append(score)
        language_scores.setdefault(lang, []).append(score)
        difficulty_scores.setdefault(diff, []).append(score)
        
        rec = {
            "prompt_id": item["id"],
            "prompt": p_text,
            "category": cat,
            "language": lang,
            "difficulty": diff,
            "expected_answer": item["expected_answer"],
            "generated_response": raw_response,
            "score": score,
            "verdict": eval_dict["verdict"],
            "reason": eval_dict["reason"]
        }
        prompt_records.append(rec)
        
        blind_records.append({
            "model_code": cand_name,
            **rec
        })
        
    elapsed = time.time() - t0
    raw_mean = float(np.mean(cand_scores))
    norm_pct = float(raw_mean / 3.0 * 100)
    
    cat_summary = {k: {"mean": float(np.mean(v)), "norm_pct": float(np.mean(v)/3.0*100), "count": len(v)} for k, v in category_scores.items()}
    lang_summary = {k: {"mean": float(np.mean(v)), "norm_pct": float(np.mean(v)/3.0*100), "count": len(v)} for k, v in language_scores.items()}
    diff_summary = {k: {"mean": float(np.mean(v)), "norm_pct": float(np.mean(v)/3.0*100), "count": len(v)} for k, v in difficulty_scores.items()}
    
    evaluation_results[cand_name] = {
        "candidate": cand_name,
        "checkpoint_path": meta["path"],
        "total_prompts": len(benchmark),
        "raw_sum": int(sum(cand_scores)),
        "raw_mean": raw_mean,
        "normalized_percentage": norm_pct,
        "score_distribution": score_dist,
        "perfect_hits_count": score_dist[3],
        "category_summary": cat_summary,
        "language_summary": lang_summary,
        "difficulty_summary": diff_summary,
        "evaluation_time_seconds": elapsed,
        "prompt_records": prompt_records
    }
    
    print(f"[{cand_name}] Evaluated {len(benchmark)} items in {elapsed:.1f}s: Raw Mean={raw_mean:.3f}/3.0 ({norm_pct:.1f}%), ScoreDist={score_dist}")

# Bootstrap confidence intervals and paired permutation tests
print("\n---> Running Bootstrap Resampling (B=10,000)...")
np.random.seed(42)
B = 10000
boot_results = {}

for cand_name, data in evaluation_results.items():
    scores = np.array([r["score"] for r in data["prompt_records"]])
    boot_means = []
    n = len(scores)
    for _ in range(B):
        sample = np.random.choice(scores, size=n, replace=True)
        boot_means.append(np.mean(sample))
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    boot_results[cand_name] = {
        "mean": float(np.mean(scores)),
        "ci_95_raw": [float(ci_low), float(ci_high)],
        "ci_95_norm_pct": [float(ci_low/3.0*100), float(ci_high/3.0*100)]
    }

# Paired Permutation Tests against NR052 38M
p_tests = {}
base_scores = np.array([r["score"] for r in evaluation_results["NR052_38M_SFT"]["prompt_records"]])

for cand_name in ["RC3_Baseline", "NR051_17M_SFT", "NR051_38M_SFT", "NR052_17M_SFT"]:
    comp_scores = np.array([r["score"] for r in evaluation_results[cand_name]["prompt_records"]])
    obs_diff = np.mean(base_scores) - np.mean(comp_scores)
    
    diffs = base_scores - comp_scores
    perm_diffs = []
    for _ in range(B):
        signs = np.random.choice([-1, 1], size=len(diffs))
        perm_diffs.append(np.mean(diffs * signs))
    p_val = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))
    p_tests[f"NR052_38M_vs_{cand_name}"] = {
        "observed_diff": float(obs_diff),
        "p_value": p_val,
        "significant_alpha_05": p_val < 0.05
    }

final_output = {
    "benchmark_version": "2.0.0",
    "total_prompts": len(benchmark),
    "evaluation_hardware": device,
    "candidates": evaluation_results,
    "bootstrap_analysis": boot_results,
    "paired_permutation_tests": p_tests
}

with open("experiments/forensics_nr055/benchmark_v2_results.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=2, ensure_ascii=False)

with open("experiments/forensics_nr055/blind_evaluation_records.json", "w", encoding="utf-8") as f:
    json.dump(blind_records, f, indent=2, ensure_ascii=False)

print("\nBENCHMARK V2 EVALUATION COMPLETE. Results saved to experiments/forensics_nr055/benchmark_v2_results.json")
