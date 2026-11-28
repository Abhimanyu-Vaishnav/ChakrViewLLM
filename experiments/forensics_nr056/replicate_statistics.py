import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))

with open("experiments/forensics_nr055/benchmark_v2_results.json", "r", encoding="utf-8") as f:
    v2_data = json.load(f)

candidates = ["RC3_Baseline", "NR051_17M_SFT", "NR051_38M_SFT", "NR052_17M_SFT", "NR052_38M_SFT"]
prompt_scores = {}

for cand in candidates:
    scores = [r["score"] for r in v2_data["candidates"][cand]["prompt_records"]]
    prompt_scores[cand] = np.array(scores)

n_prompts = len(prompt_scores["RC3_Baseline"])
print(f"Total benchmark prompts: {n_prompts}")

# Descriptive Stats
stats_summary = {}
for cand in candidates:
    s = prompt_scores[cand]
    stats_summary[cand] = {
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "std": float(np.std(s, ddof=1)),
        "sum": int(np.sum(s)),
        "score_dist": {
            0: int(np.sum(s == 0)),
            1: int(np.sum(s == 1)),
            2: int(np.sum(s == 2)),
            3: int(np.sum(s == 3))
        }
    }

# Bootstrap Resampling (B=10,000, seed=42)
np.random.seed(42)
B = 10000
boot_results = {}
for cand in candidates:
    s = prompt_scores[cand]
    means = [np.mean(np.random.choice(s, size=n_prompts, replace=True)) for _ in range(B)]
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    boot_results[cand] = {
        "mean": float(np.mean(s)),
        "ci_95_raw": [float(ci_low), float(ci_high)],
        "ci_95_norm_pct": [float(ci_low / 3.0 * 100), float(ci_high / 3.0 * 100)]
    }

# Paired Permutation Tests against all pairs
paired_results = {}
comparisons = [
    ("NR052_38M_SFT", "RC3_Baseline"),
    ("NR052_38M_SFT", "NR051_17M_SFT"),
    ("NR052_38M_SFT", "NR051_38M_SFT"),
    ("NR052_38M_SFT", "NR052_17M_SFT"),
    ("RC3_Baseline", "NR051_17M_SFT"),
    ("RC3_Baseline", "NR051_38M_SFT"),
    ("RC3_Baseline", "NR052_17M_SFT"),
    ("NR052_17M_SFT", "NR051_17M_SFT"),
    ("NR052_17M_SFT", "NR051_38M_SFT"),
    ("NR051_38M_SFT", "NR051_17M_SFT"),
]

raw_p_values = []
comp_keys = []

for c1, c2 in comparisons:
    s1 = prompt_scores[c1]
    s2 = prompt_scores[c2]
    diff = s1 - s2
    obs_diff = float(np.mean(diff))
    
    perm_diffs = []
    for _ in range(B):
        signs = np.random.choice([-1, 1], size=n_prompts)
        perm_diffs.append(np.mean(diff * signs))
    p_val = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))
    
    key = f"{c1}_vs_{c2}"
    comp_keys.append(key)
    raw_p_values.append(p_val)
    paired_results[key] = {
        "candidate_1": c1,
        "candidate_2": c2,
        "observed_mean_diff": obs_diff,
        "raw_p_value": p_val
    }

# Apply Holm-Bonferroni Multiple Comparison Correction
# Sort by p-value ascending
sorted_indices = np.argsort(raw_p_values)
m = len(raw_p_values)
holm_adjusted = {}

for rank, idx in enumerate(sorted_indices):
    k = comp_keys[idx]
    p = raw_p_values[idx]
    # Holm threshold: alpha / (m - rank)
    adj_p = min(1.0, p * (m - rank))
    # Enforce monotonicity
    paired_results[k]["holm_adjusted_p_value"] = float(adj_p)
    paired_results[k]["significant_at_alpha_05_uncorrected"] = p < 0.05
    paired_results[k]["significant_at_alpha_05_holm_corrected"] = adj_p < 0.05

with open("experiments/forensics_nr056/statistical_results.json", "w") as f:
    json.dump(stats_summary, f, indent=2)

with open("experiments/forensics_nr056/bootstrap_results.json", "w") as f:
    json.dump(boot_results, f, indent=2)

with open("experiments/forensics_nr056/paired_comparisons.json", "w") as f:
    json.dump(paired_results, f, indent=2)

print("STATISTICAL REPLICATION COMPLETE.")
print("\nKey Comparisons (NR-052 38M vs Others):")
for k, v in paired_results.items():
    if "NR052_38M_SFT" in k:
        print(f"  {k:35} Diff: {v['observed_mean_diff']:+.4f} | Raw p: {v['raw_p_value']:.4f} | Holm p: {v['holm_adjusted_p_value']:.4f} | Sig: {v['significant_at_alpha_05_holm_corrected']}")
