"""
experiments/forensics_nr054/bootstrap_statistical_analysis.py

Phase 5: Rigorous Statistical Analysis & Bootstrap Resampling
Calculates:
1. Reconciled raw benchmark means, normalized means (out of 3), and exact score distributions
2. 95% Bootstrap Confidence Intervals (B = 10,000 iterations) at prompt-level
3. Paired permutation test p-values between models (evaluating if 38M > 17M is statistically significant)
4. Separate Sampling (T=0.3) vs Greedy (T=0.0) analysis
5. Strict Exact Factual Correctness (Score 3 with verified semantic accuracy)
6. Outputs experiments/forensics_nr054/statistical_analysis_results.json
"""

import sys
import json
import random
import math
from pathlib import Path

random.seed(42)

def bootstrap_ci(scores, B=10000, alpha=0.05):
    n = len(scores)
    means = []
    for _ in range(B):
        sample = [random.choice(scores) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low_idx = int(B * (alpha / 2))
    high_idx = int(B * (1 - alpha / 2))
    return means[low_idx], means[high_idx]

def paired_permutation_test(scores_a, scores_b, B=10000):
    n = len(scores_a)
    diff_observed = (sum(scores_a) - sum(scores_b)) / n
    count = 0
    for _ in range(B):
        perm_diff = 0
        for i in range(n):
            if random.random() < 0.5:
                perm_diff += (scores_a[i] - scores_b[i])
            else:
                perm_diff += (scores_b[i] - scores_a[i])
        perm_mean_diff = perm_diff / n
        if abs(perm_mean_diff) >= abs(diff_observed):
            count += 1
    return diff_observed, count / B

def main():
    print("=== PHASE 5: BOOTSTRAP RESAMPLING & STATISTICAL ANALYSIS ===", flush=True)

    data = json.load(open("experiments/forensics_nr053/fresh_eval_results_50.json", encoding="utf-8"))
    details = data["details"]

    candidates = ["RC3_Historical_Baseline", "NR051_17M_SFT", "NR051_38M_SFT", "NR052_17M_SFT", "NR052_38M_SFT"]

    results = {
        "sampling_T03": {},
        "greedy_T00": {},
        "paired_tests": {}
    }

    # 1. Sampling Analysis
    print("\n--- SAMPLING (T=0.3, top_p=0.9, rep_penalty=1.15) ---", flush=True)
    sampling_scores = {}
    for cand in candidates:
        scores = [x["sampling_T03"]["score"] for x in details[cand]]
        sampling_scores[cand] = scores
        n = len(scores)
        raw_mean = sum(scores) / n
        norm_mean = sum(scores) / (n * 3)
        dist = [scores.count(0), scores.count(1), scores.count(2), scores.count(3)]
        ci_low, ci_high = bootstrap_ci(scores)
        
        # Exact verified hits (Tokyo, Pacific)
        verified_hits = sum(1 for x in details[cand] if x["id"] in ["FQA-01", "FQA-03"] and x["sampling_T03"]["score"] == 3)

        results["sampling_T03"][cand] = {
            "n": n,
            "score_distribution": dist,
            "total_raw_score": sum(scores),
            "raw_mean_score": round(raw_mean, 4),
            "normalized_mean_score": round(norm_mean, 4),
            "percentage_capability": round(norm_mean * 100, 2),
            "ci_95_raw_mean": [round(ci_low, 4), round(ci_high, 4)],
            "ci_95_percentage": [round(ci_low / 3 * 100, 2), round(ci_high / 3 * 100, 2)],
            "pct_scoring_ge_1": round(sum(dist[1:]) / n * 100, 1),
            "pct_scoring_ge_2": round(sum(dist[2:]) / n * 100, 1),
            "pct_scoring_eq_3": round(dist[3] / n * 100, 1),
            "exact_verified_factual_hits": verified_hits
        }
        print(f"{cand:25}: RawMean={raw_mean:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}]) | Dist={dist} | Pct={norm_mean*100:.1f}%")

    # 2. Greedy Analysis
    print("\n--- GREEDY (T=0.0) ---", flush=True)
    greedy_scores = {}
    for cand in candidates:
        scores = [x["greedy"]["score"] for x in details[cand]]
        greedy_scores[cand] = scores
        n = len(scores)
        raw_mean = sum(scores) / n
        norm_mean = sum(scores) / (n * 3)
        dist = [scores.count(0), scores.count(1), scores.count(2), scores.count(3)]
        ci_low, ci_high = bootstrap_ci(scores)

        results["greedy_T00"][cand] = {
            "n": n,
            "score_distribution": dist,
            "total_raw_score": sum(scores),
            "raw_mean_score": round(raw_mean, 4),
            "normalized_mean_score": round(norm_mean, 4),
            "percentage_capability": round(norm_mean * 100, 2),
            "ci_95_raw_mean": [round(ci_low, 4), round(ci_high, 4)],
            "pct_scoring_eq_3": round(dist[3] / n * 100, 1)
        }
        print(f"{cand:25}: RawMean={raw_mean:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}]) | Dist={dist} | Pct={norm_mean*100:.1f}%")

    # 3. Paired Permutation Tests
    print("\n--- PAIRED PERMUTATION TESTS (Sampling T=0.3) ---", flush=True)
    pairs = [
        ("NR052_38M_SFT", "NR052_17M_SFT"),
        ("NR052_38M_SFT", "NR051_38M_SFT"),
        ("NR052_38M_SFT", "RC3_Historical_Baseline"),
        ("NR051_38M_SFT", "NR051_17M_SFT")
    ]

    for a, b in pairs:
        diff, p_val = paired_permutation_test(sampling_scores[a], sampling_scores[b])
        is_sig = (p_val < 0.05)
        results["paired_tests"][f"{a}_vs_{b}"] = {
            "mean_difference": round(diff, 4),
            "p_value": round(p_val, 4),
            "statistically_significant_at_05": is_sig
        }
        sig_str = "STATISTICALLY SIGNIFICANT" if is_sig else "NOT STATISTICALLY SIGNIFICANT"
        print(f"{a} vs {b}: Mean Diff = {diff:+.4f} | p-value = {p_val:.4f} ({sig_str})")

    out_file = Path("experiments/forensics_nr054/statistical_analysis_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nStatistical analysis complete! Output written to {out_file}", flush=True)

if __name__ == "__main__":
    main()
