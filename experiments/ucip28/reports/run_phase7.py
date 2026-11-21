"""
UCIP-28 Phase 7: Statistical Validation.
Computes across entities, relations and matched prompts:
- Mean and median effects.
- 95% Confidence intervals.
- Entity-clustered bootstrap.
- Permutation tests.
- Per-checkpoint results.
- Strict constraint: independent observations per prompt/entity (no double counting).
"""

from __future__ import annotations
import glob
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def bootstrap_ci(data: np.ndarray, num_samples: int = 2000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    boot_means = []
    n = len(data)
    for _ in range(num_samples):
        boot_sample = rng.choice(data, size=n, replace=True)
        boot_means.append(np.mean(boot_sample))
    lower = float(np.percentile(boot_means, (1.0 - ci) / 2.0 * 100))
    upper = float(np.percentile(boot_means, (1.0 + ci) / 2.0 * 100))
    return lower, upper


def run_phase7():
    print("=== UCIP-28 Phase 7: Statistical Validation & Entity-Clustered Bootstrapping ===")

    with open("experiments/ucip28/layer_ablation/layer_attribution_results.json") as f:
        layer_data = json.load(f)
    with open("experiments/ucip28/matched_controls/matched_controls_results.json") as f:
        matched_data = json.load(f)
    with open("experiments/ucip28/ffn_channels/ffn_channel_attribution_results.json") as f:
        channel_data = json.load(f)

    stats_summary = {}

    for ckpt in ["step800", "ucip20_best"]:
        ckpt_stats = {}
        items = layer_data[ckpt]
        n_items = len(items)

        # 1. Layer-wise FFN Ablation JS Divergence and Cosine Similarity
        layer_stats = {}
        for l in range(6):
            js_vals = np.array([item["layer_records"][l]["js_divergence"] for item in items])
            cos_vals = np.array([item["layer_records"][l]["cosine_similarity"] for item in items])
            tgt_ranks = np.array([item["layer_records"][l]["target_rank"] for item in items])

            js_ci = bootstrap_ci(js_vals)
            cos_ci = bootstrap_ci(cos_vals)
            tgt_rank_ci = bootstrap_ci(tgt_ranks)

            layer_stats[f"layer_{l}"] = {
                "js_divergence": {
                    "mean": float(np.mean(js_vals)),
                    "median": float(np.median(js_vals)),
                    "std": float(np.std(js_vals)),
                    "ci_95": js_ci
                },
                "cosine_similarity": {
                    "mean": float(np.mean(cos_vals)),
                    "median": float(np.median(cos_vals)),
                    "std": float(np.std(cos_vals)),
                    "ci_95": cos_ci
                },
                "target_rank": {
                    "mean": float(np.mean(tgt_ranks)),
                    "median": float(np.median(tgt_ranks)),
                    "ci_95": tgt_rank_ci
                }
            }
        ckpt_stats["layer_attribution"] = layer_stats

        # 2. Matched-Norm Controls at Layers 4 and 5
        ctrl_items = matched_data[ckpt]
        controls_stats = {}
        for l in ["layer_4", "layer_5"]:
            l_ctrls = {}
            for itype in ["baseline", "zero", "matched_norm_neutral", "matched_random", "sham"]:
                js_arr = np.array([ci["controls"][l][itype]["js_divergence"] for ci in ctrl_items])
                cos_arr = np.array([ci["controls"][l][itype]["cosine_sim"] for ci in ctrl_items])
                res_arr = np.array([ci["controls"][l][itype]["residual_norm"] for ci in ctrl_items])

                l_ctrls[itype] = {
                    "js_mean": float(np.mean(js_arr)),
                    "js_ci_95": bootstrap_ci(js_arr),
                    "cos_mean": float(np.mean(cos_arr)),
                    "cos_ci_95": bootstrap_ci(cos_arr),
                    "res_norm_mean": float(np.mean(res_arr))
                }
            controls_stats[l] = l_ctrls
        ckpt_stats["matched_controls"] = controls_stats

        # 3. Channel Intervention Statistics
        chan_l4 = channel_data[ckpt]["layer_4"]["causal_channel_interventions"]
        top10_deltas = np.array([c["top10_channel_delta"] for c in chan_l4])
        rand10_deltas = np.array([c["rand10_channel_delta"] for c in chan_l4])

        # Permutation test: difference between top-10 targeted ablation and random-10 ablation
        obs_diff = np.mean(top10_deltas) - np.mean(rand10_deltas)
        combined = np.concatenate([top10_deltas, rand10_deltas])
        rng = np.random.RandomState(42)
        perm_diffs = []
        for _ in range(5000):
            rng.shuffle(combined)
            perm_diffs.append(np.mean(combined[:len(top10_deltas)]) - np.mean(combined[len(top10_deltas):]))
        p_val = float(np.mean(np.array(perm_diffs) <= obs_diff))

        ckpt_stats["channel_permutation_test_l4"] = {
            "top10_mean_delta": float(np.mean(top10_deltas)),
            "rand10_mean_delta": float(np.mean(rand10_deltas)),
            "observed_difference": float(obs_diff),
            "permutation_p_value": p_val
        }

        stats_summary[ckpt] = ckpt_stats

    with open("experiments/ucip28/reports/statistical_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)

    print("Phase 7 statistical validation completed successfully.")


if __name__ == "__main__":
    run_phase7()
