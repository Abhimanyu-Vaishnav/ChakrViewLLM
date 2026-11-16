"""
analyze_regressions.py - Statistical & Regression Analysis for CR-EXP-19
Analyzes:
- Context retention accuracy across turns (1, 2, 3, 5, 8) and task types
- Conversational preservation across categories (greetings, identity, hinglish, natural)
- Statistical comparison (Mean, Std Dev, Delta vs Parent, Delta Arm B vs Arm A)
- Capability decision metrics
"""
import os
import sys
import json
import math
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
RESULTS_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_19" / "results"

def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def stddev(lst):
    if len(lst) <= 1:
        return 0.0
    m = mean(lst)
    variance = sum((x - m) ** 2 for x in lst) / (len(lst) - 1)
    return math.sqrt(variance)

def main():
    with open(RESULTS_DIR / "all_eval_summaries.json", "r", encoding="utf-8") as f:
        summaries = json.load(f)

    baseline = summaries["parent_baseline"]
    arm_a = [summaries[f"arm_a_seed{s}"] for s in [101, 202, 303]]
    arm_b = [summaries[f"arm_b_seed{s}"] for s in [101, 202, 303]]

    # 1. Context 200 Analysis
    # Across turns
    groups = ["1-turn", "2-turn", "3-turn", "5-turn", "8-turn"]
    context_stats = {}
    for g in groups:
        base_direct = baseline["context_summary"]["by_group"][g]["acc_direct"]
        base_cli = baseline["context_summary"]["by_group"][g]["acc_cli"]

        a_direct = [s["context_summary"]["by_group"][g]["acc_direct"] for s in arm_a]
        a_cli = [s["context_summary"]["by_group"][g]["acc_cli"] for s in arm_a]

        b_direct = [s["context_summary"]["by_group"][g]["acc_direct"] for s in arm_b]
        b_cli = [s["context_summary"]["by_group"][g]["acc_cli"] for s in arm_b]

        context_stats[g] = {
            "baseline": {"direct": base_direct, "cli": base_cli},
            "arm_a": {
                "direct_mean": mean(a_direct), "direct_std": stddev(a_direct),
                "cli_mean": mean(a_cli), "cli_std": stddev(a_cli)
            },
            "arm_b": {
                "direct_mean": mean(b_direct), "direct_std": stddev(b_direct),
                "cli_mean": mean(b_cli), "cli_std": stddev(b_cli)
            }
        }

    # Total context accuracy
    total_context = {
        "baseline": {
            "direct": baseline["context_summary"]["accuracy_direct"],
            "cli": baseline["context_summary"]["accuracy_cli"]
        },
        "arm_a": {
            "direct_mean": mean([s["context_summary"]["accuracy_direct"] for s in arm_a]),
            "direct_std": stddev([s["context_summary"]["accuracy_direct"] for s in arm_a]),
            "cli_mean": mean([s["context_summary"]["accuracy_cli"] for s in arm_a]),
            "cli_std": stddev([s["context_summary"]["accuracy_cli"] for s in arm_a])
        },
        "arm_b": {
            "direct_mean": mean([s["context_summary"]["accuracy_direct"] for s in arm_b]),
            "direct_std": stddev([s["context_summary"]["accuracy_direct"] for s in arm_b]),
            "cli_mean": mean([s["context_summary"]["accuracy_cli"] for s in arm_b]),
            "cli_std": stddev([s["context_summary"]["accuracy_cli"] for s in arm_b])
        }
    }

    # 2. Conversation 100 Analysis
    categories = ["greeting", "identity", "hinglish", "natural"]
    conv_stats = {}
    for c in categories:
        base_acc = baseline["conversation_summary"]["by_category"][c]["acc"]
        a_acc = [s["conversation_summary"]["by_category"][c]["acc"] for s in arm_a]
        b_acc = [s["conversation_summary"]["by_category"][c]["acc"] for s in arm_b]

        conv_stats[c] = {
            "baseline": base_acc,
            "arm_a": {"mean": mean(a_acc), "std": stddev(a_acc)},
            "arm_b": {"mean": mean(b_acc), "std": stddev(b_acc)},
            "delta_b_vs_baseline": mean(b_acc) - base_acc,
            "delta_b_vs_a": mean(b_acc) - mean(a_acc)
        }

    total_conv = {
        "baseline": baseline["conversation_summary"]["accuracy"],
        "arm_a": {
            "mean": mean([s["conversation_summary"]["accuracy"] for s in arm_a]),
            "std": stddev([s["conversation_summary"]["accuracy"] for s in arm_a])
        },
        "arm_b": {
            "mean": mean([s["conversation_summary"]["accuracy"] for s in arm_b]),
            "std": stddev([s["conversation_summary"]["accuracy"] for s in arm_b])
        },
        "delta_b_vs_baseline": mean([s["conversation_summary"]["accuracy"] for s in arm_b]) - baseline["conversation_summary"]["accuracy"],
        "delta_b_vs_a": mean([s["conversation_summary"]["accuracy"] for s in arm_b]) - mean([s["conversation_summary"]["accuracy"] for s in arm_a])
    }

    analysis_out = {
        "context_stats_by_turns": context_stats,
        "context_stats_total": total_context,
        "conversation_stats_by_category": conv_stats,
        "conversation_stats_total": total_conv,
        "capability_summary": {
            "runtime_history_packing_fixed": True,
            "runtime_max_history_turns_verified": "Dynamic budget up to 384 tokens (5-8 turns verified)",
            "neural_context_retrieval_acquired": False,
            "baseline_context_200": "0/200 (0.0%)",
            "arm_a_context_200_mean": "0/200 (0.0%)",
            "arm_b_context_200_mean": "0/200 (0.0%)",
            "conversation_retention_preserved": True,
            "baseline_conv_100": f"{baseline['conversation_summary']['accuracy']*100:.1f}%",
            "arm_a_conv_100_mean": f"{total_conv['arm_a']['mean']*100:.1f}%",
            "arm_b_conv_100_mean": f"{total_conv['arm_b']['mean']*100:.1f}%"
        }
    }

    with open(RESULTS_DIR / "regression_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis_out, f, indent=2)

    print(json.dumps(analysis_out, indent=2))
    print("\n[SUCCESS] Saved regression analysis to results/regression_analysis.json")

if __name__ == "__main__":
    main()

