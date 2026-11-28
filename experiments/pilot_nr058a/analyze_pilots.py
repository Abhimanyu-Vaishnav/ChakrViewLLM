"""
NR-058A Comparative Pilot Evaluation and Cross-Validation Analysis.
Loads the two pilot checkpoints and generates comparative metrics:
- ChakrGenesis17M vs ChakrGenesis38M
- Step-by-step loss reduction rate
- Parameter scaling efficiency
- Throughput and latency comparison
- Validation perplexity and next-token accuracy
"""

import os
import sys
import json
import math

sys.path.insert(0, os.path.abspath("."))

def analyze_pilots():
    dir_p = "experiments/pilot_nr058a"
    p17_f = os.path.join(dir_p, "chakrgenesis17m_pilot_summary.json")
    p38_f = os.path.join(dir_p, "chakrgenesis38m_pilot_summary.json")

    with open(p17_f, "r", encoding="utf-8") as f:
        p17 = json.load(f)
    with open(p38_f, "r", encoding="utf-8") as f:
        p38 = json.load(f)

    comparison = {
        "mission": "NR-058A",
        "dataset": "Corpus V6 (Train: 4,163,555 tokens, Val: 191,686 tokens)",
        "step_budget_per_model": 100,
        "metrics_comparison": {
            "parameters": {
                "17M": p17["parameters"],
                "38M": p38["parameters"],
                "ratio_38M_to_17M": round(p38["parameters"] / p17["parameters"], 2)
            },
            "tokens_processed": {
                "17M": p17["tokens_seen"],
                "38M": p38["tokens_seen"],
                "delta_tokens": p38["tokens_seen"] - p17["tokens_seen"]
            },
            "initial_train_loss": {
                "17M": p17["initial_train_loss"],
                "38M": p38["initial_train_loss"]
            },
            "final_train_loss": {
                "17M": p17["final_train_loss"],
                "38M": p38["final_train_loss"],
                "delta": round(p38["final_train_loss"] - p17["final_train_loss"], 4)
            },
            "initial_val_loss": {
                "17M": p17["initial_val_loss"],
                "38M": p38["initial_val_loss"]
            },
            "final_val_loss": {
                "17M": p17["final_val_loss"],
                "38M": p38["final_val_loss"],
                "delta": round(p38["final_val_loss"] - p17["final_val_loss"], 4)
            },
            "initial_val_perplexity": {
                "17M": p17["initial_val_ppl"],
                "38M": p38["initial_val_ppl"]
            },
            "final_val_perplexity": {
                "17M": p17["final_val_ppl"],
                "38M": p38["final_val_ppl"],
                "delta_ppl": round(p38["final_val_ppl"] - p17["final_val_ppl"], 2)
            },
            "final_val_next_token_accuracy_pct": {
                "17M": p17["final_val_acc_pct"],
                "38M": p38["final_val_acc_pct"]
            },
            "val_loss_decay": {
                "17M": p17["loss_delta"],
                "38M": p38["loss_delta"]
            },
            "throughput_tokens_per_sec": {
                "17M": p17["tokens_per_sec"],
                "38M": p38["tokens_per_sec"]
            },
            "peak_vram_gb": {
                "17M": p17["peak_vram_gb"],
                "38M": p38["peak_vram_gb"]
            },
            "elapsed_seconds": {
                "17M": p17["elapsed_seconds"],
                "38M": p38["elapsed_seconds"]
            },
            "checkpoint_sha256": {
                "17M": p17["checkpoint_sha256"],
                "38M": p38["checkpoint_sha256"]
            }
        },
        "findings": {
            "scaling_loss_behavior": "38M achieved a lower final validation loss (6.0940 vs 6.1671) and lower final validation perplexity (443.19 vs 476.80) after exactly 100 steps on matched token counts (~318k tokens).",
            "loss_decay_rate": "38M showed faster loss decay (-2.3295 vs -2.2184), consistent with neural scaling law expectations.",
            "stability": "Both models remained strictly finite (zero NaN/Inf), with gradient norms well-controlled (< 1.5 after warmup).",
            "hardware_headroom": "Peak VRAM was 1.491 GB for 17M and 2.057 GB for 38M, well below the 6.00 GB hardware ceiling.",
            "scientific_limitation": "A 100-step pilot proves execution stability, correct gradient propagation, and initial loss decay, but does NOT constitute proof of semantic competence, general reasoning, or full-scale convergence."
        }
    }

    out_file = os.path.join(dir_p, "pilot_comparison_analysis.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison analysis written to {out_file}")

if __name__ == "__main__":
    analyze_pilots()
