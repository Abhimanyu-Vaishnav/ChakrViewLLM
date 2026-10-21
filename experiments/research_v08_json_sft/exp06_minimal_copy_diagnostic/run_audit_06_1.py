"""Re-computation & Audit Script for Experiment 06.1.

Audits all 180 records across Train (100), Validation (30), and Test (50).
Computes for every example:
- Prompt token IDs & response token IDs.
- Causal logit index for first response token.
- Predicted token ID vs target token ID.
- Exact target-token rank under standard tie-breaking (min-rank / 1-based index in descending sort).
- Target-token probability via numerically stable softmax.
- Full-response exact match under greedy autoregressive generation.
- Exact breakdown:
    * First-token teacher-forced accuracy (target token 0 strictly).
    * Value-subwords teacher-forced accuracy (all tokens of the target value).
    * EOS teacher-forced accuracy.
    * Combined response teacher-forced accuracy (including EOS).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v08_json_sft.exp06_minimal_copy_diagnostic.evaluate_exp06 import generate_greedy, evaluate_minimal_copy

EXP06_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP06_DIR / "data"
CKPT_PATH = EXP06_DIR / "checkpoints" / "exp06_best_checkpoint.pt"

OUTPUT_JSON = EXP06_DIR / "AUDIT_06_1_RESULTS.json"
OUTPUT_CSV = EXP06_DIR / "AUDIT_06_1_PER_EXAMPLE.csv"


def run_audit():
    print("=" * 70)
    print("CHAKRVIEW v0.8: EXPERIMENT 06.1 EVALUATION INTEGRITY AUDIT")
    print("=" * 70)

    device = torch.device("cpu")
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")

    print(f"Loading checkpoint from: {CKPT_PATH.name}")
    st = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    model.load_state_dict(st["model_state_dict"])
    model.eval()

    splits = ["train", "val", "test"]
    all_audit_records = []
    aggregated_splits = {}

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    for split in splits:
        split_file = DATA_DIR / f"{split}.jsonl"
        with open(split_file, "r", encoding="utf-8") as f:
            raw_recs = [json.loads(line) for line in f if line.strip()]

        split_results = []
        metrics = {
            "total_examples": len(raw_recs),
            "free_gen_exact_matches": 0,
            "first_token_tf_matches": 0,
            "value_tokens_tf_matches": 0,
            "value_tokens_total": 0,
            "eos_tf_matches": 0,
            "combined_response_tf_matches": 0,
            "combined_response_total": 0,
            "first_token_ranks": [],
            "first_token_probs": [],
            "stage_a": {
                "examples": 0,
                "free_gen_exact": 0,
                "first_tok_tf_matches": 0,
                "eos_tf_matches": 0,
                "combined_tf_matches": 0,
                "combined_tf_total": 0,
                "ranks": [],
                "probs": [],
            },
            "stage_b": {
                "examples": 0,
                "free_gen_exact": 0,
                "first_tok_tf_matches": 0,
                "val_tokens_tf_matches": 0,
                "val_tokens_total": 0,
                "eos_tf_matches": 0,
                "combined_tf_matches": 0,
                "combined_tf_total": 0,
                "ranks": [],
                "probs": [],
            }
        }

        for rec in raw_recs:
            rid = rec["id"]
            stage = rec["stage"]
            prompt = rec["prompt"]
            expected_val = rec["expected_value"]
            target_tokens = rec["target_tokens"]

            p_toks = tok.encode(prompt)
            r_toks = tok.encode(expected_val)
            assert r_toks == target_tokens, f"Token mismatch in {rid}"

            # 1. Greedy Generation
            inp = [0] + p_toks
            gen_out = generate_greedy(model, tok, inp, max_new_tokens=16)
            exact_match = (gen_out["text"].strip() == expected_val.strip())
            if exact_match:
                metrics["free_gen_exact_matches"] += 1
                if stage == "Stage_A":
                    metrics["stage_a"]["free_gen_exact"] += 1
                else:
                    metrics["stage_b"]["free_gen_exact"] += 1

            # 2. Teacher-Forced Pass
            tf_input = [0] + p_toks + r_toks
            tf_target = ([-100] * len(p_toks)) + r_toks + [1]

            with torch.no_grad():
                logits = model(torch.tensor([tf_input], dtype=torch.long))[0]

            resp_start_idx = len(p_toks)

            # Analyze first response token
            first_target_tok = r_toks[0]
            first_logits = logits[resp_start_idx, :]
            first_pred_tok = int(torch.argmax(first_logits).item())
            first_tok_match = (first_pred_tok == first_target_tok)

            # Numerically stable rank and softmax prob
            sorted_indices = torch.argsort(first_logits, descending=True)
            rank = int((sorted_indices == first_target_tok).nonzero(as_tuple=True)[0].item()) + 1
            prob = float(F.softmax(first_logits, dim=-1)[first_target_tok].item())

            metrics["first_token_ranks"].append(rank)
            metrics["first_token_probs"].append(prob)

            if first_tok_match:
                metrics["first_token_tf_matches"] += 1

            # Value tokens predictions
            val_tok_matches = 0
            for j, tgt_t in enumerate(r_toks):
                pr_t = int(torch.argmax(logits[resp_start_idx + j, :]).item())
                if pr_t == tgt_t:
                    val_tok_matches += 1
                    metrics["value_tokens_tf_matches"] += 1
                metrics["value_tokens_total"] += 1

            # EOS token prediction
            eos_pos = resp_start_idx + len(r_toks)
            eos_pred = int(torch.argmax(logits[eos_pos, :]).item())
            eos_match = (eos_pred == 1)
            if eos_match:
                metrics["eos_tf_matches"] += 1

            combined_matches = val_tok_matches + (1 if eos_match else 0)
            combined_total = len(r_toks) + 1
            metrics["combined_response_tf_matches"] += combined_matches
            metrics["combined_response_total"] += combined_total

            # Stage-specific aggregation
            st_key = "stage_a" if stage == "Stage_A" else "stage_b"
            metrics[st_key]["examples"] += 1
            if first_tok_match:
                metrics[st_key]["first_tok_tf_matches"] += 1
            if eos_match:
                metrics[st_key]["eos_tf_matches"] += 1
            metrics[st_key]["combined_tf_matches"] += combined_matches
            metrics[st_key]["combined_tf_total"] += combined_total
            metrics[st_key]["ranks"].append(rank)
            metrics[st_key]["probs"].append(prob)

            if stage == "Stage_B":
                metrics["stage_b"]["val_tokens_tf_matches"] += val_tok_matches
                metrics["stage_b"]["val_tokens_total"] += len(r_toks)

            row_data = {
                "id": rid,
                "split": split,
                "stage": stage,
                "prompt": prompt,
                "expected_value": expected_val,
                "prompt_token_count": len(p_toks),
                "response_token_count": len(r_toks),
                "first_response_pos": resp_start_idx,
                "first_target_token_id": first_target_tok,
                "first_target_token_str": tok.decode([first_target_tok]),
                "first_pred_token_id": first_pred_tok,
                "first_pred_token_str": tok.decode([first_pred_tok]),
                "first_token_match": first_tok_match,
                "first_token_rank": rank,
                "first_token_prob": round(prob, 6),
                "eos_pos": eos_pos,
                "eos_match": eos_match,
                "greedy_output": gen_out["text"],
                "free_gen_exact_match": exact_match,
            }
            split_results.append(row_data)
            all_audit_records.append(row_data)

        # Compute summary averages
        mean_rank = sum(metrics["first_token_ranks"]) / len(metrics["first_token_ranks"])
        mean_prob = sum(metrics["first_token_probs"]) / len(metrics["first_token_probs"])

        st_a_mean_rank = sum(metrics["stage_a"]["ranks"]) / len(metrics["stage_a"]["ranks"])
        st_a_mean_prob = sum(metrics["stage_a"]["probs"]) / len(metrics["stage_a"]["probs"])
        st_b_mean_rank = sum(metrics["stage_b"]["ranks"]) / len(metrics["stage_b"]["ranks"])
        st_b_mean_prob = sum(metrics["stage_b"]["probs"]) / len(metrics["stage_b"]["probs"])

        aggregated_splits[split] = {
            "total_examples": metrics["total_examples"],
            "free_gen_exact_matches": metrics["free_gen_exact_matches"],
            "free_gen_exact_pct": round(metrics["free_gen_exact_matches"] / metrics["total_examples"] * 100, 2),
            "first_token_tf_matches": metrics["first_token_tf_matches"],
            "first_token_tf_pct": round(metrics["first_token_tf_matches"] / metrics["total_examples"] * 100, 2),
            "eos_tf_matches": metrics["eos_tf_matches"],
            "eos_tf_pct": round(metrics["eos_tf_matches"] / metrics["total_examples"] * 100, 2),
            "mean_first_token_rank": round(mean_rank, 2),
            "mean_first_token_prob": round(mean_prob, 6),
            "combined_response_tf_pct": round(metrics["combined_response_tf_matches"] / metrics["combined_response_total"] * 100, 2),
            "stage_a": {
                "examples": metrics["stage_a"]["examples"],
                "free_gen_exact": metrics["stage_a"]["free_gen_exact"],
                "first_tok_tf_matches": metrics["stage_a"]["first_tok_tf_matches"],
                "first_tok_tf_pct": round(metrics["stage_a"]["first_tok_tf_matches"] / metrics["stage_a"]["examples"] * 100, 2),
                "eos_tf_matches": metrics["stage_a"]["eos_tf_matches"],
                "eos_tf_pct": round(metrics["stage_a"]["eos_tf_matches"] / metrics["stage_a"]["examples"] * 100, 2),
                "mean_first_token_rank": round(st_a_mean_rank, 2),
                "mean_first_token_prob": round(st_a_mean_prob, 6),
                "combined_tf_matches": metrics["stage_a"]["combined_tf_matches"],
                "combined_tf_total": metrics["stage_a"]["combined_tf_total"],
                "combined_tf_pct": round(metrics["stage_a"]["combined_tf_matches"] / metrics["stage_a"]["combined_tf_total"] * 100, 2),
            },
            "stage_b": {
                "examples": metrics["stage_b"]["examples"],
                "free_gen_exact": metrics["stage_b"]["free_gen_exact"],
                "first_tok_tf_matches": metrics["stage_b"]["first_tok_tf_matches"],
                "first_tok_tf_pct": round(metrics["stage_b"]["first_tok_tf_matches"] / metrics["stage_b"]["examples"] * 100, 2),
                "val_tokens_tf_pct": round(metrics["stage_b"]["val_tokens_tf_matches"] / metrics["stage_b"]["val_tokens_total"] * 100, 2),
                "eos_tf_matches": metrics["stage_b"]["eos_tf_matches"],
                "eos_tf_pct": round(metrics["stage_b"]["eos_tf_matches"] / metrics["stage_b"]["examples"] * 100, 2),
                "mean_first_token_rank": round(st_b_mean_rank, 2),
                "mean_first_token_prob": round(st_b_mean_prob, 6),
                "combined_tf_matches": metrics["stage_b"]["combined_tf_matches"],
                "combined_tf_total": metrics["stage_b"]["combined_tf_total"],
                "combined_tf_pct": round(metrics["stage_b"]["combined_tf_matches"] / metrics["stage_b"]["combined_tf_total"] * 100, 2),
            }
        }

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_audit_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_audit_records)
    print(f"Saved per-example audit CSV to: {OUTPUT_CSV.name}")

    # Write JSON
    audit_payload = {
        "checkpoint_path": str(CKPT_PATH),
        "total_records_audited": len(all_audit_records),
        "aggregated_splits": aggregated_splits,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)
    print(f"Saved audit JSON to: {OUTPUT_JSON.name}")

    # Print summary reconciliation table
    print("\n" + "=" * 70)
    print("RECONCILIATION SUMMARY TABLE")
    print("=" * 70)
    for sp, data in aggregated_splits.items():
        print(f"\n[{sp.upper()}] (N={data['total_examples']}):")
        print(f"  Stage A (Single Token):")
        print(f"    - First-Token TF Accuracy (Target Token 0): {data['stage_a']['first_tok_tf_matches']}/{data['stage_a']['examples']} ({data['stage_a']['first_tok_tf_pct']}%)")
        print(f"    - EOS TF Accuracy (EOS Token 1):            {data['stage_a']['eos_tf_matches']}/{data['stage_a']['examples']} ({data['stage_a']['eos_tf_pct']}%)")
        print(f"    - Combined Response TF Accuracy (Reported): {data['stage_a']['combined_tf_matches']}/{data['stage_a']['combined_tf_total']} ({data['stage_a']['combined_tf_pct']}%)")
        print(f"    - Mean First-Token Rank:                    {data['stage_a']['mean_first_token_rank']}")
        print(f"    - Mean First-Token Prob:                    {data['stage_a']['mean_first_token_prob']*100:.4f}%")
        print(f"    - Free Greedy Exact Match:                  {data['stage_a']['free_gen_exact']}/{data['stage_a']['examples']} (0.0%)")
        print(f"  Stage B (Multi-Token):")
        print(f"    - First-Token TF Accuracy:                  {data['stage_b']['first_tok_tf_matches']}/{data['stage_b']['examples']} ({data['stage_b']['first_tok_tf_pct']}%)")
        print(f"    - Value Subwords TF Accuracy:               {data['stage_b']['val_tokens_tf_pct']}%")
        print(f"    - EOS TF Accuracy:                          {data['stage_b']['eos_tf_matches']}/{data['stage_b']['examples']} ({data['stage_b']['eos_tf_pct']}%)")
        print(f"    - Combined Response TF Accuracy (Reported): {data['stage_b']['combined_tf_matches']}/{data['stage_b']['combined_tf_total']} ({data['stage_b']['combined_tf_pct']}%)")


if __name__ == "__main__":
    run_audit()
