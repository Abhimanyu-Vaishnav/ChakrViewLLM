"""
evaluate_exp08.py
ChakrView Experiment 08 — Comprehensive Diagnostic Evaluation Script

Loads:
- Canonical Parent Checkpoint (Step-10000)
- Best Checkpoint selected strictly by Validation Exact Match (from checkpoints/exp08_best_checkpoint.pt)

Evaluates on:
1. Validation Set (199 examples)
2. Test A: Familiar Token Identities in Novel Contexts (198 examples)
3. Test B: Held-Out Token Identities (300 examples)

Computes:
- Teacher-Forced Target-Token Accuracy (excluding EOS).
- Free-Generation Exact Match (Argmax greedy).
- Mean Target Rank & Target Probability at first response token position.
- Per-stage breakdown (Stage A: Single-token direct; Stage B: Distractor multi-item; Stage C: Varied syntax).
- Per-template and target-position accuracy breakdown.
- Output diversity & repetition collapse analysis.
- Qualitative example outputs (hits and misses).
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
CHECKPOINT_DIR = EXP_DIR / "checkpoints"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def evaluate_full_split(model: ChakrMicro, tokenizer, split_path: Path, split_name: str, device="cpu"):
    with open(split_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    model.eval()
    total = len(records)
    first_token_hits = 0
    free_gen_hits = 0
    ranks = []
    probs = []
    predictions = []

    by_stage = collections.defaultdict(lambda: {"total": 0, "tf_hits": 0, "em_hits": 0})
    by_template = collections.defaultdict(lambda: {"total": 0, "em_hits": 0})
    by_pos = collections.defaultdict(lambda: {"total": 0, "em_hits": 0})

    with torch.no_grad():
        for rec in records:
            stage = rec.get("stage", "Unknown")
            tmpl = rec.get("template_type", "Unknown")
            pos = rec.get("target_position", 0)
            prompt = rec["prompt"]
            target_val = rec["target_value"]
            target_tok_id = rec["target_token_id"]

            p_toks = tokenizer.encode(prompt)
            r_toks = tokenizer.encode(target_val)

            # 1. Teacher-Forced Target Token Logit
            full_seq = [0] + p_toks + r_toks + [1]
            inp_tensor = torch.tensor([full_seq], dtype=torch.long, device=device)
            logits = model(inp_tensor)
            
            target_logit = logits[0, len(p_toks)]
            pred_tok_id = int(torch.argmax(target_logit).item())
            softmax_probs = F.softmax(target_logit, dim=-1)
            target_prob = softmax_probs[target_tok_id].item()
            sorted_indices = torch.argsort(target_logit, descending=True)
            target_rank = int((sorted_indices == target_tok_id).nonzero(as_tuple=True)[0].item()) + 1

            ranks.append(target_rank)
            probs.append(target_prob)
            is_tf_hit = (pred_tok_id == target_tok_id)
            if is_tf_hit:
                first_token_hits += 1

            # 2. Free Autoregressive Generation (Argmax)
            curr = [0] + list(p_toks)
            gen_tokens = []
            for _ in range(8):
                g_inp = torch.tensor([curr], dtype=torch.long, device=device)
                g_logits = model(g_inp)
                nxt = int(torch.argmax(g_logits[0, -1, :]).item())
                if nxt == 1:
                    break
                gen_tokens.append(nxt)
                curr.append(nxt)

            try:
                gen_text = tokenizer.decode(gen_tokens, errors="replace").strip()
            except Exception:
                gen_text = tokenizer.decode(gen_tokens).strip()

            is_em = (gen_text == target_val.strip())
            if is_em:
                free_gen_hits += 1

            # Breakdown
            by_stage[stage]["total"] += 1
            by_stage[stage]["tf_hits"] += int(is_tf_hit)
            by_stage[stage]["em_hits"] += int(is_em)

            by_template[tmpl]["total"] += 1
            by_template[tmpl]["em_hits"] += int(is_em)

            by_pos[pos]["total"] += 1
            by_pos[pos]["em_hits"] += int(is_em)

            predictions.append({
                "stage": stage,
                "template": tmpl,
                "prompt": prompt,
                "expected": target_val,
                "generated": gen_text,
                "first_token_pred_id": pred_tok_id,
                "target_token_id": target_tok_id,
                "target_prob": target_prob,
                "target_rank": target_rank,
                "exact_match": is_em
            })

    unique_outputs = set(p["generated"] for p in predictions)
    output_diversity = len(unique_outputs) / total if total > 0 else 0.0

    return {
        "split_name": split_name,
        "total_examples": total,
        "first_token_acc": (first_token_hits / total) * 100.0,
        "first_token_hits": first_token_hits,
        "free_gen_em": (free_gen_hits / total) * 100.0,
        "free_gen_hits": free_gen_hits,
        "mean_target_prob": float(sum(probs) / total),
        "mean_target_rank": float(sum(ranks) / total),
        "unique_outputs_count": len(unique_outputs),
        "output_diversity": output_diversity,
        "by_stage": {k: {"total": v["total"], "tf_acc": (v["tf_hits"]/v["total"])*100.0, "em_acc": (v["em_hits"]/v["total"])*100.0} for k, v in by_stage.items()},
        "by_template": {k: {"total": v["total"], "em_acc": (v["em_hits"]/v["total"])*100.0} for k, v in by_template.items()},
        "by_position": {k: {"total": v["total"], "em_acc": (v["em_hits"]/v["total"])*100.0} for k, v in by_pos.items()},
        "sample_predictions": predictions[:10]
    }

def main():
    print("=" * 70)
    print("EXP 08: COMPREHENSIVE EVALUATION HARNESS")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    best_ckpt_path = CHECKPOINT_DIR / "exp08_best_checkpoint.pt"
    if not best_ckpt_path.exists():
        print(f"Waiting for best checkpoint at {best_ckpt_path}...")
        return

    # Load Best Model
    best_model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    best_state = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    best_model.load_state_dict(best_state["model_state_dict"])
    best_model.eval()
    print(f"Loaded Best Model from Step {best_state.get('step')} (Selected via Validation Set).")

    # Load Canonical Parent
    parent_model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    parent_state = torch.load(CANONICAL_CKPT, map_location=device, weights_only=False)
    parent_model.load_state_dict(parent_state["model_state_dict"])
    parent_model.eval()

    val_path = DATA_DIR / "val.jsonl"
    test_a_path = DATA_DIR / "test_a_familiar_tokens.jsonl"
    test_b_path = DATA_DIR / "test_b_held_out_tokens.jsonl"

    print("\n--- Evaluating Baseline Parent ---")
    parent_val = evaluate_full_split(parent_model, tokenizer, val_path, "Val_Parent", device=device)
    parent_test_a = evaluate_full_split(parent_model, tokenizer, test_a_path, "TestA_Parent", device=device)
    parent_test_b = evaluate_full_split(parent_model, tokenizer, test_b_path, "TestB_Parent", device=device)

    print("\n--- Evaluating Exp 08 Trained Model ---")
    trained_val = evaluate_full_split(best_model, tokenizer, val_path, "Val_Exp08", device=device)
    trained_test_a = evaluate_full_split(best_model, tokenizer, test_a_path, "TestA_Exp08", device=device)
    trained_test_b = evaluate_full_split(best_model, tokenizer, test_b_path, "TestB_Exp08", device=device)

    results = {
        "best_step": best_state.get("step"),
        "best_checkpoint_sha256": compute_sha256(best_ckpt_path),
        "parent_checkpoint_sha256": compute_sha256(CANONICAL_CKPT),
        "baseline_parent": {
            "val": parent_val,
            "test_a": parent_test_a,
            "test_b": parent_test_b
        },
        "exp08_trained": {
            "val": trained_val,
            "test_a": trained_test_a,
            "test_b": trained_test_b
        }
    }

    results_path = EXP_DIR / "EVALUATION_RESULTS.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved complete results to {results_path}")

    # Summary Display
    print("\n" + "=" * 70)
    print("EXP 08 EVALUATION SUMMARY:")
    print("=" * 70)
    print(f"Validation:  TF Acc: {trained_val['first_token_acc']:5.1f}% | Free EM: {trained_val['free_gen_em']:5.1f}% | Mean Rank: {trained_val['mean_target_rank']:6.1f} | Diversity: {trained_val['output_diversity']:.2f}")
    print(f"Test A (Fam): TF Acc: {trained_test_a['first_token_acc']:5.1f}% | Free EM: {trained_test_a['free_gen_em']:5.1f}% | Mean Rank: {trained_test_a['mean_target_rank']:6.1f} | Diversity: {trained_test_a['output_diversity']:.2f}")
    print(f"Test B (Held):TF Acc: {trained_test_b['first_token_acc']:5.1f}% | Free EM: {trained_test_b['free_gen_em']:5.1f}% | Mean Rank: {trained_test_b['mean_target_rank']:6.1f} | Diversity: {trained_test_b['output_diversity']:.2f}")

if __name__ == "__main__":
    main()
