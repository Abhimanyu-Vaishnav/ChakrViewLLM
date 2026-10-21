"""
evaluate_exp07.py
ChakrView Experiment 07 — Comprehensive Evaluation Script

Evaluates checkpoints on:
1. Tiny Train Set (16 examples)
2. Unseen Held-Out Test Set (25 examples)

Reports:
1. Teacher-forced first-token accuracy (target response token, excluding EOS).
2. Free-generation exact match (greedy argmax).
3. First-token rank and probability distribution metrics.
4. Output diversity and generic prediction collapse analysis.
5. Strict artifact integrity check.
"""

from __future__ import annotations

import copy
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

TRAIN_DATA_PATH = DATA_DIR / "tiny_train_16.jsonl"
UNSEEN_DATA_PATH = DATA_DIR / "unseen_test_25.jsonl"
CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_dataset(model: ChakrMicro, tokenizer, dataset_path: Path, split_name: str, device="cpu"):
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    model.eval()
    total = len(records)
    first_token_hits = 0
    free_gen_hits = 0
    ranks = []
    probs = []
    predictions = []

    with torch.no_grad():
        for rec in records:
            prompt = rec["prompt"]
            target_val = rec["response"]
            p_toks = tokenizer.encode(prompt)
            r_toks = tokenizer.encode(target_val)
            first_target_id = r_toks[0]

            # 1. Teacher-Forced First Token
            full_ids = [0] + p_toks + r_toks + [1]
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
            logits = model(input_tensor)
            
            # First response token logit is at index len(p_toks)
            first_logit = logits[0, len(p_toks)]
            pred_tok_id = int(torch.argmax(first_logit).item())
            
            # Rank and Probability of the target token
            softmax_probs = F.softmax(first_logit, dim=-1)
            target_prob = softmax_probs[first_target_id].item()
            sorted_indices = torch.argsort(first_logit, descending=True)
            target_rank = int((sorted_indices == first_target_id).nonzero(as_tuple=True)[0].item()) + 1

            ranks.append(target_rank)
            probs.append(target_prob)

            if pred_tok_id == first_target_id:
                first_token_hits += 1

            # 2. Free Autoregressive Generation (Argmax)
            curr = [0] + list(p_toks)
            gen_tokens = []
            for _ in range(10):
                inp = torch.tensor([curr], dtype=torch.long, device=device)
                gen_logits = model(inp)
                nxt = int(torch.argmax(gen_logits[0, -1, :]).item())
                if nxt == 1:
                    break
                gen_tokens.append(nxt)
                curr.append(nxt)

            try:
                gen_text = tokenizer.decode(gen_tokens, errors="replace").strip()
            except Exception:
                gen_text = tokenizer.decode(gen_tokens).strip()

            is_match = (gen_text == target_val.strip())
            if is_match:
                free_gen_hits += 1

            predictions.append({
                "prompt": prompt,
                "expected": target_val,
                "generated": gen_text,
                "first_token_pred_id": pred_tok_id,
                "first_target_id": first_target_id,
                "target_prob": target_prob,
                "target_rank": target_rank,
                "exact_match": is_match
            })

    # Output diversity analysis
    gen_outputs = [p["generated"] for p in predictions]
    unique_outputs = set(gen_outputs)
    output_diversity = len(unique_outputs) / total

    return {
        "split_name": split_name,
        "total_examples": total,
        "first_token_acc": (first_token_hits / total) * 100.0,
        "first_token_hits": first_token_hits,
        "free_gen_em": (free_gen_hits / total) * 100.0,
        "free_gen_hits": free_gen_hits,
        "mean_target_prob": float(sum(probs) / total),
        "mean_target_rank": float(sum(ranks) / total),
        "unique_outputs": len(unique_outputs),
        "output_diversity": output_diversity,
        "predictions": predictions
    }


def main():
    print("=" * 70)
    print("EXP 07: EVALUATION OF BEST AND CANONICAL CHECKPOINTS")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # 1. Evaluate Canonical Parent Checkpoint (Baseline)
    print("\n--- [1] Evaluating Canonical Parent Checkpoint ---")
    parent_model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    parent_state = torch.load(CANONICAL_CKPT, map_location=device, weights_only=False)
    parent_model.load_state_dict(parent_state["model_state_dict"])
    
    baseline_train = evaluate_dataset(parent_model, tokenizer, TRAIN_DATA_PATH, "Train_Baseline", device=device)
    baseline_unseen = evaluate_dataset(parent_model, tokenizer, UNSEEN_DATA_PATH, "Unseen_Baseline", device=device)
    print(f"Parent -> Train EM: {baseline_train['free_gen_em']:.1f}% | Unseen EM: {baseline_unseen['free_gen_em']:.1f}%")

    # 2. Evaluate Experiment 07 Best Checkpoint
    best_ckpt_path = CHECKPOINT_DIR / "exp07_best_checkpoint.pt"
    if not best_ckpt_path.exists():
        print(f"Waiting for {best_ckpt_path} to exist...")
        return

    print("\n--- [2] Evaluating Experiment 07 Best Checkpoint ---")
    best_model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    best_state = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    best_model.load_state_dict(best_state["model_state_dict"])
    
    exp07_train = evaluate_dataset(best_model, tokenizer, TRAIN_DATA_PATH, "Train_Exp07", device=device)
    exp07_unseen = evaluate_dataset(best_model, tokenizer, UNSEEN_DATA_PATH, "Unseen_Exp07", device=device)

    print(f"Exp07 -> Train First-Token Acc: {exp07_train['first_token_acc']:.1f}% ({exp07_train['first_token_hits']}/16) | "
          f"Train Free EM: {exp07_train['free_gen_em']:.1f}% ({exp07_train['free_gen_hits']}/16) | "
          f"Mean Rank: {exp07_train['mean_target_rank']:.1f} | Mean Prob: {exp07_train['mean_target_prob']*100:.2f}%")
    
    print(f"Exp07 -> Unseen First-Token Acc: {exp07_unseen['first_token_acc']:.1f}% ({exp07_unseen['first_token_hits']}/25) | "
          f"Unseen Free EM: {exp07_unseen['free_gen_em']:.1f}% ({exp07_unseen['free_gen_hits']}/25) | "
          f"Mean Rank: {exp07_unseen['mean_target_rank']:.1f} | Mean Prob: {exp07_unseen['mean_target_prob']*100:.2f}%")

    # Save detailed evaluation artifact
    results_path = EXP_DIR / "EVALUATION_RESULTS.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline": {
                "train": baseline_train,
                "unseen": baseline_unseen
            },
            "exp07_trained": {
                "step": best_state.get("step"),
                "train": exp07_train,
                "unseen": exp07_unseen
            },
            "integrity": {
                "canonical_parent_hash": compute_sha256(CANONICAL_CKPT),
                "exp07_best_checkpoint_hash": compute_sha256(best_ckpt_path)
            }
        }, f, indent=2)
    print(f"\nSaved complete evaluation results to {results_path}")


if __name__ == "__main__":
    main()
