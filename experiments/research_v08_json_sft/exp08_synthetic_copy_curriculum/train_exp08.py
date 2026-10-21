"""
train_exp08.py
ChakrView Experiment 08 — Synthetic Copying Curriculum Training Script

Trains ChakrMicro strictly on synthetic curriculum train.jsonl (2,000 examples).
Uses ordinary unweighted cross-entropy loss.
Predeclared budget: 400 optimizer steps (with batch size 8 = 3,200 sequence exposures, 1.6 passes over the 2,000 distinct examples).
Evaluates every 50 steps on:
- Training loss
- Validation loss, teacher-forced target-token accuracy, and free-generation exact match (199 examples)
- Test A (Familiar token identities in novel contexts)
- Test B (Held-out token identities)

Checkpoint selection is strictly based on Validation Exact Match.
Test A and Test B are purely diagnostic and never used for checkpoint selection.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
CHECKPOINT_DIR = EXP_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

EXPECTED_PARENT_SHA = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def evaluate_split(model: ChakrMicro, tokenizer, records, device="cpu", max_eval_records=None):
    model.eval()
    if max_eval_records and len(records) > max_eval_records:
        eval_recs = records[:max_eval_records]
    else:
        eval_recs = records

    total = len(eval_recs)
    first_token_hits = 0
    free_gen_hits = 0
    loss_sum = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Per-stage and per-template tracking
    by_stage = collections.defaultdict(lambda: {"total": 0, "first_token_hits": 0, "free_gen_hits": 0})
    by_template = collections.defaultdict(lambda: {"total": 0, "free_gen_hits": 0})

    with torch.no_grad():
        for rec in eval_recs:
            stage = rec.get("stage", "Unknown")
            tmpl = rec.get("template_type", "Unknown")
            prompt = rec["prompt"]
            target_val = rec["target_value"]
            target_tok_id = rec["target_token_id"]

            p_toks = tokenizer.encode(prompt)
            r_toks = tokenizer.encode(target_val)
            full_seq = [0] + p_toks + r_toks + [1]
            labels = [-100] * (len(p_toks) + 1) + r_toks + [1]

            inp_tensor = torch.tensor([full_seq], dtype=torch.long, device=device)
            lbl_tensor = torch.tensor([labels], dtype=torch.long, device=device)

            logits = model(inp_tensor)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = lbl_tensor[..., 1:].contiguous()
            l = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss_sum += l.item()

            # Teacher-forced target token logit position: len(p_toks)
            target_logit = logits[0, len(p_toks)]
            pred_tok_id = int(torch.argmax(target_logit).item())
            is_tf_hit = (pred_tok_id == target_tok_id)
            if is_tf_hit:
                first_token_hits += 1

            # Free generation
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
            by_stage[stage]["first_token_hits"] += int(is_tf_hit)
            by_stage[stage]["free_gen_hits"] += int(is_em)

            by_template[tmpl]["total"] += 1
            by_template[tmpl]["free_gen_hits"] += int(is_em)

    model.train()
    return {
        "loss": loss_sum / total,
        "first_token_acc": (first_token_hits / total) * 100.0,
        "first_token_hits": first_token_hits,
        "free_gen_em": (free_gen_hits / total) * 100.0,
        "free_gen_hits": free_gen_hits,
        "total": total,
        "by_stage": dict(by_stage),
        "by_template": dict(by_template)
    }

import collections

def main():
    print("=" * 70)
    print("CHAKRVIEW EXPERIMENT 08: SYNTHETIC COPYING CURRICULUM TRAINING")
    print("=" * 70)

    # 1. Verify parent hash
    parent_sha_before = compute_sha256(CANONICAL_CKPT)
    assert parent_sha_before == EXPECTED_PARENT_SHA, "Parent checkpoint altered!"
    print(f"Parent Checkpoint Hash: {parent_sha_before[:16]}... (Verified)")

    # 2. Tokenizer and Data
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    with open(DATA_DIR / "train.jsonl", "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f if line.strip()]
    with open(DATA_DIR / "val.jsonl", "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f if line.strip()]
    with open(DATA_DIR / "test_a_familiar_tokens.jsonl", "r", encoding="utf-8") as f:
        test_a_records = [json.loads(line) for line in f if line.strip()]
    with open(DATA_DIR / "test_b_held_out_tokens.jsonl", "r", encoding="utf-8") as f:
        test_b_records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded records -> Train: {len(train_records)} | Val: {len(val_records)} | Test A: {len(test_a_records)} | Test B: {len(test_b_records)}")

    # 3. Model setup in memory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    parent_state = torch.load(CANONICAL_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(parent_state["model_state_dict"])
    model.train()

    # Pre-tokenize training data
    train_tensors = []
    for rec in train_records:
        p_toks = tokenizer.encode(rec["prompt"])
        r_toks = tokenizer.encode(rec["target_value"])
        full_seq = [0] + p_toks + r_toks + [1]
        labels = [-100] * (len(p_toks) + 1) + r_toks + [1]
        train_tensors.append((
            torch.tensor(full_seq, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        ))

    # Pre-training baseline evaluation (Step 0)
    print("\n--- Pre-Training Step 0 Baseline ---")
    val_b0 = evaluate_split(model, tokenizer, val_records, device=device)
    test_a_b0 = evaluate_split(model, tokenizer, test_a_records, device=device)
    test_b_b0 = evaluate_split(model, tokenizer, test_b_records, device=device)
    print(f"Step 0 | Val EM: {val_b0['free_gen_em']:.1f}% | Test A EM: {test_a_b0['free_gen_em']:.1f}% | Test B EM: {test_b_b0['free_gen_em']:.1f}%")

    # Hyperparameters
    MAX_STEPS = 400
    EVAL_INTERVAL = 50
    BATCH_SIZE = 8
    LR = 3e-4
    WEIGHT_DECAY = 0.01

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    history = [{
        "step": 0,
        "train_loss": None,
        "val": val_b0,
        "test_a": test_a_b0,
        "test_b": test_b_b0
    }]

    best_val_em = val_b0["free_gen_em"]
    best_val_loss = val_b0["loss"]
    best_step = 0
    best_state_dict = copy.deepcopy(model.state_dict())

    item_idx = 0
    step_loss_accum = 0.0

    print(f"\nStarting curriculum optimization ({MAX_STEPS} steps, batch_size={BATCH_SIZE}, lr={LR})...")

    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad()
        batch_loss = 0.0

        for b in range(BATCH_SIZE):
            inp_seq, tgt_seq = train_tensors[item_idx]
            item_idx = (item_idx + 1) % len(train_tensors)

            inp = inp_seq.unsqueeze(0).to(device)
            tgt = tgt_seq.unsqueeze(0).to(device)

            logits = model(inp)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tgt[..., 1:].contiguous()

            l = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) / BATCH_SIZE
            l.backward()
            batch_loss += l.item() * BATCH_SIZE

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step_loss_accum += batch_loss

        if step % EVAL_INTERVAL == 0:
            avg_train_loss = step_loss_accum / EVAL_INTERVAL
            step_loss_accum = 0.0

            val_metrics = evaluate_split(model, tokenizer, val_records, device=device)
            test_a_metrics = evaluate_split(model, tokenizer, test_a_records, device=device)
            test_b_metrics = evaluate_split(model, tokenizer, test_b_records, device=device)

            history.append({
                "step": step,
                "train_loss": round(avg_train_loss, 4),
                "val": val_metrics,
                "test_a": test_a_metrics,
                "test_b": test_b_metrics
            })

            print(f"Step {step:3d}/{MAX_STEPS} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val EM: {val_metrics['free_gen_em']:5.1f}% | "
                  f"Test A EM (Familiar): {test_a_metrics['free_gen_em']:5.1f}% | "
                  f"Test B EM (Held-out): {test_b_metrics['free_gen_em']:5.1f}%")

            # Checkpoint selection strictly based on validation data
            if (val_metrics["free_gen_em"] > best_val_em) or (val_metrics["free_gen_em"] == best_val_em and val_metrics["loss"] < best_val_loss):
                best_val_em = val_metrics["free_gen_em"]
                best_val_loss = val_metrics["loss"]
                best_step = step
                best_state_dict = copy.deepcopy(model.state_dict())

    # Save checkpoints
    best_ckpt_path = CHECKPOINT_DIR / "exp08_best_checkpoint.pt"
    final_ckpt_path = CHECKPOINT_DIR / "exp08_final_checkpoint.pt"

    torch.save({
        "step": best_step,
        "model_state_dict": best_state_dict,
        "best_val_em": best_val_em,
        "best_val_loss": best_val_loss,
        "parent_hash": parent_sha_before
    }, best_ckpt_path)

    torch.save({
        "step": MAX_STEPS,
        "model_state_dict": model.state_dict(),
        "final_train_loss": avg_train_loss,
        "parent_hash": parent_sha_before
    }, final_ckpt_path)

    print(f"\nSaved Best Checkpoint (Selected via Val, Step {best_step}) to {best_ckpt_path}")
    print(f"Saved Final Checkpoint (Step {MAX_STEPS}) to {final_ckpt_path}")

    # Save history
    with open(EXP_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "best_step": best_step,
            "best_val_em": best_val_em,
            "best_checkpoint_sha256": compute_sha256(best_ckpt_path),
            "final_checkpoint_sha256": compute_sha256(final_ckpt_path)
        }, f, indent=2)

    # Post-training integrity check
    parent_sha_after = compute_sha256(CANONICAL_CKPT)
    assert parent_sha_after == parent_sha_before, "CRITICAL: Parent checkpoint modified!"
    print(f"[PASS] Post-training integrity verified. Parent checkpoint hash unchanged: {parent_sha_after[:16]}...")

if __name__ == "__main__":
    main()
