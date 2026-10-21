"""
train_exp10_2.py
ChakrView Experiment 10.2 — Start-Conditioned Span Pointer Training Script

Trains ChakrStartConditionedSpanPointer on unchanged train.jsonl (2,400 records).
Conditions the end-pointer query on h_query + h_start.
Settings: 400 optimizer steps, AdamW (lr=3e-4, batch_size=8, seed=1020).
Checkpoint selection strictly based on Validation split (val.jsonl) Span Exact Match.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v08_json_sft.exp10_span_pointer.exp10_2_end_boundary.start_conditioned_span_model import ChakrStartConditionedSpanPointer

EXP10_DIR = Path(__file__).resolve().parents[1]
EXP10_2_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP10_DIR / "data"
CHECKPOINT_DIR = EXP10_2_DIR / "checkpoints"
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

def evaluate_span_split(model: ChakrStartConditionedSpanPointer, tokenizer, records, device="cpu"):
    model.eval()
    total = len(records)
    span_hits = 0
    start_hits = 0
    end_hits = 0
    val_em_hits = 0
    loss_sum = 0.0

    with torch.no_grad():
        for rec in records:
            prompt = rec["prompt"]
            target_val = rec["target_value"]
            s_gt = rec["start_token_index"]
            e_gt = rec["end_token_index"]

            p_toks = tokenizer.encode(prompt)
            full_seq = [0] + p_toks
            seq_len = len(full_seq)

            inp = torch.tensor([full_seq], dtype=torch.long, device=device)
            mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
            mask[0, 1:seq_len - 2] = True
            mask[0, s_gt] = True
            mask[0, e_gt] = True

            tgt_s = torch.tensor([s_gt], dtype=torch.long, device=device)
            tgt_e = torch.tensor([e_gt], dtype=torch.long, device=device)

            s_scores, e_scores = model(inp, mask, gold_start=tgt_s)
            l = F.cross_entropy(s_scores, tgt_s) + F.cross_entropy(e_scores, tgt_e)
            loss_sum += l.item()

            pred_s, pred_e, toks, _ = model.predict_span(inp, mask)
            dec_text = tokenizer.decode(toks).strip().lstrip(":=").rstrip(",").strip()

            is_s = (pred_s == s_gt)
            is_e = (pred_e == e_gt)
            is_span = is_s and is_e
            is_val = (dec_text == target_val.strip())

            if is_s:
                start_hits += 1
            if is_e:
                end_hits += 1
            if is_span:
                span_hits += 1
            if is_val:
                val_em_hits += 1

    model.train()
    return {
        "loss": loss_sum / total,
        "span_em": (span_hits / total) * 100.0,
        "value_em": (val_em_hits / total) * 100.0,
        "start_acc": (start_hits / total) * 100.0,
        "end_acc": (end_hits / total) * 100.0,
        "total": total
    }

def main():
    print("=" * 70)
    print("EXP 10.2: START-CONDITIONED SPAN POINTER TRAINING")
    print("=" * 70)

    p_sha = compute_sha256(CANONICAL_CKPT)
    assert p_sha == EXPECTED_PARENT_SHA, "Canonical parent hash mismatch!"
    print(f"Parent Checkpoint Hash: {p_sha[:16]}... (Verified)")

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    with open(DATA_DIR / "train.jsonl", "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f if line.strip()]
    with open(DATA_DIR / "val.jsonl", "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded Train: {len(train_records)} | Val: {len(val_records)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    config = ModelConfig(max_seq_len=512)
    model = ChakrStartConditionedSpanPointer(config, max_span_len=8, freeze_backbone=False).to(device)
    parent_state = torch.load(CANONICAL_CKPT, map_location=device, weights_only=False)
    model.backbone.load_state_dict(parent_state["model_state_dict"])
    model.train()

    train_items = []
    for rec in train_records:
        p_toks = tokenizer.encode(rec["prompt"])
        full_seq = [0] + p_toks
        seq_len = len(full_seq)
        s_gt = min(rec["start_token_index"], seq_len - 1)
        e_gt = min(rec["end_token_index"], seq_len - 1)

        mask = [False] * seq_len
        for i in range(1, seq_len - 2):
            mask[i] = True
        mask[s_gt] = True
        mask[e_gt] = True

        train_items.append((
            torch.tensor(full_seq, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(s_gt, dtype=torch.long),
            torch.tensor(e_gt, dtype=torch.long)
        ))

    MAX_STEPS = 400
    EVAL_INTERVAL = 50
    BATCH_SIZE = 8
    LR = 3e-4
    WEIGHT_DECAY = 0.01

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    val_b0 = evaluate_span_split(model, tokenizer, val_records, device=device)
    print(f"Step 0 Baseline | Val Loss: {val_b0['loss']:.4f} | Val Span EM: {val_b0['span_em']:.1f}% | Val Value EM: {val_b0['value_em']:.1f}%")

    best_val_em = val_b0["span_em"]
    best_val_loss = val_b0["loss"]
    best_step = 0
    best_state_dict = copy.deepcopy(model.state_dict())

    history = [{
        "step": 0,
        "train_loss": None,
        "val_loss": val_b0["loss"],
        "val_span_em": val_b0["span_em"],
        "val_value_em": val_b0["value_em"]
    }]

    item_idx = 0
    step_loss_accum = 0.0

    print(f"\nStarting Start-Conditioned training loop ({MAX_STEPS} steps, batch_size={BATCH_SIZE}, lr={LR})...")
    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad()
        batch_loss = 0.0

        for _ in range(BATCH_SIZE):
            inp_seq, mask, s_tgt, e_tgt = train_items[item_idx]
            item_idx = (item_idx + 1) % len(train_items)

            inp = inp_seq.unsqueeze(0).to(device)
            m = mask.unsqueeze(0).to(device)
            tgt_s = s_tgt.unsqueeze(0).to(device)
            tgt_e = e_tgt.unsqueeze(0).to(device)

            s_scores, e_scores = model(inp, m, gold_start=tgt_s)
            l = (F.cross_entropy(s_scores, tgt_s) + F.cross_entropy(e_scores, tgt_e)) / BATCH_SIZE
            l.backward()
            batch_loss += l.item() * BATCH_SIZE

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step_loss_accum += batch_loss

        if step % EVAL_INTERVAL == 0:
            avg_train_loss = step_loss_accum / EVAL_INTERVAL
            step_loss_accum = 0.0

            val_metrics = evaluate_span_split(model, tokenizer, val_records, device=device)
            history.append({
                "step": step,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_span_em": val_metrics["span_em"],
                "val_value_em": val_metrics["value_em"],
                "val_start_acc": val_metrics["start_acc"],
                "val_end_acc": val_metrics["end_acc"]
            })

            print(f"Step {step:3d}/{MAX_STEPS} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | Val Span EM: {val_metrics['span_em']:5.1f}% | "
                  f"Val Value EM: {val_metrics['value_em']:5.1f}% | Start Acc: {val_metrics['start_acc']:5.1f}% | End Acc: {val_metrics['end_acc']:5.1f}%")

            if (val_metrics["span_em"] > best_val_em) or (val_metrics["span_em"] == best_val_em and val_metrics["loss"] < best_val_loss):
                best_val_em = val_metrics["span_em"]
                best_val_loss = val_metrics["loss"]
                best_step = step
                best_state_dict = copy.deepcopy(model.state_dict())

    best_ckpt_path = CHECKPOINT_DIR / "exp10_2_start_cond_best.pt"
    torch.save({
        "step": best_step,
        "model_state_dict": best_state_dict,
        "best_val_em": best_val_em,
        "parent_hash": p_sha
    }, best_ckpt_path)

    with open(EXP10_2_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "best_step": best_step,
            "best_val_em": best_val_em,
            "checkpoint_sha256": compute_sha256(best_ckpt_path)
        }, f, indent=2)

    print(f"\nTraining finished. Best Checkpoint (Step {best_step}, Val Span EM {best_val_em:.1f}%) saved to {best_ckpt_path}")

if __name__ == "__main__":
    main()
