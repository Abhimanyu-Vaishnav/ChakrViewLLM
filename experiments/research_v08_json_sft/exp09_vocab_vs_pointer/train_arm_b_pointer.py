"""
train_arm_b_pointer.py
ChakrView Experiment 09 — Arm B: Isolated Pointer-Head Training

Trains ChakrPointerModel on the exact same 2,500 training examples.
Uses CrossEntropyLoss on the source token position index.
Predeclared budget: 400 optimizer steps (batch size 8).
Checkpoint selection strictly based on Validation split (val.jsonl) pointer exact match.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v08_json_sft.exp09_vocab_vs_pointer.pointer_head import ChakrPointerModel

EXP09_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP09_DIR / "data"
CHECKPOINT_DIR = EXP09_DIR / "checkpoints"
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

def evaluate_pointer_split(model: ChakrPointerModel, tokenizer, records, device="cpu"):
    model.eval()
    total = len(records)
    pos_hits = 0
    token_em_hits = 0
    loss_sum = 0.0

    with torch.no_grad():
        for rec in records:
            prompt = rec["prompt"]
            target_val = rec["target_value"]
            target_tok_id = rec["target_token_id"]
            src_pos = rec["source_token_index"]
            cand_indices = rec.get("candidate_token_indices", [src_pos])

            p_toks = tokenizer.encode(prompt)
            full_seq = [0] + p_toks
            seq_len = len(full_seq)

            inp_tensor = torch.tensor([full_seq], dtype=torch.long, device=device)
            # Candidate mask
            mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
            for c_idx in cand_indices:
                if c_idx < seq_len:
                    mask[0, c_idx] = True

            # If src_pos is out of bounds or not in mask, ensure valid fallback
            valid_src = min(src_pos, seq_len - 1)
            mask[0, valid_src] = True

            scores = model(inp_tensor, mask)
            tgt_tensor = torch.tensor([valid_src], dtype=torch.long, device=device)
            l = F.cross_entropy(scores, tgt_tensor)
            loss_sum += l.item()

            pred_pos = int(torch.argmax(scores, dim=-1)[0].item())
            copied_tok_id = int(inp_tensor[0, pred_pos].item())

            if pred_pos == valid_src:
                pos_hits += 1
            if copied_tok_id == target_tok_id:
                token_em_hits += 1

    model.train()
    return {
        "loss": loss_sum / total,
        "position_acc": (pos_hits / total) * 100.0,
        "token_em": (token_em_hits / total) * 100.0,
        "total": total
    }

def main():
    print("=" * 70)
    print("EXP 09 ARM B: ISOLATED POINTER-HEAD TRAINING")
    print("=" * 70)

    parent_sha_before = compute_sha256(CANONICAL_CKPT)
    assert parent_sha_before == EXPECTED_PARENT_SHA, "Parent checkpoint altered!"
    print(f"Parent Checkpoint Hash: {parent_sha_before[:16]}... (Verified)")

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    with open(DATA_DIR / "train.jsonl", "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f if line.strip()]
    with open(DATA_DIR / "val.jsonl", "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded Train: {len(train_records)} | Val: {len(val_records)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    # Initialize pointer model from canonical parent weights
    config = ModelConfig(max_seq_len=512)
    model = ChakrPointerModel(config, freeze_backbone=False).to(device)
    parent_state = torch.load(CANONICAL_CKPT, map_location=device, weights_only=False)
    model.backbone.load_state_dict(parent_state["model_state_dict"])
    model.train()

    # Pre-process training tensors
    train_items = []
    for rec in train_records:
        p_toks = tokenizer.encode(rec["prompt"])
        full_seq = [0] + p_toks
        seq_len = len(full_seq)
        src_pos = min(rec["source_token_index"], seq_len - 1)
        cands = rec.get("candidate_token_indices", [src_pos])

        mask = [False] * seq_len
        for c in cands:
            if c < seq_len:
                mask[c] = True
        mask[src_pos] = True

        train_items.append((
            torch.tensor(full_seq, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
            torch.tensor(src_pos, dtype=torch.long)
        ))

    MAX_STEPS = 400
    EVAL_INTERVAL = 50
    BATCH_SIZE = 8
    LR = 3e-4
    WEIGHT_DECAY = 0.01

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    val_b0 = evaluate_pointer_split(model, tokenizer, val_records, device=device)
    print(f"Step 0 Baseline | Val Loss: {val_b0['loss']:.4f} | Val Pos Acc: {val_b0['position_acc']:.1f}% | Val Token EM: {val_b0['token_em']:.1f}%")

    best_val_em = val_b0["token_em"]
    best_val_loss = val_b0["loss"]
    best_step = 0
    best_state_dict = copy.deepcopy(model.state_dict())

    history = [{
        "step": 0,
        "train_loss": None,
        "val_loss": val_b0["loss"],
        "val_token_em": val_b0["token_em"],
        "val_pos_acc": val_b0["position_acc"]
    }]

    item_idx = 0
    step_loss_accum = 0.0

    print(f"\nStarting Arm B Pointer training loop ({MAX_STEPS} steps, batch_size={BATCH_SIZE}, lr={LR})...")
    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad()
        batch_loss = 0.0

        for _ in range(BATCH_SIZE):
            inp_seq, mask, tgt_pos = train_items[item_idx]
            item_idx = (item_idx + 1) % len(train_items)

            inp = inp_seq.unsqueeze(0).to(device)
            m = mask.unsqueeze(0).to(device)
            tgt = tgt_pos.unsqueeze(0).to(device)

            scores = model(inp, m)
            l = F.cross_entropy(scores, tgt) / BATCH_SIZE
            l.backward()
            batch_loss += l.item() * BATCH_SIZE

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step_loss_accum += batch_loss

        if step % EVAL_INTERVAL == 0:
            avg_train_loss = step_loss_accum / EVAL_INTERVAL
            step_loss_accum = 0.0

            val_metrics = evaluate_pointer_split(model, tokenizer, val_records, device=device)
            history.append({
                "step": step,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_token_em": val_metrics["token_em"],
                "val_pos_acc": val_metrics["position_acc"]
            })

            print(f"Step {step:3d}/{MAX_STEPS} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | Val Pos Acc: {val_metrics['position_acc']:5.1f}% | "
                  f"Val Token EM: {val_metrics['token_em']:5.1f}%")

            if (val_metrics["token_em"] > best_val_em) or (val_metrics["token_em"] == best_val_em and val_metrics["loss"] < best_val_loss):
                best_val_em = val_metrics["token_em"]
                best_val_loss = val_metrics["loss"]
                best_step = step
                best_state_dict = copy.deepcopy(model.state_dict())

    best_ckpt_path = CHECKPOINT_DIR / "arm_b_best_checkpoint.pt"
    torch.save({
        "step": best_step,
        "model_state_dict": best_state_dict,
        "best_val_em": best_val_em,
        "parent_hash": parent_sha_before
    }, best_ckpt_path)

    with open(EXP09_DIR / "arm_b_training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "best_step": best_step,
            "best_val_em": best_val_em,
            "checkpoint_sha256": compute_sha256(best_ckpt_path)
        }, f, indent=2)

    print(f"\nArm B training finished. Best Checkpoint (Step {best_step}, Val Token EM {best_val_em:.1f}%) saved to {best_ckpt_path}")

if __name__ == "__main__":
    main()
