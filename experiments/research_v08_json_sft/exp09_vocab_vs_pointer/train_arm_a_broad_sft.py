"""
train_arm_a_broad_sft.py
ChakrView Experiment 09 — Arm A: Broader-Vocabulary Causal SFT

Trains standard ChakrMicro on the expanded 2,500-example curriculum covering 450 distinct target tokens.
Uses ordinary unweighted causal cross-entropy loss.
Predeclared budget: 400 optimizer steps (batch size 8).
Checkpoint selection strictly based on Validation split (val.jsonl).
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
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

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

def evaluate_split(model: ChakrMicro, tokenizer, records, device="cpu"):
    model.eval()
    total = len(records)
    first_token_hits = 0
    free_gen_hits = 0
    loss_sum = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for rec in records:
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

            target_logit = logits[0, len(p_toks)]
            pred_tok_id = int(torch.argmax(target_logit).item())
            if pred_tok_id == target_tok_id:
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

            if gen_text == target_val.strip():
                free_gen_hits += 1

    model.train()
    return {
        "loss": loss_sum / total,
        "first_token_acc": (first_token_hits / total) * 100.0,
        "free_gen_em": (free_gen_hits / total) * 100.0,
        "total": total
    }

def main():
    print("=" * 70)
    print("EXP 09 ARM A: BROADER-VOCABULARY CAUSAL SFT TRAINING")
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

    model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    parent_state = torch.load(CANONICAL_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(parent_state["model_state_dict"])
    model.train()

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

    MAX_STEPS = 400
    EVAL_INTERVAL = 50
    BATCH_SIZE = 8
    LR = 3e-4
    WEIGHT_DECAY = 0.01

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    val_b0 = evaluate_split(model, tokenizer, val_records, device=device)
    print(f"Step 0 Baseline | Val Loss: {val_b0['loss']:.4f} | Val EM: {val_b0['free_gen_em']:.1f}%")

    best_val_em = val_b0["free_gen_em"]
    best_val_loss = val_b0["loss"]
    best_step = 0
    best_state_dict = copy.deepcopy(model.state_dict())

    history = [{
        "step": 0,
        "train_loss": None,
        "val_loss": val_b0["loss"],
        "val_em": val_b0["free_gen_em"]
    }]

    item_idx = 0
    step_loss_accum = 0.0

    print(f"\nStarting Arm A training loop ({MAX_STEPS} steps, batch_size={BATCH_SIZE}, lr={LR})...")
    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad()
        batch_loss = 0.0

        for _ in range(BATCH_SIZE):
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
            history.append({
                "step": step,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_em": val_metrics["free_gen_em"]
            })

            print(f"Step {step:3d}/{MAX_STEPS} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | Val EM: {val_metrics['free_gen_em']:5.1f}%")

            if (val_metrics["free_gen_em"] > best_val_em) or (val_metrics["free_gen_em"] == best_val_em and val_metrics["loss"] < best_val_loss):
                best_val_em = val_metrics["free_gen_em"]
                best_val_loss = val_metrics["loss"]
                best_step = step
                best_state_dict = copy.deepcopy(model.state_dict())

    best_ckpt_path = CHECKPOINT_DIR / "arm_a_best_checkpoint.pt"
    torch.save({
        "step": best_step,
        "model_state_dict": best_state_dict,
        "best_val_em": best_val_em,
        "parent_hash": parent_sha_before
    }, best_ckpt_path)

    with open(EXP09_DIR / "arm_a_training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "best_step": best_step,
            "best_val_em": best_val_em,
            "checkpoint_sha256": compute_sha256(best_ckpt_path)
        }, f, indent=2)

    print(f"\nArm A training finished. Best Checkpoint (Step {best_step}, Val EM {best_val_em:.1f}%) saved to {best_ckpt_path}")

if __name__ == "__main__":
    main()
