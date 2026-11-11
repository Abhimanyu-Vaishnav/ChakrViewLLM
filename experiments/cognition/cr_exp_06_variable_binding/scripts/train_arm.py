"""
Stage 4: Unified Training Script for Control Arm A and Treatment Arm B
Executes bounded SFT (max 150 optimizer steps) on CPU with AdamW, grad clipping,
validation checkpoints, and logging.
"""

import sys
import os
import json
import time
import math
import hashlib
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

def hash_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def prepare_batch_tensors(samples, tok, relational_multiplier=1.0, max_len=512):
    batch_x = []
    batch_y = []
    batch_weights = []

    for s in samples:
        prompt = s["prompt"]
        response = " " + s["response"] + "\n"
        spans = s.get("target_spans", [])

        p_ids = tok.encode(prompt)
        r_ids = tok.encode(response)

        all_ids = p_ids + r_ids
        if len(all_ids) > max_len:
            all_ids = all_ids[:max_len]
            # Truncate response tokens if needed
            r_ids = all_ids[len(p_ids):]

        x_ids = all_ids[:-1]
        y_ids = all_ids[1:]

        # Create target array with -100 masking for prompt
        y_tensor = list(y_ids)
        mask_len = len(p_ids) - 1
        for i in range(mask_len):
            y_tensor[i] = -100

        # Create token-level weights array
        weights = [0.0] * len(y_ids)
        # Determine character positions of response tokens if multiplier > 1.0
        # By default, all active response tokens have weight 1.0
        for i in range(mask_len, len(y_ids)):
            weights[i] = 1.0

        if relational_multiplier > 1.0 and spans:
            # Map tokens in r_ids to character offsets in s['response']
            curr_char = 0
            raw_resp = s["response"]
            for r_idx, tok_id in enumerate(r_ids):
                tok_str = tok.decode([tok_id], errors="replace")
                # find start position of this piece in raw_resp
                clean_tok_str = tok_str.lstrip()
                if clean_tok_str:
                    match_pos = raw_resp.find(clean_tok_str, curr_char)
                    if match_pos != -1:
                        tok_start = match_pos
                        tok_end = match_pos + len(clean_tok_str)
                        curr_char = tok_end
                        # Check overlap with any target span
                        for sp in spans:
                            sp_start, sp_end = sp[0], sp[1]
                            if not (tok_end <= sp_start or tok_start >= sp_end):
                                # Overlap found!
                                seq_pos = mask_len + r_idx
                                if seq_pos < len(weights):
                                    weights[seq_pos] = relational_multiplier

        batch_x.append(x_ids)
        batch_y.append(y_tensor)
        batch_weights.append(weights)

    # Pad batch to longest sequence in batch
    max_b_len = max(len(seq) for seq in batch_x)
    padded_x = []
    padded_y = []
    padded_w = []

    for seq_x, seq_y, seq_w in zip(batch_x, batch_y, batch_weights):
        pad_len = max_b_len - len(seq_x)
        padded_x.append(seq_x + [0] * pad_len)
        padded_y.append(seq_y + [-100] * pad_len)
        padded_w.append(seq_w + [0.0] * pad_len)

    return (
        torch.tensor(padded_x, dtype=torch.long),
        torch.tensor(padded_y, dtype=torch.long),
        torch.tensor(padded_w, dtype=torch.float32)
    )

def evaluate_validation_loss(model, val_samples, tok, relational_multiplier=1.0, batch_size=8):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    total_weighted_loss = 0.0
    total_weight_sum = 0.0

    with torch.no_grad():
        for i in range(0, len(val_samples), batch_size):
            batch = val_samples[i:i+batch_size]
            x, y, w = prepare_batch_tensors(batch, tok, relational_multiplier=relational_multiplier)
            logits = model(x) # (B, T, V)
            
            flat_logits = logits.view(-1, logits.size(-1))
            flat_y = y.view(-1)
            flat_w = w.view(-1)

            # Filter valid (non-masked) tokens
            valid_mask = flat_y != -100
            if valid_mask.sum() == 0:
                continue

            loss_per_token = criterion(flat_logits[valid_mask], flat_y[valid_mask])
            valid_w = flat_w[valid_mask]

            total_weighted_loss += (loss_per_token * valid_w).sum().item()
            total_weight_sum += valid_w.sum().item()

    model.train()
    if total_weight_sum > 0:
        return total_weighted_loss / total_weight_sum
    return float("nan")

def run_training_arm(config_path_str):
    with open(config_path_str, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    arm_name = cfg["arm"]
    multiplier = cfg["relational_token_multiplier"]
    max_steps = cfg["max_steps"]
    batch_size = cfg["batch_size"]
    lr_initial = cfg["learning_rate"]
    min_lr = cfg["min_learning_rate"]
    weight_decay = cfg["weight_decay"]
    grad_clip = cfg["grad_clip"]
    seed = cfg["seed"]
    eval_every = cfg["eval_every_steps"]

    print(f"\n=======================================================")
    print(f" STARTING TRAINING ARM: {arm_name.upper()}")
    print(f" Target Loss Weight Multiplier: {multiplier}x")
    print(f" Steps: {max_steps} | Batch Size: {batch_size} | LR: {lr_initial}")
    print(f"=======================================================")

    # Set seeds
    torch.manual_seed(seed)

    # Load Tokenizer
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # Load Parent Model
    parent_p = REPO_ROOT / cfg["parent_checkpoint"]
    parent_ckpt = torch.load(parent_p, map_location="cpu", weights_only=False)
    m_cfg = ModelConfig()
    model = ChakrMicro(m_cfg)
    model.load_state_dict(parent_ckpt["model_state_dict"], strict=True)
    model.train()

    # Load Data
    train_samples = []
    with open(REPO_ROOT / cfg["dataset_train"], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_samples.append(json.loads(line))

    val_samples = []
    with open(REPO_ROOT / cfg["dataset_val"], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_samples.append(json.loads(line))

    print(f"Loaded {len(train_samples)} training samples, {len(val_samples)} validation samples.")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_initial, weight_decay=weight_decay)
    
    # Cosine Annealing LR Schedule
    def get_lr(step):
        if step >= max_steps:
            return min_lr
        decay_ratio = step / max_steps
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (lr_initial - min_lr)

    criterion = nn.CrossEntropyLoss(reduction="none")

    # Evaluate initial validation loss
    initial_val_loss = evaluate_validation_loss(model, val_samples, tok, relational_multiplier=multiplier)
    print(f"Step 0 (Pre-training) Validation Loss: {initial_val_loss:.4f}")

    best_val_loss = initial_val_loss
    step = 0
    sample_idx = 0
    training_logs = []
    start_time = time.time()

    arm_dir = REPO_ROOT / "experiments" / "cognition" / "cr_exp_06_variable_binding" / arm_name
    ckpt_dir = arm_dir / "checkpoints"
    log_dir = arm_dir / "logs"

    while step < max_steps:
        # Collect mini-batch
        batch = []
        for _ in range(batch_size):
            batch.append(train_samples[sample_idx % len(train_samples)])
            sample_idx += 1

        curr_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

        optimizer.zero_grad()
        x, y, w = prepare_batch_tensors(batch, tok, relational_multiplier=multiplier)

        logits = model(x)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_y = y.view(-1)
        flat_w = w.view(-1)

        valid_mask = flat_y != -100
        loss_per_token = criterion(flat_logits[valid_mask], flat_y[valid_mask])
        valid_w = flat_w[valid_mask]

        loss = (loss_per_token * valid_w).sum() / valid_w.sum()

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        optimizer.step()

        step += 1
        train_loss_val = loss.item()

        # Logging and evaluation
        if step % eval_every == 0 or step == max_steps:
            val_loss = evaluate_validation_loss(model, val_samples, tok, relational_multiplier=multiplier)
            elapsed = time.time() - start_time
            print(f"Step {step:3d}/{max_steps} | Train Loss: {train_loss_val:.4f} | Val Loss: {val_loss:.4f} | Grad Norm: {grad_norm:.4f} | LR: {curr_lr:.2e} | Elapsed: {elapsed:.1f}s")
            
            log_entry = {
                "step": step,
                "train_loss": round(train_loss_val, 4),
                "val_loss": round(val_loss, 4),
                "grad_norm": round(grad_norm, 4),
                "lr": curr_lr,
                "elapsed_seconds": round(elapsed, 1)
            }
            training_logs.append(log_entry)

            # Save best validation checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = ckpt_dir / f"{arm_name}_best_val.pt"
                torch.save({
                    "step": step,
                    "val_loss": val_loss,
                    "train_loss": train_loss_val,
                    "model_state_dict": model.state_dict(),
                    "config": cfg
                }, best_ckpt_path)
                print(f"  --> Saved new best validation checkpoint to {best_ckpt_path.name} (val_loss: {val_loss:.4f})")

    # Save final checkpoint
    final_ckpt_path = ckpt_dir / f"{arm_name}_final_step_{max_steps}.pt"
    torch.save({
        "step": max_steps,
        "val_loss": val_loss,
        "train_loss": train_loss_val,
        "model_state_dict": model.state_dict(),
        "config": cfg
    }, final_ckpt_path)
    print(f"Saved final checkpoint to {final_ckpt_path}")

    # Write training log
    log_file = log_dir / f"{arm_name}_training_log.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump({
            "arm": arm_name,
            "config": cfg,
            "best_val_loss": round(best_val_loss, 4),
            "final_val_loss": round(val_loss, 4),
            "best_checkpoint_sha256": hash_file(best_ckpt_path) if best_ckpt_path.exists() else None,
            "final_checkpoint_sha256": hash_file(final_ckpt_path) if final_ckpt_path.exists() else None,
            "history": training_logs
        }, f, indent=2)
    print(f"Training log written to {log_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_arm.py <config_path>")
        sys.exit(1)
    run_training_arm(sys.argv[1])

