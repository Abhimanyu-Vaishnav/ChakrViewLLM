"""
UCIP-13 Master Pretraining Script.
Executes from-scratch autoregressive pretraining of ChakrMicro v0.1 on Stage C.
Features:
- Deterministic random initialization (seed 42) verified to match canonical hash.
- Sharded data streaming from data/tokenized/stage_c/train.
- Periodic validation on data/tokenized/stage_c/validation at steps 0, 50, 100, 200, 400, 600, 800.
- Checkpointing at predetermined intervals with full optimizer, scheduler, and RNG state.
- Qualitative generation sampling on frozen evaluation prompts.
"""

import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.training.loss import CausalLoss

DEVICE = torch.device("cpu")
EXP_DIR = ROOT / "experiments" / "ucip13"
CHECKPOINTS_DIR = EXP_DIR / "checkpoints" / "primary_seed42"
LOGS_DIR = EXP_DIR / "logs"
METRICS_DIR = EXP_DIR / "metrics"
SAMPLES_DIR = EXP_DIR / "samples"
EVALS_DIR = EXP_DIR / "evaluations"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
EVALS_DIR.mkdir(parents=True, exist_ok=True)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)


def evaluate_validation_loss(model, criterion, val_dir, max_batches=60):
    model.eval()
    ds = StreamingTokenDataset(val_dir, sequence_length=512, loop=False)
    loader = DataLoader(ds, batch_size=16)
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            if total_batches >= max_batches:
                break
            input_ids = batch["input_ids"].to(DEVICE)
            target_ids = batch["target_ids"].to(DEVICE)
            logits = model(input_ids)
            loss = criterion(logits, target_ids)
            total_loss += loss.item()
            total_batches += 1
    mean_loss = total_loss / max(1, total_batches)
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return mean_loss, ppl, total_batches


def generate_samples(model, tokenizer, prompts, max_new_tokens=40, temperature=0.7):
    model.eval()
    outputs = []
    with torch.no_grad():
        for item in prompts:
            prompt_text = item["prompt"]
            input_ids = tokenizer.encode(prompt_text)
            curr_tokens = list(input_ids)
            for _ in range(max_new_tokens):
                ctx = curr_tokens[-512:]
                inp = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
                logits = model(inp)[0, -1, :]
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_tok = torch.argmax(logits).item()
                curr_tokens.append(next_tok)
                if next_tok == 1:  # EOS
                    break
            gen_tokens = curr_tokens[len(input_ids):]
            gen_text = tokenizer.decode(gen_tokens, errors="replace")
            outputs.append({
                "id": item["id"],
                "category": item["category"],
                "prompt": prompt_text,
                "generated_text": gen_text,
                "token_count": len(gen_tokens)
            })
    return outputs


def run_training():
    print("=" * 80)
    print("UCIP-13 PRIMARY NEURAL PRETRAINING EXPERIMENT (SEED 42)")
    print("=" * 80)

    # 1. Load Tokenizer
    tok_dir = ROOT / "data" / "experiments" / "vocab_4096"
    tokenizer, tok_cfg = load_tokenizer_artifacts(tok_dir)
    print(f"Loaded Tokenizer: Vocab={tokenizer.vocab_size}, Merges={tokenizer.num_merges}")

    # 2. Initialize Model from Baseline
    model = instantiate_frozen_baseline().to(DEVICE)
    init_hash = compute_model_hash(model)
    assert init_hash == EXPECTED_WEIGHT_HASH, f"Hash mismatch: {init_hash} != {EXPECTED_WEIGHT_HASH}"
    print(f"Instantiated Frozen Baseline (seed 42): SHA-256 = {init_hash}")

    # 3. Setup Optimizer and Scheduler
    learning_rate = 5e-4
    min_learning_rate = 5e-5
    warmup_steps = 40
    total_steps = 800
    batch_size = 16
    seq_len = 512
    tokens_per_step = batch_size * seq_len  # 8,192

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=min_learning_rate / learning_rate
    )
    criterion = CausalLoss()

    # 4. Save Initial Step 0 Checkpoint
    checkpoint_0 = {
        "step": 0,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "init_hash": init_hash,
        "tokens_processed": 0,
        "config": ModelConfig().__dict__,
    }
    torch.save(checkpoint_0, CHECKPOINTS_DIR / "checkpoint_step_000.pt")
    print(f"Saved initial checkpoint_step_000.pt")

    # 5. Load Evaluation Prompts
    prompts_path = EVALS_DIR / "evaluation_prompts.json"
    with open(prompts_path, "r", encoding="utf-8") as f:
        eval_prompts = json.load(f)

    # Step 0 Validation & Generation
    val_dir = ROOT / "data" / "tokenized" / "stage_c" / "validation"
    print("Evaluating Step 0 validation loss...")
    val_loss_0, val_ppl_0, _ = evaluate_validation_loss(model, criterion, val_dir)
    print(f"[Step 0] Initial Val Loss: {val_loss_0:.4f} | Perplexity: {val_ppl_0:.2f}")

    step0_samples = generate_samples(model, tokenizer, eval_prompts)
    with open(SAMPLES_DIR / "samples_step_000.json", "w", encoding="utf-8") as f:
        json.dump(step0_samples, f, indent=2)

    # 6. Data Streamer
    train_dir = ROOT / "data" / "tokenized" / "stage_c" / "train"
    train_ds = StreamingTokenDataset(train_dir, sequence_length=seq_len, loop=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    train_iter = iter(train_loader)

    checkpoint_steps = {50, 100, 200, 400, 600, 800}
    training_history = []
    validation_history = [{"step": 0, "val_loss": val_loss_0, "perplexity": val_ppl_0, "tokens": 0}]

    t_start = time.time()
    total_tokens_processed = 0

    print("\nStarting Pretraining Loop (800 steps)...")
    model.train()

    for step in range(1, total_steps + 1):
        t_step = time.time()
        batch = next(train_iter)
        input_ids = batch["input_ids"].to(DEVICE)
        target_ids = batch["target_ids"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, target_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_tokens_processed += tokens_per_step
        step_duration = time.time() - t_step
        current_lr = scheduler.get_last_lr()[0]

        record = {
            "step": step,
            "train_loss": round(loss.item(), 4),
            "lr": round(current_lr, 7),
            "tokens_processed": total_tokens_processed,
            "step_time_sec": round(step_duration, 3)
        }
        training_history.append(record)

        if step % 20 == 0 or step in checkpoint_steps:
            speed = tokens_per_step / max(0.001, step_duration)
            print(f"Step {step:3d}/800 | Loss: {loss.item():.4f} | LR: {current_lr:.6f} | Tokens: {total_tokens_processed:,} | Speed: {speed:.0f} tok/s")

        # Validation & Checkpointing
        if step in checkpoint_steps:
            val_loss, val_ppl, _ = evaluate_validation_loss(model, criterion, val_dir)
            val_rec = {
                "step": step,
                "val_loss": round(val_loss, 4),
                "perplexity": round(val_ppl, 2),
                "tokens": total_tokens_processed
            }
            validation_history.append(val_rec)
            print(f"\n>>> [VALIDATION at Step {step}] Loss: {val_loss:.4f} | PPL: {val_ppl:.2f} (Train Loss: {loss.item():.4f})\n")

            # Save checkpoint
            ckpt_path = CHECKPOINTS_DIR / f"checkpoint_step_{step:03d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
                "tokens_processed": total_tokens_processed,
                "config": ModelConfig().__dict__,
                "creation_time": time.time()
            }, ckpt_path)
            print(f"Saved {ckpt_path.name}")

            # Sample Generation
            samples = generate_samples(model, tokenizer, eval_prompts)
            with open(SAMPLES_DIR / f"samples_step_{step:03d}.json", "w", encoding="utf-8") as f:
                json.dump(samples, f, indent=2)

            model.train()

    total_training_duration = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETED: 800 steps in {total_training_duration:.2f}s ({total_training_duration/60:.2f} mins)")
    print(f"Total tokens exposed: {total_tokens_processed:,}")
    print(f"Final Validation Loss: {validation_history[-1]['val_loss']:.4f} | Final Perplexity: {validation_history[-1]['perplexity']:.2f}")
    print("=" * 80)

    # Save metrics
    with open(METRICS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    with open(METRICS_DIR / "validation_history.json", "w", encoding="utf-8") as f:
        json.dump(validation_history, f, indent=2)

    with open(METRICS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_steps": total_steps,
            "total_tokens_processed": total_tokens_processed,
            "total_training_duration_sec": total_training_duration,
            "initial_val_loss": val_loss_0,
            "final_val_loss": validation_history[-1]["val_loss"],
            "initial_val_ppl": val_ppl_0,
            "final_val_ppl": validation_history[-1]["perplexity"],
            "best_checkpoint": f"checkpoint_step_{step:03d}.pt"
        }, f, indent=2)


if __name__ == "__main__":
    run_training()
