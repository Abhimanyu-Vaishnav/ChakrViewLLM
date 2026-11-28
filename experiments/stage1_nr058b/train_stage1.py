"""
Mission NR-058B Stage 1: Controlled Pretraining Engine (1 Complete Epoch = 1,305 Steps)
Models: ChakrGenesis17M & ChakrGenesis38M
Corpus: Corpus V6 Train (data/corpus_v6/splits/corpus_v6_train.jsonl)
Validation: Corpus V6 Val (data/corpus_v6/splits/corpus_v6_val.jsonl)
Tokenizer: Canonical Byte-Level BPE 4096 (data/experiments/vocab_4096)
Sequence Length: 256
Batch Size: 16
Step Budget: Exactly 1,305 steps per model (1 epoch)
Learning Rate Schedule:
  - 17M: Peak LR 3e-4, 100 warmup steps, cosine decay to 3e-5 (10%)
  - 38M: Peak LR 2.5e-4, 100 warmup steps, cosine decay to 2.5e-5 (10%)
Checkpoints: Periodic checkpoints saved to experiments/stage1_nr058b/
Isolation: Strictly isolated to experiments/stage1_nr058b/
"""

import sys
import os
import json
import time
import math
import hashlib
import random
import psutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.brain.genesis.chakr_genesis_38m import ChakrGenesis38M, Genesis38MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

class CorpusV6PretrainDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=256, max_samples=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                tokens = tokenizer.encode(text)
                if len(tokens) == 0:
                    continue
                tokens.append(2)  # EOS
                for i in range(0, len(tokens), max_len):
                    chunk = tokens[i:i + max_len]
                    if len(chunk) >= 16:
                        self.samples.append(chunk)
                        if max_samples and len(self.samples) >= max_samples:
                            break
                if max_samples and len(self.samples) >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        if len(chunk) < self.max_len:
            pad_len = self.max_len - len(chunk)
            input_ids = chunk + [0] * pad_len
        else:
            input_ids = chunk[:self.max_len]
        return torch.tensor(input_ids, dtype=torch.long)

def evaluate(model, val_loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    batches = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            inp = batch[:, :-1].contiguous()
            tgt = batch[:, 1:].contiguous()
            mask = (tgt != 0)
            tgt_loss = tgt.clone()
            tgt_loss[~mask] = -100

            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(inp)
                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                loss = criterion(logits.reshape(-1, 4096), tgt_loss.reshape(-1))

            n_active = mask.sum().item()
            total_loss += loss.item() * n_active
            total_tokens += n_active

            preds = logits.argmax(dim=-1)
            correct_tokens += ((preds == tgt) & mask).sum().item()
            batches += 1
            if max_batches and batches >= max_batches:
                break

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, avg_loss))
    acc = (correct_tokens / max(1, total_tokens)) * 100.0
    model.train()
    return avg_loss, ppl, acc

def run_stage1_pretraining(model_name: str, model_cls, config, batch_size: int = 16, max_steps: int = 1305, peak_lr: float = 3e-4, warmup_steps: int = 100, seed: int = 42, checkpoint_every: int = 200):
    print(f"\n=======================================================", flush=True)
    print(f"STARTING NR-058B STAGE 1 PRETRAINING: {model_name}", flush=True)
    print(f"TARGET: Exactly 1 Epoch ({max_steps} Steps) | Batch Size: {batch_size} | Peak LR: {peak_lr:.2e}", flush=True)
    print(f"=======================================================", flush=True)

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("experiments/stage1_nr058b")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_dir = "data/experiments/vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    print("Building datasets from Corpus V6...", flush=True)
    train_ds = CorpusV6PretrainDataset("data/corpus_v6/splits/corpus_v6_train.jsonl", tokenizer, max_len=256)
    val_ds = CorpusV6PretrainDataset("data/corpus_v6/splits/corpus_v6_val.jsonl", tokenizer, max_len=256)

    # Use fixed generator seed for reproducibility
    g_train = torch.Generator()
    g_train.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train dataset chunks: {len(train_ds):,}, Val dataset chunks: {len(val_ds):,}", flush=True)

    # Deterministic fresh initialization from seed
    model = model_cls(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"{model_name} total unique parameters: {param_count:,}", flush=True)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)

    min_lr_ratio = 0.1  # Decay to 10% of peak lr
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')

    # Initial Pre-training Validation
    print("Evaluating pre-training baseline on validation set...", flush=True)
    val_loss_init, val_ppl_init, val_acc_init = evaluate(model, val_loader, criterion, device, max_batches=50)
    print(f"Baseline Val Loss: {val_loss_init:.4f} | PPL: {val_ppl_init:.2f} | Acc: {val_acc_init:.2f}%", flush=True)

    history = []
    checkpoint_records = []
    step = 0
    tokens_seen = 0
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process()

    model.train()
    loader_iter = iter(train_loader)

    while step < max_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        t_step_start = time.perf_counter()
        step += 1
        batch = batch.to(device)
        inp = batch[:, :-1].contiguous()
        tgt = batch[:, 1:].contiguous()
        mask = (tgt != 0)
        tgt_loss = tgt.clone()
        tgt_loss[~mask] = -100

        n_tokens = mask.sum().item()
        tokens_seen += n_tokens

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(inp)
            logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
            loss = criterion(logits.reshape(-1, 4096), tgt_loss.reshape(-1))

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            raise RuntimeError(f"Step {step}: Non-finite loss encountered ({loss_val})")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        t_step_end = time.perf_counter()
        step_ms = (t_step_end - t_step_start) * 1000
        current_lr = scheduler.get_last_lr()[0]

        step_rec = {
            "step": step,
            "train_loss": round(loss_val, 4),
            "grad_norm": round(grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm), 4),
            "lr": current_lr,
            "step_ms": round(step_ms, 2),
            "tokens_processed": n_tokens,
            "cumulative_tokens": tokens_seen
        }
        history.append(step_rec)

        if step % 50 == 0 or step == max_steps:
            print(f"Step {step:04d}/{max_steps} | Loss: {loss_val:.4f} | Grad: {step_rec['grad_norm']:.4f} | LR: {current_lr:.2e} | Latency: {step_ms:.1f}ms", flush=True)

        # Periodic checkpointing and validation evaluation
        if step % checkpoint_every == 0 or step == max_steps:
            val_loss_curr, val_ppl_curr, val_acc_curr = evaluate(model, val_loader, criterion, device, max_batches=50)
            ckpt_path = out_dir / f"chakr_{model_name.lower()}_stage1_step{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config,
                "tokens_seen": tokens_seen,
                "train_loss": loss_val,
                "val_loss": val_loss_curr,
                "val_ppl": val_ppl_curr,
                "val_acc": val_acc_curr
            }, ckpt_path)

            ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
            with open(ckpt_path, "rb") as f:
                ckpt_sha = hashlib.sha256(f.read()).hexdigest()

            ckpt_info = {
                "step": step,
                "val_loss": round(val_loss_curr, 4),
                "val_ppl": round(val_ppl_curr, 2),
                "val_acc": round(val_acc_curr, 2),
                "checkpoint_path": str(ckpt_path),
                "size_mb": round(ckpt_size_mb, 2),
                "sha256": ckpt_sha
            }
            checkpoint_records.append(ckpt_info)
            print(f"--> [Checkpoint Step {step:04d}] Val Loss: {val_loss_curr:.4f} | PPL: {val_ppl_curr:.2f} | Acc: {val_acc_curr:.2f}% | SHA: {ckpt_sha[:10]}...", flush=True)

    elapsed_time = time.time() - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    sys_ram_gb = process.memory_info().rss / (1024 ** 3)

    # Full Final Validation over all batches
    print("\nRunning comprehensive final validation evaluation (all batches)...", flush=True)
    val_loss_final, val_ppl_final, val_acc_final = evaluate(model, val_loader, criterion, device, max_batches=None)
    print(f"Final Val Loss: {val_loss_final:.4f} | PPL: {val_ppl_final:.2f} | Acc: {val_acc_final:.2f}%", flush=True)

    # Final Stage 1 Checkpoint
    final_ckpt_path = out_dir / f"chakr_{model_name.lower()}_stage1_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config,
        "tokens_seen": tokens_seen,
        "train_loss": history[-1]["train_loss"],
        "val_loss": val_loss_final,
        "val_ppl": val_ppl_final,
        "val_acc": val_acc_final
    }, final_ckpt_path)

    final_ckpt_size_mb = os.path.getsize(final_ckpt_path) / (1024 * 1024)
    with open(final_ckpt_path, "rb") as f:
        final_ckpt_sha = hashlib.sha256(f.read()).hexdigest()

    result = {
        "mission": "NR-058B Stage 1 Controlled Pretraining",
        "model_name": model_name,
        "parameters": param_count,
        "batch_size": batch_size,
        "seq_len": 256,
        "epochs_executed": 1.0,
        "total_steps_executed": step,
        "tokens_seen": tokens_seen,
        "elapsed_seconds": round(elapsed_time, 2),
        "tokens_per_sec": round(tokens_seen / elapsed_time, 1),
        "steps_per_sec": round(step / elapsed_time, 2),
        "peak_learning_rate": peak_lr,
        "final_learning_rate": scheduler.get_last_lr()[0],
        "initial_train_loss": history[0]["train_loss"],
        "final_train_loss": history[-1]["train_loss"],
        "initial_val_loss": round(val_loss_init, 4),
        "final_val_loss": round(val_loss_final, 4),
        "initial_val_ppl": round(val_ppl_init, 2),
        "final_val_ppl": round(val_ppl_final, 2),
        "final_val_acc_pct": round(val_acc_final, 2),
        "loss_delta": round(val_loss_final - val_loss_init, 4),
        "ppl_delta": round(val_ppl_final - val_ppl_init, 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "system_ram_gb": round(sys_ram_gb, 2),
        "checkpoint_records": checkpoint_records,
        "final_checkpoint_path": str(final_ckpt_path),
        "final_checkpoint_size_mb": round(final_ckpt_size_mb, 2),
        "final_checkpoint_sha256": final_ckpt_sha,
        "step_history": history
    }

    out_json = out_dir / f"{model_name.lower()}_stage1_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n=======================================================")
    print(f"STAGE 1 COMPLETE: {model_name}")
    print(f"Elapsed: {elapsed_time:.1f}s | Final Val Loss: {val_loss_final:.4f} | PPL: {val_ppl_final:.2f} | Acc: {val_acc_final:.2f}%")
    print(f"Final Checkpoint: {final_ckpt_path} (SHA: {final_ckpt_sha})")
    print(f"Summary JSON: {out_json}")
    print(f"=======================================================\n")
    return result

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        print("=== RUNNING NR-058B STAGE 1 SMOKE TEST (5 STEPS, BS=16) ===")
        cfg17 = Genesis17MConfig()
        run_stage1_pretraining("ChakrGenesis17M_Smoke", ChakrGenesis17M, cfg17, batch_size=16, max_steps=5, peak_lr=3e-4, warmup_steps=2, checkpoint_every=5)
        cfg38 = Genesis38MConfig()
        run_stage1_pretraining("ChakrGenesis38M_Smoke", ChakrGenesis38M, cfg38, batch_size=16, max_steps=5, peak_lr=2.5e-4, warmup_steps=2, checkpoint_every=5)
        print("SMOKE TESTS PASSED.")
    elif mode == "17m":
        cfg17 = Genesis17MConfig()
        run_stage1_pretraining("ChakrGenesis17M", ChakrGenesis17M, cfg17, batch_size=16, max_steps=1305, peak_lr=3e-4, warmup_steps=100, checkpoint_every=200)
    elif mode == "38m":
        cfg38 = Genesis38MConfig()
        run_stage1_pretraining("ChakrGenesis38M", ChakrGenesis38M, cfg38, batch_size=16, max_steps=1305, peak_lr=2.5e-4, warmup_steps=100, checkpoint_every=200)
    else:
        print(f"Unknown mode: {mode}")
