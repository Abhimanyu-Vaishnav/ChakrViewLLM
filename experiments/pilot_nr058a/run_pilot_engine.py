"""
Mission NR-058A: Controlled Training Pilot Engine.
Authorizes EXACTLY up to 100 optimizer steps per model (17M and 38M).
Uses:
- Corpus V6 train split (data/corpus_v6/splits/corpus_v6_train.jsonl)
- Corpus V6 val split (data/corpus_v6/splits/corpus_v6_val.jsonl)
- Canonical Byte-Level BPE 4096 tokenizer
- Sequence length = 256 (matches canonical Genesis pretraining dataset chunking)
- AdamW optimizer (betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
- Cosine annealing scheduler with warmup
- AMP fp16 autocast and gradient scaler
- Strict safety limits (max 100 steps per model)
- Saves outputs exclusively to experiments/pilot_nr058a/
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

def evaluate(model, val_loader, criterion, device, max_batches=50):
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
    return avg_loss, ppl, acc

def run_pilot(model_name: str, model_cls, config, batch_size: int = 16, max_steps: int = 100, lr: float = 3e-4, seed: int = 42):
    print(f"\n=======================================================", flush=True)
    print(f"STARTING NR-058A PILOT: {model_name} (LIMIT: {max_steps} STEPS)", flush=True)
    print(f"=======================================================", flush=True)

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("experiments/pilot_nr058a")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_dir = "data/experiments/vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    print("Building datasets from Corpus V6...", flush=True)
    train_ds = CorpusV6PretrainDataset("data/corpus_v6/splits/corpus_v6_train.jsonl", tokenizer, max_len=256)
    val_ds = CorpusV6PretrainDataset("data/corpus_v6/splits/corpus_v6_val.jsonl", tokenizer, max_len=256, max_samples=800)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train dataset chunks: {len(train_ds):,}, Val dataset chunks: {len(val_ds):,}", flush=True)

    # Initialize model fresh from seed (or base if specified)
    model = model_cls(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"{model_name} total unique parameters: {param_count:,}", flush=True)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)

    warmup_steps = 10
    total_steps = max_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')

    # Initial Validation
    val_loss_init, val_ppl_init, val_acc_init = evaluate(model, val_loader, criterion, device, max_batches=30)
    print(f"Pre-training Val Loss: {val_loss_init:.4f} | PPL: {val_ppl_init:.2f} | Acc: {val_acc_init:.2f}%", flush=True)

    history = []
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

        # Check finite loss
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

        if step % 20 == 0 or step == max_steps:
            print(f"Step {step:03d}/{max_steps} | Loss: {loss_val:.4f} | Grad: {step_rec['grad_norm']:.4f} | LR: {current_lr:.2e} | Latency: {step_ms:.1f}ms", flush=True)

    elapsed_time = time.time() - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    sys_ram_gb = process.memory_info().rss / (1024 ** 3)

    # Final Validation
    print("Running final validation evaluation...", flush=True)
    val_loss_final, val_ppl_final, val_acc_final = evaluate(model, val_loader, criterion, device, max_batches=50)
    print(f"Final Val Loss: {val_loss_final:.4f} | PPL: {val_ppl_final:.2f} | Acc: {val_acc_final:.2f}%", flush=True)

    # Save Checkpoint
    ckpt_path = out_dir / f"chakr_{model_name.lower()}_pilot_100step.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "config": config,
        "tokens_seen": tokens_seen,
        "val_loss": val_loss_final,
        "val_ppl": val_ppl_final
    }, ckpt_path)

    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    with open(ckpt_path, "rb") as f:
        ckpt_sha = hashlib.sha256(f.read()).hexdigest()

    result = {
        "model_name": model_name,
        "parameters": param_count,
        "batch_size": batch_size,
        "seq_len": 256,
        "total_steps_executed": step,
        "tokens_seen": tokens_seen,
        "elapsed_seconds": round(elapsed_time, 2),
        "tokens_per_sec": round(tokens_seen / elapsed_time, 1),
        "initial_train_loss": history[0]["train_loss"],
        "final_train_loss": history[-1]["train_loss"],
        "initial_val_loss": round(val_loss_init, 4),
        "final_val_loss": round(val_loss_final, 4),
        "initial_val_ppl": round(val_ppl_init, 2),
        "final_val_ppl": round(val_ppl_final, 2),
        "final_val_acc_pct": round(val_acc_final, 2),
        "loss_delta": round(val_loss_final - val_loss_init, 4),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "system_ram_gb": round(sys_ram_gb, 2),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
        "checkpoint_sha256": ckpt_sha,
        "history": history
    }

    out_json = out_dir / f"{model_name.lower()}_pilot_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Pilot {model_name} completed in {elapsed_time:.1f}s. Summary: {out_json}")
    return result

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        print("=== RUNNING NR-058A SMOKE TEST (5 STEPS) ===")
        cfg17 = Genesis17MConfig()
        res = run_pilot("ChakrGenesis17M_Smoke", ChakrGenesis17M, cfg17, batch_size=8, max_steps=5)
        print("SMOKE TEST PASSED.")
    elif mode == "17m":
        cfg17 = Genesis17MConfig()
        run_pilot("ChakrGenesis17M", ChakrGenesis17M, cfg17, batch_size=16, max_steps=100)
    elif mode == "38m":
        cfg38 = Genesis38MConfig()
        run_pilot("ChakrGenesis38M", ChakrGenesis38M, cfg38, batch_size=16, max_steps=100)
