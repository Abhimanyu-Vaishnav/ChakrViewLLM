"""
Mission CV-24H: Conversational SFT Engine for ChakrGenesis38M.
Performs response-only supervised fine-tuning:
- Takes the pretrained base: experiments/stage1_nr058b/chakr_chakrgenesis38m_stage1_final.pt
- Loads canonical 4096 Byte-Level BPE tokenizer
- Formats: Prompt + " " + Response + EOS (token 2)
- Target loss masking: user prompt tokens are set to -100 so loss is computed EXCLUSIVELY on assistant response
- Uses conservative learning rate (e.g., 5e-5) with warmup and cosine decay
- Evaluates validation loss, response accuracy, and perplexity
- Saves isolated SFT checkpoint: experiments/cv24h_release/chakr_38m_conversational_sft.pt
"""

import os
import sys
import json
import time
import math
import hashlib
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))
from chakrview.brain.genesis.chakr_genesis_38m import ChakrGenesis38M, Genesis38MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

class ConversationalSFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                prompt = obj["prompt"]
                response = obj["response"]

                p_tokens = tokenizer.encode(prompt)
                r_tokens = tokenizer.encode(" " + response) + [2]  # EOS token

                # If total tokens exceeds max_len, truncate prompt from left or response from right
                if len(p_tokens) + len(r_tokens) > max_len:
                    # Keep prompt concise, give response at least 64 tokens
                    avail_p = max_len - len(r_tokens)
                    if avail_p > 10:
                        p_tokens = p_tokens[-avail_p:]
                    else:
                        p_tokens = p_tokens[:30]
                        r_tokens = r_tokens[:(max_len - len(p_tokens) - 1)] + [2]

                input_ids = p_tokens + r_tokens
                # Mask prompt: labels = -100 for prompt, response tokens for response
                labels = [-100] * len(p_tokens) + r_tokens

                pad_len = max_len - len(input_ids)
                if pad_len > 0:
                    input_ids = input_ids + [0] * pad_len
                    labels = labels + [-100] * pad_len
                else:
                    input_ids = input_ids[:max_len]
                    labels = labels[:max_len]

                self.samples.append((
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def evaluate_sft(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            inp = input_ids[:, :-1].contiguous()
            tgt = labels[:, 1:].contiguous()
            mask = (tgt != -100)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(inp)
                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                loss = criterion(logits.reshape(-1, 4096), tgt.reshape(-1))

            n_active = mask.sum().item()
            if n_active > 0:
                total_loss += loss.item() * n_active
                total_tokens += n_active
                preds = logits.argmax(dim=-1)
                correct_tokens += ((preds == tgt) & mask).sum().item()

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(20.0, avg_loss))
    acc = (correct_tokens / max(1, total_tokens)) * 100.0
    model.train()
    return avg_loss, ppl, acc

def train_sft(base_ckpt_path: str, epochs: int = 3, batch_size: int = 16, lr: float = 5e-5, seed: int = 42):
    print("=======================================================")
    print(f"STARTING CONVERSATIONAL SFT: ChakrGenesis38M ({epochs} Epochs)")
    print(f"Base Pretrained Checkpoint: {base_ckpt_path}")
    print("=======================================================")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("experiments/cv24h_release")
    out_dir.mkdir(parents=True, exist_ok=True)

    tok_dir = "data/experiments/vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    train_ds = ConversationalSFTDataset("experiments/cv24h_release/sft_mixture_train.jsonl", tokenizer, max_len=256)
    val_ds = ConversationalSFTDataset("experiments/cv24h_release/sft_mixture_val.jsonl", tokenizer, max_len=256)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Loaded SFT Train: {len(train_ds)} samples, SFT Val: {len(val_ds)} samples")

    print("Loading pretrained base model...")
    ckpt = torch.load(base_ckpt_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = ChakrGenesis38M(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    param_count = sum(p.numel() for p in model.parameters())
    print(f"ChakrGenesis38M parameters loaded: {param_count:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')

    # Initial zero-shot evaluation on SFT validation set
    val_loss_init, val_ppl_init, val_acc_init = evaluate_sft(model, val_loader, criterion, device)
    print(f"Pre-SFT Baseline Val Loss: {val_loss_init:.4f} | PPL: {val_ppl_init:.2f} | Acc: {val_acc_init:.2f}%")

    step = 0
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats(device)

    model.train()
    history = []

    for ep in range(epochs):
        for input_ids, labels in train_loader:
            step += 1
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            inp = input_ids[:, :-1].contiguous()
            tgt = labels[:, 1:].contiguous()

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(inp)
                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                loss = criterion(logits.reshape(-1, 4096), tgt.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            cur_lr = scheduler.get_last_lr()[0]
            loss_val = loss.item()

            if step % 20 == 0 or step == total_steps:
                print(f"Epoch {ep+1}/{epochs} | Step {step:03d}/{total_steps} | Loss: {loss_val:.4f} | Grad: {grad_norm.item():.4f} | LR: {cur_lr:.2e}")

        # End of epoch val
        val_l, val_p, val_a = evaluate_sft(model, val_loader, criterion, device)
        print(f"--> [Epoch {ep+1} Complete] SFT Val Loss: {val_l:.4f} | PPL: {val_p:.2f} | Acc: {val_a:.2f}%")
        history.append({"epoch": ep+1, "val_loss": round(val_l, 4), "val_ppl": round(val_p, 2), "val_acc": round(val_a, 2)})

    elapsed_time = time.time() - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # Save isolated SFT checkpoint
    sft_ckpt_path = out_dir / "chakr_38m_conversational_sft.pt"
    torch.save({
        "mission": "CV-24H Conversational SFT",
        "model_name": "ChakrGenesis38M-Conversational-Alpha",
        "epochs": epochs,
        "steps": step,
        "model_state_dict": model.state_dict(),
        "config": config,
        "final_val_loss": history[-1]["val_loss"],
        "final_val_ppl": history[-1]["val_ppl"],
        "final_val_acc": history[-1]["val_acc"]
    }, sft_ckpt_path)

    with open(sft_ckpt_path, "rb") as f:
        sft_sha = hashlib.sha256(f.read()).hexdigest()

    summary = {
        "mission": "CV-24H Conversational SFT",
        "model_architecture": "ChakrGenesis38M",
        "parameters": param_count,
        "base_pretrained_checkpoint": base_ckpt_path,
        "sft_checkpoint_path": str(sft_ckpt_path),
        "sft_checkpoint_sha256": sft_sha,
        "epochs": epochs,
        "total_steps": step,
        "batch_size": batch_size,
        "learning_rate": lr,
        "initial_val_loss": round(val_loss_init, 4),
        "final_val_loss": history[-1]["val_loss"],
        "initial_val_ppl": round(val_ppl_init, 2),
        "final_val_ppl": history[-1]["val_ppl"],
        "initial_val_acc": round(val_acc_init, 2),
        "final_val_acc": history[-1]["val_acc"],
        "elapsed_seconds": round(elapsed_time, 2),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "history": history
    }

    sum_path = out_dir / "sft_training_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=======================================================")
    print(f"CONVERSATIONAL SFT COMPLETE in {elapsed_time:.1f}s")
    print(f"Final Val Loss: {history[-1]['val_loss']} (from {val_loss_init:.4f})")
    print(f"Final Val Acc: {history[-1]['val_acc']}% (from {val_acc_init:.2f}%)")
    print(f"Checkpoint: {sft_ckpt_path} (SHA: {sft_sha})")
    print("=======================================================\n")
    return summary

if __name__ == "__main__":
    base_ckpt = "experiments/stage1_nr058b/chakr_chakrgenesis38m_stage1_final.pt"
    train_sft(base_ckpt, epochs=3, batch_size=16, lr=5e-5)
