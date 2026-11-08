"""
CR-EXP-14 Candidate Training Engine
Trains:
- Candidate A: Explicit Subject-Value Conditioning
- Candidate B: Interleaved State-Update Curriculum
- Candidate C: Structured Entity Delimiters

Parent for all: Verified copy of CR-12 Candidate B (SHA-256: 60271c8b...)
Saves checkpoints to: experiments/cognition/cr_exp_14_entity_binding/checkpoints/candidate_{a,b,c}/
Logs loss progression and validation metrics.
"""

import sys
import json
import time
import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
PARENT_CKPT = REPO_ROOT / "experiments" / "cognition" / "cr_exp_14_entity_binding" / "checkpoints" / "baseline_cr12_b" / "cr_exp_12_candidate_b_best_val_copy.pt"
EXP_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_14_entity_binding"

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len=256):
        with open(data_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        self.examples = []
        for it in items:
            text = it["text"] if isinstance(it, dict) else it
            ids = tokenizer.encode(text)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
            self.examples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    pad_id = 0
    max_len = max(x.size(0) for x in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :x.size(0)] = x
    return padded

def evaluate_loss(model, dataloader, device):
    model.eval()
    tot_loss = 0.0
    tot_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        for batch in dataloader:
            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)
            if x.size(1) == 0:
                continue
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            non_pad = (y != 0).sum().item()
            tot_loss += loss.item() * non_pad
            tot_tokens += non_pad
    model.train()
    return tot_loss / max(tot_tokens, 1)

def train_candidate(candidate_name, train_file, val_file, steps=250, lr=2.5e-5, batch_size=8, accum_steps=2):
    print(f"\n=======================================================")
    print(f"Starting Training: {candidate_name.upper()}")
    print(f"Parent Checkpoint: {PARENT_CKPT}")
    print(f"Train File: {train_file}")
    print(f"Steps: {steps}, LR: {lr}, Effective Batch Size: {batch_size * accum_steps}")
    print(f"=======================================================")

    device = torch.device("cpu")
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    train_ds = TextDataset(train_file, tok)
    val_ds = TextDataset(val_file, tok)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    ckpt = torch.load(PARENT_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Initial validation
    init_val_loss = evaluate_loss(model, val_loader, device)
    print(f"[{candidate_name}] Step 0 - Initial Val Loss: {init_val_loss:.4f}")

    out_dir = EXP_DIR / "checkpoints" / candidate_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = init_val_loss
    step = 0
    t0 = time.time()
    train_iter = iter(train_loader)

    history = [{
        "step": 0,
        "train_loss": None,
        "val_loss": round(init_val_loss, 4),
        "lr": lr
    }]

    running_loss = 0.0
    accum_count = 0

    while step < steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        if x.size(1) == 0:
            continue

        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss_scaled = loss / accum_steps
        loss_scaled.backward()

        running_loss += loss.item()
        accum_count += 1

        if accum_count == accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            # Cosine decay with warmup
            cur_step = step + 1
            if cur_step < 20:
                cur_lr = lr * (cur_step / 20)
            else:
                progress = (cur_step - 20) / max(steps - 20, 1)
                cur_lr = lr * 0.1 + 0.5 * lr * 0.9 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            optimizer.step()
            optimizer.zero_grad()
            step += 1
            accum_count = 0

            if step % 25 == 0 or step == steps:
                avg_train = running_loss / (25 * accum_steps if step % 25 == 0 else max(accum_count, 1))
                val_loss = evaluate_loss(model, val_loader, device)
                dt = time.time() - t0
                print(f"[{candidate_name}] Step {step}/{steps} ({dt:.1f}s) - Train Loss: {avg_train:.4f}, Val Loss: {val_loss:.4f}, LR: {cur_lr:.2e}")
                history.append({
                    "step": step,
                    "train_loss": round(avg_train, 4),
                    "val_loss": round(val_loss, 4),
                    "lr": cur_lr
                })
                running_loss = 0.0

                if val_loss < best_val_loss or step == 25:
                    best_val_loss = val_loss
                    best_path = out_dir / f"cr_exp_14_{candidate_name}_best_val.pt"
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "val_loss": val_loss,
                        "candidate": candidate_name
                    }, best_path)
                    print(f"  -> Saved best val checkpoint: {best_path.name} (Val Loss: {val_loss:.4f})")

    # Save final checkpoint
    final_path = out_dir / f"cr_exp_14_{candidate_name}_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "final_val_loss": val_loss,
        "candidate": candidate_name
    }, final_path)

    # Save history
    hist_path = out_dir / f"{candidate_name}_training_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"[{candidate_name}] Training finished. Best Val Loss: {best_val_loss:.4f}")
    return history

def main():
    val_file = EXP_DIR / "data" / "val.json"

    # 1. Candidate A
    file_a = EXP_DIR / "data" / "train_candidate_a.json"
    train_candidate("candidate_a", file_a, val_file, steps=250)

    # 2. Candidate B
    file_b = EXP_DIR / "data" / "train_candidate_b.json"
    train_candidate("candidate_b", file_b, val_file, steps=250)

    # 3. Candidate C
    file_c = EXP_DIR / "data" / "train_candidate_c.json"
    train_candidate("candidate_c", file_c, val_file, steps=250)

    print("\nALL CANDIDATE TRAININGS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
