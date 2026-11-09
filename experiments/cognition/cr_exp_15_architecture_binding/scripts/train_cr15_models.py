"""
CR-EXP-15 Training & Architecture Ablation Engine
Trains:
1. Training Objective Ablation (Baseline Architecture, d_model=192, 6 heads, d_head=32, parent=CR12-B):
   - Candidate A: Standard LM Control
   - Candidate B: Entity Copying & Selection Curriculum
   - Candidate C: Static Binding Curriculum
   - Candidate D: Dynamic State Updates Curriculum

2. Architecture Capacity Ablation (Curriculum D, Matched Steps/Tokens, Matched Init Seed=42):
   - Arch Arm A: Baseline (192, 6 heads, d_head=32) -> Candidate D
   - Arch Arm B: Alternate Head Partition (192, 3 heads, d_head=64)
   - Arch Arm C: Moderate Width (256, 4 heads, d_head=64)

Saves checkpoints and loss histories to checkpoints/.
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
EXP_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_15_architecture_binding"

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
            if x.size(1) == 0: continue
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            non_pad = (y != 0).sum().item()
            tot_loss += loss.item() * non_pad
            tot_tokens += non_pad
    model.train()
    return tot_loss / max(tot_tokens, 1)

def train_run(name, model, train_file, val_file, steps=200, lr=2.5e-5, batch_size=8, accum_steps=2):
    print(f"\n=======================================================")
    print(f"Starting Training: {name}")
    print(f"Steps: {steps}, LR: {lr}, Effective Batch Size: {batch_size * accum_steps}")
    print(f"=======================================================")

    device = torch.device("cpu")
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    train_ds = TextDataset(train_file, tok)
    val_ds = TextDataset(val_file, tok)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    init_val = evaluate_loss(model, val_loader, device)
    print(f"[{name}] Step 0 - Initial Val Loss: {init_val:.4f}")

    out_dir = EXP_DIR / "checkpoints" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = init_val
    step = 0
    t0 = time.time()
    train_iter = iter(train_loader)

    history = [{"step": 0, "train_loss": None, "val_loss": round(init_val, 4), "lr": lr}]
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
        if x.size(1) == 0: continue

        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        (loss / accum_steps).backward()

        running_loss += loss.item()
        accum_count += 1

        if accum_count == accum_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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
                print(f"[{name}] Step {step}/{steps} ({dt:.1f}s) - Train: {avg_train:.4f}, Val: {val_loss:.4f}, LR: {cur_lr:.2e}")
                history.append({
                    "step": step,
                    "train_loss": round(avg_train, 4),
                    "val_loss": round(val_loss, 4),
                    "lr": cur_lr
                })
                running_loss = 0.0

                if val_loss < best_val_loss or step == 25:
                    best_val_loss = val_loss
                    best_path = out_dir / f"{name}_best_val.pt"
                    torch.save({"step": step, "model_state_dict": model.state_dict(), "val_loss": val_loss}, best_path)

    final_path = out_dir / f"{name}_final.pt"
    torch.save({"step": step, "model_state_dict": model.state_dict(), "final_val_loss": val_loss}, final_path)

    with open(out_dir / f"{name}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"[{name}] Completed. Best Val Loss: {best_val_loss:.4f}")
    return history

def load_parent_model():
    cfg = ModelConfig(d_model=192, n_layers=6, n_heads=6, hidden_dim=512)
    m = ChakrMicro(cfg)
    ck = torch.load(PARENT_CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck))
    return m

def main():
    val_file = EXP_DIR / "data" / "val.json"

    # PART 1: Training Objective Ablation (using verified CR-12 Parent)
    # Candidate A: Standard LM Control
    mA = load_parent_model()
    train_run("candidate_a", mA, EXP_DIR / "data" / "train_candidate_a.json", val_file, steps=200)

    # Candidate B: Copy & Select Curriculum
    mB = load_parent_model()
    train_run("candidate_b", mB, EXP_DIR / "data" / "train_candidate_b.json", val_file, steps=200)

    # Candidate C: Static Binding Curriculum
    mC = load_parent_model()
    train_run("candidate_c", mC, EXP_DIR / "data" / "train_candidate_c.json", val_file, steps=200)

    # Candidate D: Dynamic Updates Curriculum (Arch Arm A)
    mD = load_parent_model()
    train_run("candidate_d", mD, EXP_DIR / "data" / "train_candidate_d.json", val_file, steps=200)

    # PART 2: Architecture Capacity Experiments (Matched Seed 42, Matched Train Curriculum D)
    torch.manual_seed(42)

    # Arch Arm B: 192, 6 layers, 3 heads (d_head=64)
    cfg_b = ModelConfig(d_model=192, n_layers=6, n_heads=3, hidden_dim=512)
    m_arch_b = ChakrMicro(cfg_b)
    train_run("arch_arm_b_heads3_d64", m_arch_b, EXP_DIR / "data" / "train_candidate_d.json", val_file, steps=200)

    # Arch Arm C: 256, 6 layers, 4 heads (d_head=64)
    torch.manual_seed(42)
    cfg_c = ModelConfig(d_model=256, n_layers=6, n_heads=4, hidden_dim=682)
    m_arch_c = ChakrMicro(cfg_c)
    train_run("arch_arm_c_width256_d64", m_arch_c, EXP_DIR / "data" / "train_candidate_d.json", val_file, steps=200)

    print("\nALL TRAINING AND ARCHITECTURE EXPERIMENTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
