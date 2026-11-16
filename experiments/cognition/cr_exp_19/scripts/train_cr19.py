import os
import sys
import math
import json
import time
import argparse
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.conversation.cr_exp_17.conversation_formatter import ConversationFormatter
from experiments.conversation.cr_exp_17.conversation_loss import ConversationLoss

TOKENIZER_DIR = REPO_ROOT / "data/experiments/vocab_4096"
tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

PARENT_CKPT_PATH = REPO_ROOT / "releases/v0.9.1_conversational_preview/chakrview_v091_model.pt"

formatter = ConversationFormatter(tokenizer, max_seq_len=512)
loss_fn = ConversationLoss(ignore_index=-100)

class ConversationalDataset(Dataset):
    def __init__(self, jsonl_path):
        self.samples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                item = json.loads(line)
                p = item["prompt"]
                r = item["response"]
                inp_ids, labels = formatter.prepare_training_pair(p, r)
                self.samples.append((inp_ids, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    max_len = max(len(inp) for inp, _ in batch)
    input_ids = []
    labels = []
    for inp, lab in batch:
        pad_len = max_len - len(inp)
        input_ids.append(inp + [0] * pad_len)
        labels.append(lab + [-100] * pad_len)
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def train_arm(arm_name, train_file, seed, total_steps=300):
    print(f"\n=======================================================")
    print(f"  TRAINING {arm_name.upper()} | SEED {seed}")
    print(f"=======================================================")
    torch.manual_seed(seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    
    # 1. Instantiate Canonical Model Architecture
    cfg = ModelConfig(
        vocab_size=4096,
        d_model=192,
        n_layers=6,
        n_heads=6,
        hidden_dim=512,
        max_seq_len=512
    )
    model = ChakrMicro(cfg)
    
    # 2. Load Parent Checkpoint weights strictly
    torch.serialization.add_safe_globals([ModelConfig])
    payload = torch.load(str(PARENT_CKPT_PATH), map_location="cpu", weights_only=True)
    state_dict = payload.get("model_state_dict", payload)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert len(missing) == 0 and len(unexpected) == 0, "Strict parent load failed!"
    print(f"Loaded parent checkpoint strictly ({sum(p.numel() for p in model.parameters())} params)")
    
    model.to(device)
    
    # 3. Data Loaders
    train_ds = ConversationalDataset(train_file)
    val_ds = ConversationalDataset(REPO_ROOT / "experiments/cognition/cr_exp_19/data/extended_validation.jsonl")
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # 4. Optimizer and Cosine Schedule
    lr_max = 6e-5
    lr_min = 6e-6
    warmup_steps = 20
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95), weight_decay=0.01)
    
    def get_lr(it):
        if it < warmup_steps:
            return lr_max * (it + 1) / warmup_steps
        if it > total_steps:
            return lr_min
        decay_ratio = (it - warmup_steps) / (total_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return lr_min + coeff * (lr_max - lr_min)

    # 5. Evaluate Initial Validation Loss
    def eval_val():
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        batches = 0
        with torch.no_grad():
            for inps, labs in val_loader:
                inps, labs = inps.to(device), labs.to(device)
                logits = model(inps)
                loss, acc = loss_fn(logits, labs)
                total_loss += loss.item()
                total_acc += acc.item()
                batches += 1
        return total_loss / max(batches, 1), total_acc / max(batches, 1)

    init_loss, init_acc = eval_val()
    print(f"Initial Val Loss: {init_loss:.4f} | Val Token Acc: {init_acc*100:.2f}%")
    
    # 6. Training Loop (Effective batch size = 8 via grad accum = 2)
    step = 0
    grad_accum_steps = 2
    best_val_loss = float("inf")
    best_step = 0
    
    out_ckpt_dir = REPO_ROOT / f"experiments/cognition/cr_exp_19/checkpoints/{arm_name}"
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_ckpt_dir / f"{arm_name}_seed{seed}_best.pt"
    
    model.train()
    data_iter = iter(train_loader)
    
    start_time = time.time()
    
    while step < total_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for micro_step in range(grad_accum_steps):
            try:
                inps, labs = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                inps, labs = next(data_iter)
                
            inps, labs = inps.to(device), labs.to(device)
            logits = model(inps)
            loss, acc = loss_fn(logits, labs)
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
            accum_loss += loss.item()
            
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        
        if step % 50 == 0 or step == total_steps:
            v_loss, v_acc = eval_val()
            model.train()
            print(f"Step {step:3d}/{total_steps} | Train Loss: {accum_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc*100:.2f}% | LR: {lr:.2e}")
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_step = step
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_loss": v_loss,
                    "val_acc": v_acc,
                    "arm": arm_name,
                    "seed": seed,
                    "parent_checkpoint": str(PARENT_CKPT_PATH),
                    "parent_sha256": "f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10"
                }, best_ckpt_path)
                
    elapsed = time.time() - start_time
    h_best = hashlib.sha256(best_ckpt_path.read_bytes()).hexdigest()
    print(f"Completed {arm_name} seed {seed} in {elapsed:.1f}s. Best Step: {best_step} (Val Loss: {best_val_loss:.4f})")
    print(f"Saved Checkpoint: {best_ckpt_path.name} | SHA-256: {h_best}")
    return {
        "arm": arm_name,
        "seed": seed,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "checkpoint_file": best_ckpt_path.name,
        "sha256": h_best,
        "elapsed_sec": round(elapsed, 2)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, choices=["arm_a", "arm_b", "all"], default="all")
    args = parser.parse_args()
    
    seeds = [101, 202, 303]
    records = []
    
    if args.arm in ["arm_a", "all"]:
        train_file = REPO_ROOT / "experiments/cognition/cr_exp_19/data/short_control_train.jsonl"
        for s in seeds:
            rec = train_arm("arm_a", train_file, s, total_steps=300)
            records.append(rec)
            
    if args.arm in ["arm_b", "all"]:
        train_file = REPO_ROOT / "experiments/cognition/cr_exp_19/data/extended_train.jsonl"
        for s in seeds:
            rec = train_arm("arm_b", train_file, s, total_steps=300)
            records.append(rec)
            
    with open(REPO_ROOT / "experiments/cognition/cr_exp_19/results/training_summary.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print("\nAll 6 training runs completed successfully.")

if __name__ == "__main__":
    main()
