"""Replication Training Script for CR-EXP-16D-R
Trains 12 independent runs across 3 new seeds (202, 303, 404) for all 4 arms:
- Arm A (0% copy replay / 4000 retrieval)
- Arm B (25% copy replay / 3000 retrieval)
- Arm C (50% copy replay / 2000 retrieval)
- Arm D (75% copy replay / 1000 retrieval)

Strictly initialized from the exact same CR-16A Arm D parent checkpoint.
Logs target token exposure, peak memory, training duration, and saves checkpoints.
"""
import os
import sys
import json
import time
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

PARENT_CKPT_PATH = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery/checkpoints/arm_d_generalization/arm_d_generalization_best.pt"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
DATA_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/data"
REP_CKPT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_replication_audit/checkpoints"
LOG_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_replication_audit/logs"
REP_CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

class MultiTaskDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len=64):
        with open(data_path, "r", encoding="utf-8") as f:
            self.examples = json.load(f)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        prompt = item["prompt"]
        target = item["target"]
        
        prompt_tokens = self.tokenizer.encode(prompt)
        target_tokens = self.tokenizer.encode(target)
        
        full_tokens = prompt_tokens + target_tokens
        full_tokens = full_tokens[:self.max_seq_len]
        
        input_ids = full_tokens[:-1]
        labels = full_tokens[1:]
        
        prompt_len = len(prompt_tokens)
        masked_labels = []
        for i, tok in enumerate(labels):
            if i < prompt_len - 1:
                masked_labels.append(-100)
            else:
                masked_labels.append(tok)
                
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(masked_labels, dtype=torch.long),
            "task": item["task"],
            "prompt_len": prompt_len,
            "target_len": len(target_tokens)
        }

def collate_fn(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids_padded = []
    labels_padded = []
    tasks = [b["task"] for b in batch]
    prompt_lens = [b["prompt_len"] for b in batch]
    target_lens = [b["target_len"] for b in batch]
    
    for b in batch:
        inp = b["input_ids"]
        lab = b["labels"]
        pad_len = max_len - len(inp)
        input_ids_padded.append(torch.cat([inp, torch.full((pad_len,), 0, dtype=torch.long)]))
        labels_padded.append(torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)]))
        
    return {
        "input_ids": torch.stack(input_ids_padded),
        "labels": torch.stack(labels_padded),
        "tasks": tasks,
        "prompt_lens": prompt_lens,
        "target_lens": target_lens
    }

def train_replication_arm(arm_name, data_file, seed, total_steps=400):
    print(f"\n=======================================================")
    print(f"TRAINING REPLICATION: {arm_name} | Seed: {seed} | Steps: {total_steps}")
    print(f"=======================================================")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    train_dataset = MultiTaskDataset(DATA_DIR / data_file, tokenizer)
    val_dataset = MultiTaskDataset(DATA_DIR / "cr16d_multitask_val.json", tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Load from verified parent checkpoint
    parent_ckpt = torch.load(PARENT_CKPT_PATH, map_location=device, weights_only=False)
    config = parent_ckpt["config"] if "config" in parent_ckpt else parent_ckpt["model_config"]
    model = ChakrMicro(config).to(device)
    model.load_state_dict(parent_ckpt["model_state_dict"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95))
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    arm_ckpt_dir = REP_CKPT_DIR / f"{arm_name}_seed{seed}"
    arm_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    step = 0
    best_val_loss = float("inf")
    train_iter = iter(train_loader)
    
    total_copy_target_tokens = 0
    total_retr_target_tokens = 0
    
    while step < total_steps:
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        tasks = batch["tasks"]
        t_lens = batch["target_lens"]
        for t, tl in zip(tasks, t_lens):
            if t == "copying": total_copy_target_tokens += tl
            else: total_retr_target_tokens += tl
            
        optimizer.zero_grad()
        logits = model(input_ids)
        if isinstance(logits, tuple): logits = logits[0]
        
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        
        if step % 50 == 0 or step == total_steps:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            correct_tokens = 0
            total_tokens = 0
            
            with torch.no_grad():
                for v_batch in val_loader:
                    v_inp = v_batch["input_ids"].to(device)
                    v_lab = v_batch["labels"].to(device)
                    v_logits = model(v_inp)
                    if isinstance(v_logits, tuple): v_logits = v_logits[0]
                    v_l = criterion(v_logits.view(-1, v_logits.size(-1)), v_lab.view(-1))
                    val_loss += v_l.item()
                    val_steps += 1
                    
                    preds = v_logits.argmax(dim=-1)
                    mask = (v_lab != -100)
                    correct_tokens += (preds[mask] == v_lab[mask]).sum().item()
                    total_tokens += mask.sum().item()
                    
            avg_val_loss = val_loss / max(val_steps, 1)
            target_acc = correct_tokens / max(total_tokens, 1)
            print(f"[{arm_name} s{seed}] Step {step} | Val Loss: {avg_val_loss:.4f} | Target Acc: {target_acc*100:.2f}%")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "model_config": config,
                    "val_loss": avg_val_loss,
                    "target_acc": target_acc,
                    "seed": seed,
                    "arm": arm_name
                }, arm_ckpt_dir / f"{arm_name}_best.pt")
                
    duration = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    print(f"Finished {arm_name} s{seed} in {duration:.1f}s. Peak VRAM: {peak_vram:.1f}MB. Copy tokens: {total_copy_target_tokens}, Retr tokens: {total_retr_target_tokens}")
    
    # Save training log
    log_info = {
        "arm_name": arm_name,
        "seed": seed,
        "total_steps": total_steps,
        "duration_seconds": duration,
        "peak_vram_mb": peak_vram,
        "best_val_loss": best_val_loss,
        "copy_target_tokens_seen": total_copy_target_tokens,
        "retr_target_tokens_seen": total_retr_target_tokens
    }
    with open(LOG_DIR / f"{arm_name}_seed{seed}_train_log.json", "w") as f:
        json.dump(log_info, f, indent=2)

if __name__ == "__main__":
    arms = [
        ("arm_a_replay00", "arm_a_replay00_train.json"),
        ("arm_b_replay25", "arm_b_replay25_train.json"),
        ("arm_c_replay50", "arm_c_replay50_train.json"),
        ("arm_d_replay75", "arm_d_replay75_train.json"),
    ]
    seeds = [202, 303, 404]
    for arm_name, data_file in arms:
        for seed in seeds:
            train_replication_arm(arm_name, data_file, seed=seed, total_steps=400)

