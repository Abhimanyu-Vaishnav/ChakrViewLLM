"""CR-16E-R2 Controlled Gradient-Interference Training Script:
Executes Arm G1 (Negative-target loss enabled) vs Arm G2 (Negative-target loss masked).
Dataset: arm_d_shuffled_balancedspan_train.json
Seeds: 2026, 3030, 4040
Parent: CR-16A Arm D best checkpoint.
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
DATA_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/data/arm_d_shuffled_balancedspan_train.json"
VAL_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/data/cr16e_multitask_val.json"

OUT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_r2_answerability_gradient_study"
CKPT_DIR = OUT_DIR / "checkpoints"
LOG_DIR = OUT_DIR / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

class ControlledStudyDataset(Dataset):
    def __init__(self, data_path, tokenizer, mask_abstention_loss=False, max_seq_len=64):
        with open(data_path, "r", encoding="utf-8") as f:
            self.examples = json.load(f)
        self.tokenizer = tokenizer
        self.mask_abstention_loss = mask_abstention_loss
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        prompt = item["prompt"]
        target = item["target"]
        task = item["task"]
        
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
            elif self.mask_abstention_loss and task == "abstention":
                # Arm G2: mask out negative abstention target tokens completely
                masked_labels.append(-100)
            else:
                masked_labels.append(tok)
                
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(masked_labels, dtype=torch.long),
            "task": task,
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

def train_controlled_run(condition_name, mask_abstention, seed, total_steps=400):
    run_name = f"{condition_name}_seed{seed}"
    print(f"\n=======================================================")
    print(f"STARTING RUN: {run_name} | Mask Abstention: {mask_abstention} | Steps: {total_steps}")
    print(f"=======================================================")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    
    train_dataset = ControlledStudyDataset(DATA_FILE, tokenizer, mask_abstention_loss=mask_abstention)
    val_dataset = ControlledStudyDataset(VAL_FILE, tokenizer, mask_abstention_loss=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    parent_ckpt = torch.load(PARENT_CKPT_PATH, map_location=device, weights_only=False)
    config = parent_ckpt["config"] if "config" in parent_ckpt else parent_ckpt["model_config"]
    model = ChakrMicro(config).to(device)
    model.load_state_dict(parent_ckpt["model_state_dict"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95))
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    run_ckpt_dir = CKPT_DIR / run_name
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    step = 0
    best_val_loss = float("inf")
    train_iter = iter(train_loader)
    
    active_tokens_counted = 0
    
    while step < total_steps:
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        active_tokens_in_batch = (labels != -100).sum().item()
        active_tokens_counted += active_tokens_in_batch
        
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
            print(f"[{run_name}] Step {step} | Val Loss: {avg_val_loss:.4f} | Target Acc: {target_acc*100:.2f}% | Active Toks Seen: {active_tokens_counted}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "model_config": config,
                    "val_loss": avg_val_loss,
                    "target_acc": target_acc,
                    "seed": seed,
                    "condition": condition_name,
                    "mask_abstention": mask_abstention
                }, run_ckpt_dir / f"{condition_name}_best.pt")
                
    duration = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    print(f"Finished {run_name} in {duration:.1f}s. Peak VRAM: {peak_vram:.1f}MB.")
    
    log_info = {
        "run_name": run_name,
        "condition": condition_name,
        "mask_abstention": mask_abstention,
        "seed": seed,
        "total_steps": total_steps,
        "duration_seconds": duration,
        "peak_vram_mb": peak_vram,
        "best_val_loss": best_val_loss,
        "active_tokens_counted": active_tokens_counted
    }
    with open(LOG_DIR / f"{run_name}_train_log.json", "w") as f:
        json.dump(log_info, f, indent=2)

if __name__ == "__main__":
    seeds = [2026, 3030, 4040]
    # Arm G1: Negative Loss Enabled (mask_abstention=False)
    for s in seeds:
        train_controlled_run("arm_g1_enabled", mask_abstention=False, seed=s, total_steps=400)
    # Arm G2: Negative Loss Masked (mask_abstention=True)
    for s in seeds:
        train_controlled_run("arm_g2_masked", mask_abstention=True, seed=s, total_steps=400)

