"""CR-16F Factorial Training Harness:
Executes the 2x2 Factorial Design across Arms A, B, C, D:
- Arm A: 25% Copy Replay, 1.00x Abstention Loss
- Arm B: 50% Copy Replay, 1.00x Abstention Loss
- Arm C: 25% Copy Replay, 0.25x Abstention Loss
- Arm D: 50% Copy Replay, 0.25x Abstention Loss
Seeds: 2026, 3030, 4040 (12 models total).
Parent: Verified CR-16A Arm D checkpoint.
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
BASE_DATA_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/data/arm_d_shuffled_balancedspan_train.json"
VAL_FILE = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16e_position_invariant_retrieval/data/cr16e_multitask_val.json"

OUT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16f_multicapability_retention"
CKPT_DIR = OUT_DIR / "checkpoints"
LOG_DIR = OUT_DIR / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

# Load data pools
with open(BASE_DATA_FILE, "r", encoding="utf-8") as f:
    full_train_data = json.load(f)
copy_pool = [x for x in full_train_data if x.get("task") == "copying"]
retrieval_pool = [x for x in full_train_data if x.get("task") in ("retrieval", "abstention")]

with open(VAL_FILE, "r", encoding="utf-8") as f:
    val_data = json.load(f)

def tokenize_sample(item, tokenizer, max_seq_len=64):
    prompt = item["prompt"]
    target = item["target"]
    task = item["task"]
    
    prompt_tokens = tokenizer.encode(prompt)
    target_tokens = tokenizer.encode(target)
    
    full_tokens = (prompt_tokens + target_tokens)[:max_seq_len]
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
        "task": task
    }

class ValDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.samples = [tokenize_sample(x, tokenizer) for x in examples]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_val(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    inp_pad, lab_pad = [], []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        inp_pad.append(torch.cat([b["input_ids"], torch.full((pad_len,), 0, dtype=torch.long)]))
        lab_pad.append(torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
    return torch.stack(inp_pad), torch.stack(lab_pad)

def create_balanced_batch(copy_samples, retr_samples, copy_count, retr_count, tokenizer):
    selected_items = copy_samples[:copy_count] + retr_samples[:retr_count]
    tokenized = [tokenize_sample(x, tokenizer) for x in selected_items]
    max_len = max(len(b["input_ids"]) for b in tokenized)
    
    inp_pad, lab_pad, task_weights = [], [], []
    tasks = []
    
    for b in tokenized:
        pad_len = max_len - len(b["input_ids"])
        inp_pad.append(torch.cat([b["input_ids"], torch.full((pad_len,), 0, dtype=torch.long)]))
        lab_pad.append(torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        tasks.append(b["task"])
        
    return {
        "input_ids": torch.stack(inp_pad).to(device),
        "labels": torch.stack(lab_pad).to(device),
        "tasks": tasks
    }

def train_cr16f_model(arm_name, replay_pct, lambda_neg, seed, total_steps=400):
    run_name = f"{arm_name}_seed{seed}"
    print(f"\n=======================================================")
    print(f"RUN: {run_name} | Replay: {replay_pct}% | Lambda: {lambda_neg}x | Steps: {total_steps}")
    print(f"=======================================================")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    
    # Determine batch split (batch size 16)
    copy_per_batch = int(16 * (replay_pct / 100))
    retr_per_batch = 16 - copy_per_batch
    
    # Shuffle pools with deterministic seed
    c_pool = list(copy_pool)
    r_pool = list(retrieval_pool)
    random.shuffle(c_pool)
    random.shuffle(r_pool)
    
    val_dataset = ValDataset(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_val)
    
    parent_ckpt = torch.load(PARENT_CKPT_PATH, map_location=device, weights_only=False)
    config = parent_ckpt["config"] if "config" in parent_ckpt else parent_ckpt["model_config"]
    model = ChakrMicro(config).to(device)
    model.load_state_dict(parent_ckpt["model_state_dict"])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95))
    ce_loss_none = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    
    run_ckpt_dir = CKPT_DIR / run_name
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    step = 0
    best_val_loss = float("inf")
    
    c_idx, r_idx = 0, 0
    step_logs = []
    
    while step < total_steps:
        model.train()
        
        # Slice batch
        if c_idx + copy_per_batch > len(c_pool):
            random.shuffle(c_pool)
            c_idx = 0
        if r_idx + retr_per_batch > len(r_pool):
            random.shuffle(r_pool)
            r_idx = 0
            
        batch_copy = c_pool[c_idx:c_idx + copy_per_batch]
        batch_retr = r_pool[r_idx:r_idx + retr_per_batch]
        c_idx += copy_per_batch
        r_idx += retr_per_batch
        
        batch = create_balanced_batch(batch_copy, batch_retr, copy_per_batch, retr_per_batch, tokenizer)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        tasks = batch["tasks"]
        
        optimizer.zero_grad()
        logits = model(input_ids)
        if isinstance(logits, tuple): logits = logits[0]
        
        # Compute unreduced loss
        B, S, V = logits.shape
        loss_unreduced = ce_loss_none(logits.view(-1, V), labels.view(-1)).view(B, S)
        
        # Build task weight tensor
        weights = torch.ones(B, 1, device=device)
        for i, t in enumerate(tasks):
            if t == "abstention":
                weights[i, 0] = lambda_neg
                
        # Weighted loss
        active_mask = (labels != -100).float()
        weighted_loss = (loss_unreduced * weights * active_mask).sum() / active_mask.sum().clamp(min=1.0)
        
        weighted_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        optimizer.step()
        step += 1
        
        # Compute separate component losses for monitoring
        with torch.no_grad():
            copy_mask = torch.tensor([t == "copying" for t in tasks], device=device).unsqueeze(1) * active_mask
            pos_mask = torch.tensor([t == "retrieval" for t in tasks], device=device).unsqueeze(1) * active_mask
            neg_mask = torch.tensor([t == "abstention" for t in tasks], device=device).unsqueeze(1) * active_mask
            
            l_copy = (loss_unreduced * copy_mask).sum() / copy_mask.sum().clamp(min=1.0)
            l_pos = (loss_unreduced * pos_mask).sum() / pos_mask.sum().clamp(min=1.0)
            l_neg = (loss_unreduced * neg_mask).sum() / neg_mask.sum().clamp(min=1.0)
            
        if step % 50 == 0 or step == total_steps:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            correct_tokens = 0
            total_tokens = 0
            
            with torch.no_grad():
                for v_inp, v_lab in val_loader:
                    v_inp = v_inp.to(device)
                    v_lab = v_lab.to(device)
                    v_logits = model(v_inp)
                    if isinstance(v_logits, tuple): v_logits = v_logits[0]
                    v_loss_unred = ce_loss_none(v_logits.view(-1, V), v_lab.view(-1)).view(v_lab.shape)
                    v_mask = (v_lab != -100).float()
                    val_loss += (v_loss_unred * v_mask).sum().item() / v_mask.sum().clamp(min=1.0).item()
                    val_steps += 1
                    
                    preds = v_logits.argmax(dim=-1)
                    correct_tokens += ((preds == v_lab) & (v_lab != -100)).sum().item()
                    total_tokens += v_mask.sum().item()
                    
            avg_val_loss = val_loss / max(val_steps, 1)
            target_acc = correct_tokens / max(total_tokens, 1)
            
            log_entry = {
                "step": step,
                "train_loss": weighted_loss.item(),
                "l_copy": l_copy.item(),
                "l_pos_retrieval": l_pos.item(),
                "l_neg_abstention": l_neg.item(),
                "grad_norm": grad_norm,
                "val_loss": avg_val_loss,
                "target_acc": target_acc
            }
            step_logs.append(log_entry)
            print(f"[{run_name}] Step {step} | TrainLoss: {weighted_loss.item():.4f} (Copy:{l_copy.item():.3f}, Pos:{l_pos.item():.3f}, Neg:{l_neg.item():.3f}) | ValLoss: {avg_val_loss:.4f} | ValAcc: {target_acc*100:.2f}%")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "model_config": config,
                    "val_loss": avg_val_loss,
                    "target_acc": target_acc,
                    "seed": seed,
                    "arm": arm_name,
                    "replay_pct": replay_pct,
                    "lambda_neg": lambda_neg
                }, run_ckpt_dir / f"{arm_name}_best.pt")
                
    duration = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    print(f"Finished {run_name} in {duration:.1f}s. Peak VRAM: {peak_vram:.1f}MB. Best Val Loss: {best_val_loss:.4f}")
    
    summary = {
        "run_name": run_name,
        "arm": arm_name,
        "replay_pct": replay_pct,
        "lambda_neg": lambda_neg,
        "seed": seed,
        "duration_seconds": duration,
        "peak_vram_mb": peak_vram,
        "best_val_loss": best_val_loss,
        "step_logs": step_logs
    }
    with open(LOG_DIR / f"{run_name}_train_log.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    arms = [
        ("arm_a_replay25_lambda100", 25, 1.00),
        ("arm_b_replay50_lambda100", 50, 1.00),
        ("arm_c_replay25_lambda025", 25, 0.25),
        ("arm_d_replay50_lambda025", 50, 0.25),
    ]
    seeds = [2026, 3030, 4040]
    
    for arm_name, replay_pct, lambda_neg in arms:
        for s in seeds:
            train_cr16f_model(arm_name, replay_pct, lambda_neg, seed=s, total_steps=400)
    print("\nAll 12 CR-16F factorial models trained successfully!")

