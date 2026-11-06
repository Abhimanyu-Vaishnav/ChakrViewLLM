"""
Stage 4: Bounded Conversational Fine-Tuning Pilot
Disciplined training with anti-collapse protections:
1. Warm-starts strictly from verified Parent (cbr_sft_04_step_200.pt, SHA-256: 327118ce...)
2. Micro-batch size = 4, Gradient Accumulation = 4 (Effective batch size = 16)
3. Learning Rate = 5e-5 with Cosine Decay to 1e-5
4. AdamW with weight decay = 0.05
5. Gradient clipping = 1.0
6. Target tokens only masked cross-entropy loss (prompt tokens masked with -100)
7. Total steps: 100 optimizer steps (100 * 16 = 1,600 samples processed = 4 epochs over 400 train samples)
8. Regular validation evaluation every 10 optimizer steps
9. Saves best checkpoint by validation loss and final checkpoint at step 100.
"""

import os
import sys
import time
import json
import math
import hashlib
from pathlib import Path
import psutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ModelConfig, ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

PARENT_PATH = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_04" / "checkpoints" / "sft_checkpoint_step_200.pt"
EXPECTED_PARENT_SHA256 = "327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153"

PILOT_DIR = REPO_ROOT / "experiments" / "cognition" / "autonomous_12h" / "conversational_pilot"
DATA_DIR = PILOT_DIR / "dataset"
CKPT_DIR = PILOT_DIR / "checkpoints"
LOG_DIR = PILOT_DIR / "logs"

class DialogueSFTDataset(Dataset):
    def __init__(self, jsonl_path, tok, max_seq_len=512):
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tok = tok
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = item["prompt"]
        response = item["response"] + "\n"
        
        prompt_ids = self.tok.encode(prompt)
        resp_ids = self.tok.encode(response)
        
        all_ids = prompt_ids + resp_ids
        if len(all_ids) > self.max_seq_len:
            all_ids = all_ids[:self.max_seq_len]
            
        input_ids = all_ids[:-1]
        labels = all_ids[1:]
        
        # Mask prompt tokens with -100 so loss is computed ONLY on response
        mask_len = len(prompt_ids) - 1
        labels[:mask_len] = [-100] * mask_len
        
        # Pad up to max_seq_len - 1
        pad_len = (self.max_seq_len - 1) - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [0] * pad_len
            labels = labels + [-100] * pad_len
            
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def compute_eval_loss(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(1, count)

def main():
    print("=== CR-EXP-03 / AUTONOMOUS 12H — STAGE 4 CONVERSATIONAL PILOT ===")
    torch.set_num_threads(4)
    device = torch.device("cpu")
    
    # 1. Verify parent checkpoint
    assert PARENT_PATH.exists(), f"Parent checkpoint not found: {PARENT_PATH}"
    actual_parent_sha = hashlib.sha256(PARENT_PATH.read_bytes()).hexdigest()
    assert actual_parent_sha == EXPECTED_PARENT_SHA256, f"Parent hash mismatch: {actual_parent_sha}"
    print(f"Parent Checkpoint Verified: {actual_parent_sha[:16]}...")
    
    # 2. Tokenizer & Datasets
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    train_ds = DialogueSFTDataset(DATA_DIR / "train.jsonl", tok)
    val_ds = DialogueSFTDataset(DATA_DIR / "val.jsonl", tok)
    
    MICRO_BATCH = 4
    ACCUM_STEPS = 4  # Effective batch = 16
    train_loader = DataLoader(train_ds, batch_size=MICRO_BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=MICRO_BATCH, shuffle=False)
    
    # 3. Model setup
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, max_seq_len=512, tie_embeddings=True)
    model = ChakrMicro(cfg)
    ckpt_data = torch.load(PARENT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.to(device)
    model.train()
    
    # 4. Optimizer, Scheduler & Criterion
    BASE_LR = 5e-5
    MIN_LR = 1e-5
    TOTAL_OPTIM_STEPS = 100
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    def get_lr(step):
        # Cosine decay
        progress = step / TOTAL_OPTIM_STEPS
        return MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))
        
    initial_val_loss = compute_eval_loss(model, val_loader, criterion, device)
    print(f"Initial Parent Baseline Validation Loss: {initial_val_loss:.4f}")
    
    # 5. Training Loop
    optim_step = 0
    micro_step = 0
    accum_loss = 0.0
    history = []
    best_val_loss = float("inf")
    best_ckpt_path = None
    
    t0 = time.time()
    print(f"\nStarting training: {TOTAL_OPTIM_STEPS} optimizer steps (micro_batch={MICRO_BATCH}, accum={ACCUM_STEPS})...")
    
    while optim_step < TOTAL_OPTIM_STEPS:
        for x, y in train_loader:
            if optim_step >= TOTAL_OPTIM_STEPS:
                break
            micro_step += 1
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / ACCUM_STEPS
            loss.backward()
            accum_loss += loss.item()
            
            if micro_step % ACCUM_STEPS == 0:
                optim_step += 1
                
                # Update learning rate
                current_lr = get_lr(optim_step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
                    
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                optimizer.step()
                optimizer.zero_grad()
                
                # Validation evaluation
                val_loss = None
                if optim_step % 10 == 0 or optim_step == TOTAL_OPTIM_STEPS:
                    val_loss = compute_eval_loss(model, val_loader, criterion, device)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_ckpt_path = CKPT_DIR / "conversational_pilot_best_val.pt"
                        torch.save({
                            "optim_step": optim_step,
                            "model_state_dict": model.state_dict(),
                            "val_loss": val_loss,
                            "parent_sha256": EXPECTED_PARENT_SHA256
                        }, best_ckpt_path)
                        
                proc = psutil.Process(os.getpid())
                rss_mb = round(proc.memory_info().rss / (1024 * 1024), 1)
                
                log_entry = {
                    "optim_step": optim_step,
                    "train_loss": round(accum_loss * ACCUM_STEPS, 4),
                    "val_loss": round(val_loss, 4) if val_loss is not None else None,
                    "grad_norm": round(grad_norm, 4),
                    "lr": round(current_lr, 7),
                    "rss_mb": rss_mb
                }
                history.append(log_entry)
                
                if optim_step % 10 == 0 or optim_step == 1:
                    v_str = f" | val_loss: {val_loss:.4f}" if val_loss is not None else ""
                    print(f"Step {optim_step:03d}/{TOTAL_OPTIM_STEPS} | train_loss: {log_entry['train_loss']:.4f}{v_str} | grad_norm: {grad_norm:.3f} | LR: {current_lr:.6f} | RSS: {rss_mb} MB")
                    
                accum_loss = 0.0
                
    total_time = round(time.time() - t0, 2)
    print(f"\nTraining completed: {optim_step} optimizer steps in {total_time}s.")
    
    # 6. Save Final Checkpoint
    final_ckpt_path = CKPT_DIR / "conversational_pilot_final_step_100.pt"
    torch.save({
        "optim_step": TOTAL_OPTIM_STEPS,
        "model_state_dict": model.state_dict(),
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "best_val_loss": best_val_loss,
        "parent_sha256": EXPECTED_PARENT_SHA256,
        "total_elapsed_sec": total_time
    }, final_ckpt_path)
    
    final_sha = hashlib.sha256(final_ckpt_path.read_bytes()).hexdigest()
    best_sha = hashlib.sha256(best_ckpt_path.read_bytes()).hexdigest() if best_ckpt_path else None
    
    print(f"Final Checkpoint Saved: {final_ckpt_path}")
    print(f"  SHA-256: {final_sha}")
    print(f"Best Val Checkpoint: {best_ckpt_path}")
    print(f"  SHA-256: {best_sha}")
    
    # Save training log JSON
    training_log_path = LOG_DIR / "training_log.json"
    with open(training_log_path, "w", encoding="utf-8") as f:
        json.dump({
            "parent_checkpoint": str(PARENT_PATH),
            "parent_sha256": EXPECTED_PARENT_SHA256,
            "final_checkpoint": str(final_ckpt_path),
            "final_sha256": final_sha,
            "best_checkpoint": str(best_ckpt_path),
            "best_sha256": best_sha,
            "initial_val_loss": round(initial_val_loss, 4),
            "best_val_loss": round(best_val_loss, 4),
            "history": history
        }, f, indent=2)
    print(f"Saved training log to: {training_log_path}")

if __name__ == "__main__":
    main()

