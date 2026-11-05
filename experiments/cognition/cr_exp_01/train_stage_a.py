"""
CR-EXP-01 Stage A Isolated Pilot Trainer
Enforces:
- STRICT LIMIT: exactly 100 optimizer steps.
- Batch size: 1.
- Sequence length: 512.
- CPU-first execution (4 threads for optimal efficiency).
- AdamW optimizer (lr=1e-4, weight_decay=0.01).
- Loss masking: compute cross-entropy ONLY on tokens after 'Answer: '.
- Full telemetry logging (step, train loss, val loss, grad norm, lr, elapsed, RSS).
- Saves candidate checkpoint at step 100 and stops immediately.
"""

import os
import sys
import time
import json
import psutil
import hashlib
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ModelConfig, ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

OUTPUT_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_01" / "runs" / "stage_a_pilot"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

PARENT_CKPT_PATH = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_04" / "checkpoints" / "sft_checkpoint_step_200.pt"
EXPECTED_PARENT_SHA256 = "327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153"

class VariableBindingDataset(Dataset):
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
        text = item["text"]
        
        # Determine prefix and target split
        prefix, target = text.split("Answer:", 1)
        prefix = prefix + "Answer:"
        
        prefix_ids = self.tok.encode(prefix)
        target_ids = self.tok.encode(target) + [self.tok.encode("\n")[0] if self.tok.encode("\n") else 1]
        
        all_ids = prefix_ids + target_ids
        if len(all_ids) > self.max_seq_len:
            all_ids = all_ids[:self.max_seq_len]
            
        input_ids = all_ids[:-1]
        labels = all_ids[1:]
        
        # Mask prefix labels with -100 so loss is computed only on answer
        mask_len = len(prefix_ids) - 1
        labels[:mask_len] = [-100] * mask_len
        
        # Pad to max_seq_len - 1
        pad_len = (self.max_seq_len - 1) - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [0] * pad_len
            labels = labels + [-100] * pad_len
            
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

def compute_val_loss(model, val_loader, criterion, device):
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
            if count >= 10:  # evaluate on 10 batches
                break
    model.train()
    return total_loss / max(1, count)

def main():
    print("=== CR-EXP-01 STAGE A PILOT TRAINING ===")
    torch.set_num_threads(4)
    device = torch.device("cpu")
    
    # 1. Verify parent checkpoint
    assert PARENT_CKPT_PATH.exists(), f"Missing parent checkpoint: {PARENT_CKPT_PATH}"
    actual_sha = hashlib.sha256(PARENT_CKPT_PATH.read_bytes()).hexdigest()
    assert actual_sha == EXPECTED_PARENT_SHA256, f"Parent hash mismatch! Got {actual_sha}"
    print(f"Parent Checkpoint Verified: {PARENT_CKPT_PATH.name} ({actual_sha[:16]}...)")
    
    # 2. Tokenizer & Datasets
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    train_ds = VariableBindingDataset(REPO_ROOT / "data" / "sft" / "cr_exp01" / "train_1000.jsonl", tok)
    val_ds = VariableBindingDataset(REPO_ROOT / "data" / "sft" / "cr_exp01" / "val_100.jsonl", tok)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    # 3. Model setup
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, max_seq_len=512, tie_embeddings=True)
    model = ChakrMicro(cfg)
    
    ckpt_data = torch.load(PARENT_CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.to(device)
    model.train()
    
    # 4. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Initial validation loss
    initial_val_loss = compute_val_loss(model, val_loader, criterion, device)
    print(f"Initial Baseline Validation Loss: {initial_val_loss:.4f}")
    
    # 5. Training Loop: EXACTLY 100 STEPS
    MAX_STEPS = 100
    step = 0
    history = []
    t_start = time.time()
    
    print("\nStarting isolated 100-step training loop...")
    while step < MAX_STEPS:
        for x, y in train_loader:
            if step >= MAX_STEPS:
                break
            step += 1
            t_step_0 = time.time()
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            optimizer.step()
            
            t_step_1 = time.time()
            step_time = t_step_1 - t_step_0
            
            # Validation every 20 steps
            val_loss = None
            if step % 20 == 0 or step == MAX_STEPS:
                val_loss = compute_val_loss(model, val_loader, criterion, device)
                
            proc = psutil.Process(os.getpid())
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            
            entry = {
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4) if val_loss is not None else None,
                "grad_norm": round(grad_norm, 4),
                "lr": 1e-4,
                "step_time_sec": round(step_time, 4),
                "rss_mb": round(rss_mb, 2)
            }
            history.append(entry)
            
            if step % 10 == 0 or step == 1:
                v_str = f" | val_loss: {val_loss:.4f}" if val_loss is not None else ""
                print(f"Step {step:03d}/{MAX_STEPS} | train_loss: {loss.item():.4f}{v_str} | grad_norm: {grad_norm:.3f} | RSS: {rss_mb:.1f} MB")
                
    t_end = time.time()
    total_elapsed = t_end - t_start
    print(f"\nTraining loop finished: exactly {step} steps completed in {total_elapsed:.2f} seconds.")
    
    # 6. Save Candidate Checkpoint
    cand_ckpt_path = CHECKPOINT_DIR / "stage_a_candidate_step_100.pt"
    save_payload = {
        "sft_step": 100,
        "seed": 101,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "parent_checkpoint": str(PARENT_CKPT_PATH),
        "parent_sha256": EXPECTED_PARENT_SHA256,
        "history": history,
        "total_elapsed_sec": round(total_elapsed, 2),
        "tokens_processed": MAX_STEPS * 512,
        "initial_val_loss": round(initial_val_loss, 4),
        "final_val_loss": round(history[-1]["val_loss"], 4)
    }
    torch.save(save_payload, cand_ckpt_path)
    
    cand_sha = hashlib.sha256(cand_ckpt_path.read_bytes()).hexdigest()
    print(f"Candidate Checkpoint Saved: {cand_ckpt_path}")
    print(f"Candidate File Size: {cand_ckpt_path.stat().st_size:,} bytes")
    print(f"Candidate SHA-256: {cand_sha}")
    
    # Save training log JSON
    log_path = OUTPUT_DIR / "stage_a_training_log.json"
    log_path.write_text(json.dumps({
        "parent_checkpoint": str(PARENT_CKPT_PATH),
        "candidate_checkpoint": str(cand_ckpt_path),
        "candidate_sha256": cand_sha,
        "max_steps": MAX_STEPS,
        "total_elapsed_sec": round(total_elapsed, 2),
        "history": history
    }, indent=2), encoding="utf-8")
    print(f"Saved training telemetry to {log_path}")

if __name__ == "__main__":
    main()

