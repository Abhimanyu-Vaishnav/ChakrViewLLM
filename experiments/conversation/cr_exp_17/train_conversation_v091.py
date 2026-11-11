"""
Phase F: Main Conversational Training Pipeline (train_conversation_v091.py)
Executes staged supervised conversational fine-tuning:
- Model: ChakrMicro v0.1 (3,443,136 parameters)
- Initializer: Conversational SFT Parent (CBR-SFT-04 step 200, SHA-256: 327118ce...)
- Optimizer: AdamW (lr=8e-5, weight_decay=0.01, betas=(0.9, 0.95)) with Cosine Annealing
- Objective: Assistant-target cross-entropy loss with prompt masking
- Monitoring: Steps, train loss, val loss, val token accuracy, gradient norm, generation samples
- Checkpoint strategy: Saves best validation loss checkpoint separately
"""

import sys
import os
import json
import time
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ModelConfig, ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.conversation.cr_exp_17.conversation_formatter import ConversationFormatter
from experiments.conversation.cr_exp_17.conversation_loss import ConversationLoss

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

PARENT_CKPT_PATH = REPO_ROOT / "experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt"
TRAIN_DATA_PATH = REPO_ROOT / "experiments/conversation/cr_exp_17/data/conversation_train_1080.jsonl"
VAL_DATA_PATH = REPO_ROOT / "experiments/conversation/cr_exp_17/data/conversation_val_120.jsonl"

OUT_DIR = REPO_ROOT / "experiments/conversation/cr_exp_17"
CKPT_DIR = OUT_DIR / "checkpoints"
LOG_DIR = OUT_DIR / "logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing training on: {device.upper()}")

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
    max_len = max(len(x[0]) for x in batch)
    # Pad to multiple of 8 for efficiency
    pad_len = ((max_len + 7) // 8) * 8
    
    input_ids_batch = []
    labels_batch = []
    
    for inp, lab in batch:
        pad_size = pad_len - len(inp)
        # Pad with 0 (<pad>) and -100 for labels
        input_ids_batch.append(inp + [0] * pad_size)
        labels_batch.append(lab + [-100] * pad_size)
        
    return torch.tensor(input_ids_batch, dtype=torch.long), torch.tensor(labels_batch, dtype=torch.long)

def generate_sample(model, prompt_str, max_tokens=30):
    model.eval()
    input_ids = tokenizer.encode(prompt_str)
    curr_ids = list(input_ids)
    gen_ids = []
    with torch.no_grad():
        for _ in range(max_tokens):
            if len(curr_ids) >= 510: break
            x = torch.tensor([curr_ids], dtype=torch.long, device=device)
            logits = model(x)[0, -1, :]
            nxt = int(torch.argmax(logits).item())
            if nxt == 1: break # EOS
            curr_ids.append(nxt)
            gen_ids.append(nxt)
            text = tokenizer.decode(gen_ids, errors="replace")
            if "\nUser:" in text or "User:" in text:
                break
    resp = tokenizer.decode(gen_ids, errors="replace")
    for st in ["\nUser:", "User:"]:
        if st in resp: resp = resp.split(st)[0]
    return resp.strip()

def train(total_steps=500, micro_batch_size=4, grad_accum_steps=2, lr=8e-5, seed=2026):
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Load model
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, max_seq_len=512, tie_embeddings=True)
    model = ChakrMicro(cfg).to(device)
    
    print(f"Loading parent checkpoint: {PARENT_CKPT_PATH.name}")
    parent_ckpt = torch.load(PARENT_CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(parent_ckpt["model_state_dict"])
    
    train_dataset = ConversationalDataset(TRAIN_DATA_PATH)
    val_dataset = ConversationalDataset(VAL_DATA_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=micro_batch_size, shuffle=False, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)
    
    print(f"\nStarting Conversational Training ({total_steps} steps, effective batch size={micro_batch_size * grad_accum_steps})...")
    
    step = 0
    best_val_loss = float("inf")
    step_logs = []
    start_time = time.time()
    
    probe_prompts = [
        "User: Hello! How are you today?\nAssistant:",
        "User: My name is Alice.\nAssistant: Nice to meet you Alice!\nUser: What is my name?\nAssistant:",
        "User: I live in Mumbai.\nAssistant: Mumbai is a wonderful city.\nUser: Which city do I live in?\nAssistant:",
        "User: Namaste! Aap kaise hain?\nAssistant:"
    ]
    
    train_iter = iter(train_loader)
    optimizer.zero_grad()
    
    while step < total_steps:
        model.train()
        accum_loss = 0.0
        
        for _ in range(grad_accum_steps):
            try:
                inp_b, lab_b = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inp_b, lab_b = next(train_iter)
                
            inp_b = inp_b.to(device)
            lab_b = lab_b.to(device)
            
            logits = model(inp_b)
            loss, acc = loss_fn(logits, lab_b)
            loss = loss / grad_accum_steps
            loss.backward()
            accum_loss += loss.item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        step += 1
        
        if step % 25 == 0 or step == total_steps:
            model.eval()
            val_loss_sum = 0.0
            val_acc_sum = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for v_inp, v_lab in val_loader:
                    v_inp = v_inp.to(device)
                    v_lab = v_lab.to(device)
                    v_logits = model(v_inp)
                    v_loss, v_acc = loss_fn(v_logits, v_lab)
                    val_loss_sum += v_loss.item()
                    val_acc_sum += v_acc.item()
                    val_batches += 1
                    
            avg_val_loss = val_loss_sum / max(val_batches, 1)
            avg_val_acc = val_acc_sum / max(val_batches, 1)
            
            elapsed = time.time() - start_time
            print(f"[Step {step:03d}/{total_steps}] TrainLoss: {accum_loss:.4f} | ValLoss: {avg_val_loss:.4f} | ValAcc: {avg_val_acc*100:.2f}% | Elapsed: {elapsed:.1f}s")
            
            # Print sample outputs
            if step % 50 == 0:
                print("--- Sample Generations ---")
                for pr in probe_prompts:
                    ans = generate_sample(model, pr)
                    print(f"  Q: {repr(pr)}")
                    print(f"  A: {repr(ans)}")
                print("--------------------------")
                
            log_entry = {
                "step": step,
                "train_loss": accum_loss,
                "val_loss": avg_val_loss,
                "val_acc": avg_val_acc,
                "lr": scheduler.get_last_lr()[0]
            }
            step_logs.append(log_entry)
            
            # Save periodic and best
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "model_config": cfg,
                "val_loss": avg_val_loss,
                "val_acc": avg_val_acc,
                "step_logs": step_logs
            }, CKPT_DIR / "chakrview_v091_conversational_latest.pt")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"*** New best val loss: {best_val_loss:.4f}! Saving best model... ***")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "model_config": cfg,
                    "val_loss": best_val_loss,
                    "val_acc": avg_val_acc,
                    "best_step": step,
                    "parent_checkpoint": str(PARENT_CKPT_PATH),
                    "parent_sha256": "327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153"
                }, CKPT_DIR / "chakrview_v091_conversational_best.pt")
                
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s! Best Val Loss: {best_val_loss:.4f}")
    
    with open(LOG_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_steps": total_steps,
            "total_time_seconds": total_time,
            "best_val_loss": best_val_loss,
            "history": step_logs
        }, f, indent=2)

if __name__ == "__main__":
    train(total_steps=500, micro_batch_size=4, grad_accum_steps=2, lr=8e-5, seed=2026)
