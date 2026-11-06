"""
CR-EXP-09 Candidate 3 Training Engine
Fine-tunes from Candidate 2 best validation checkpoint
using the expanded dataset (train_candidate3.json, val_candidate2.json):
- Max updates: 250
- Learning rate: 2.5e-5
- Effective batch size: 8
- Relational answer token weight: 2.0, conversational/abstention: 1.0
- AdamW, grad clipping 0.5
Outputs checkpoints to checkpoints/candidate_3/
"""

import sys
import json
import random
import hashlib
from pathlib import Path
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
MISSION_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_09_autonomous_mission"
DATA_DIR = MISSION_DIR / "data"
CKPT_DIR = MISSION_DIR / "checkpoints" / "candidate_3"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

INIT_CKPT = MISSION_DIR / "checkpoints" / "candidate_2" / "cr_exp_09_candidate_2_best_val.pt"

def load_initial_candidate():
    cfg = ModelConfig()
    m = ChakrMicro(cfg)
    ck = torch.load(INIT_CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("model_state_dict", ck), strict=True)
    return m

def prepare_batch(samples, tok):
    input_ids_list = []
    token_weights_list = []
    
    for s in samples:
        p = s["prompt"]
        c = s["completion"]
        is_rel = s["is_relational"]
        
        enc_p = tok.encode(p)
        enc_full = tok.encode(p + c)
        
        input_ids_list.append(enc_full)
        w_val = 2.0 if is_rel else 1.0
        w = [0.0] * len(enc_p) + [w_val] * (len(enc_full) - len(enc_p))
        token_weights_list.append(w)
        
    max_len = max(len(ids) for ids in input_ids_list)
    padded_inputs = []
    padded_targets = []
    padded_weights = []
    
    for ids, w in zip(input_ids_list, token_weights_list):
        pad_len = max_len - len(ids)
        padded_inputs.append(ids + [0] * pad_len)
        padded_targets.append(ids + [0] * pad_len)
        padded_weights.append(w + [0.0] * pad_len)
        
    return {
        "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
        "target_ids": torch.tensor(padded_targets, dtype=torch.long),
        "token_weights": torch.tensor(padded_weights, dtype=torch.float)
    }

def evaluate_val_loss(model, val_samples, tok, batch_size=8):
    model.eval()
    total_loss = 0.0
    batches = [val_samples[i:i+batch_size] for i in range(0, len(val_samples), batch_size)]
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    
    with torch.no_grad():
        for b_samples in batches:
            b_data = prepare_batch(b_samples, tok)
            inp = b_data["input_ids"]
            tgt = b_data["target_ids"]
            w = b_data["token_weights"][:, 1:].contiguous()
            
            logits = model(inp)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = tgt[:, 1:].contiguous()
            
            t_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1)).view(shift_targets.size())
            m_loss = (t_losses * w).sum() / (w.sum() + 1e-8)
            total_loss += m_loss.item()
            
    return total_loss / len(batches) if batches else 0.0

def main():
    print("=== CR-EXP-09: Training Candidate 3 (Multi-Task Deductive Regularization) ===")
    random.seed(42)
    torch.manual_seed(42)
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    with open(DATA_DIR / "train_candidate3.json", encoding="utf-8") as f:
        train_samples = json.load(f)
        
    with open(DATA_DIR / "val_candidate2.json", encoding="utf-8") as f:
        val_samples = json.load(f)
        
    model = load_initial_candidate()
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-5, weight_decay=0.01)
    
    pre_loss = evaluate_val_loss(model, val_samples, tok)
    print(f"Pre-training Val Loss: {pre_loss:.4f}")
    
    best_val_loss = pre_loss
    total_steps = 250
    batch_size = 8
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    
    history = []
    
    for step in range(1, total_steps + 1):
        batch = random.sample(train_samples, batch_size)
        b_data = prepare_batch(batch, tok)
        inp = b_data["input_ids"]
        tgt = b_data["target_ids"]
        w = b_data["token_weights"][:, 1:].contiguous()
        
        optimizer.zero_grad()
        logits = model(inp)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = tgt[:, 1:].contiguous()
        
        t_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1)).view(shift_targets.size())
        loss = (t_losses * w).sum() / (w.sum() + 1e-8)
        
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        if step % 25 == 0 or step == total_steps:
            val_loss = evaluate_val_loss(model, val_samples, tok)
            print(f"Step {step:3d}/{total_steps} | Loss: {loss.item():.4f} | GradNorm: {gnorm.item():.4f} | Val Loss: {val_loss:.4f}")
            history.append({
                "step": step,
                "loss": round(loss.item(), 4),
                "grad_norm": round(gnorm.item(), 4),
                "val_loss": round(val_loss, 4)
            })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = CKPT_DIR / "cr_exp_09_candidate_3_best_val.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "model_class": "ChakrMicro",
                    "training_arm": "cr_exp_09_candidate_3"
                }, best_path)
                print(f"  --> Saved new best validation checkpoint (Val Loss: {val_loss:.4f})")
                
    final_path = CKPT_DIR / "cr_exp_09_candidate_3_final.pt"
    torch.save({
        "step": total_steps,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "model_class": "ChakrMicro",
        "training_arm": "cr_exp_09_candidate_3_final"
    }, final_path)
    
    with open(MISSION_DIR / "candidate_3_training_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "pre_val_loss": pre_loss,
            "best_val_loss": best_val_loss,
            "history": history
        }, f, indent=2)
        
    print(f"Candidate 3 training completed. Results written to {MISSION_DIR / 'candidate_3_training_results.json'}")

if __name__ == "__main__":
    main()
