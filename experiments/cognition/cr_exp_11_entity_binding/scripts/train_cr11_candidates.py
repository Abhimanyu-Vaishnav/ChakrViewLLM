"""
CR-EXP-11 Candidate Training Engine
Trains Candidate A (Balanced Anchoring) and Candidate B (Structured Entity Curriculum)
Starting Checkpoint: Immutable copy of verified CR-10 Arm B (8c15c2...)
Safeguards:
- Frozen parent immutable copy untouched
- Max steps: 250 per candidate
- Learning rate: 3e-5 with AdamW and grad clip 0.5
- Saves best validation checkpoint and final checkpoint
- Full training history logged to JSON
"""

import sys
import json
import random
from pathlib import Path
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
EXP_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_11_entity_binding"
DATA_DIR = EXP_DIR / "data"
INIT_CKPT = EXP_DIR / "checkpoints" / "baseline_arm_b" / "cr_exp_10_arm_b_best_val_copy.pt"

def load_parent_model():
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
        w_val = s.get("weight", 2.0)
        
        enc_p = tok.encode(p)
        enc_full = tok.encode(p + c)
        
        input_ids_list.append(enc_full)
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

def train_candidate(candidate_name, train_file, total_steps=250, lr=3e-5):
    print(f"\n=======================================================")
    print(f"Starting Training: CR-EXP-11 Candidate {candidate_name.upper()}")
    print(f"Data file: {train_file.name} | Steps: {total_steps} | LR: {lr}")
    print(f"=======================================================")
    
    random.seed(42)
    torch.manual_seed(42)
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    with open(train_file, "r", encoding="utf-8") as f:
        train_samples = json.load(f)
    with open(DATA_DIR / "val.json", "r", encoding="utf-8") as f:
        val_samples = json.load(f)
        
    out_dir = EXP_DIR / "checkpoints" / f"candidate_{candidate_name.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = load_parent_model()
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    
    pre_val = evaluate_val_loss(model, val_samples, tok)
    print(f"Pre-training Validation Loss: {pre_val:.4f}")
    
    best_val = pre_val
    batch_size = 8
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
            print(f"Step {step:3d}/{total_steps} | Train Loss: {loss.item():.4f} | GradNorm: {gnorm.item():.4f} | Val Loss: {val_loss:.4f}")
            history.append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "grad_norm": round(gnorm.item(), 4),
                "val_loss": round(val_loss, 4)
            })
            
            if val_loss < best_val:
                best_val = val_loss
                best_path = out_dir / f"cr_exp_11_candidate_{candidate_name.lower()}_best_val.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "model_class": "ChakrMicro",
                    "training_candidate": f"cr_exp_11_{candidate_name.lower()}"
                }, best_path)
                print(f"  [+] Saved new best validation checkpoint: {best_path.name} (Val Loss: {val_loss:.4f})")
                
    final_path = out_dir / f"cr_exp_11_candidate_{candidate_name.lower()}_final.pt"
    torch.save({
        "step": total_steps,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "model_class": "ChakrMicro",
        "training_candidate": f"cr_exp_11_{candidate_name.lower()}"
    }, final_path)
    print(f"Final checkpoint saved: {final_path.name}")
    
    results_path = EXP_DIR / f"candidate_{candidate_name.lower()}_training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "candidate": candidate_name,
            "train_file": str(train_file.name),
            "total_steps": total_steps,
            "best_val_loss": round(best_val, 4),
            "final_val_loss": round(val_loss, 4),
            "history": history
        }, f, indent=2)

def main():
    train_file_a = DATA_DIR / "train_candidate_a.json"
    train_file_b = DATA_DIR / "train_candidate_b.json"
    
    train_candidate("a", train_file_a, total_steps=250, lr=3e-5)
    train_candidate("b", train_file_b, total_steps=250, lr=3e-5)

if __name__ == "__main__":
    main()

