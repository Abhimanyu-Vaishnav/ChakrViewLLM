"""
CR-EXP-10 Candidate Training Engine
Supports Arm A (Balanced) and Arm B (Targeted Gap-Closure)
Fine-tunes from verified Candidate 3 immutable copy.
"""

import sys
import json
import random
import argparse
from pathlib import Path
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
MISSION_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_10_readiness"
DATA_DIR = MISSION_DIR / "data"
INIT_CKPT = MISSION_DIR / "checkpoints" / "baseline_cand3" / "cr_exp_09_candidate_3_best_val_copy.pt"

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

def train_arm(arm_name, train_file, total_steps=250, lr=2.5e-5):
    print(f"=== CR-EXP-10: Training Arm '{arm_name}' ({train_file.name}) ===")
    random.seed(42)
    torch.manual_seed(42)
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    with open(train_file, "r", encoding="utf-8") as f:
        train_samples = json.load(f)
    with open(DATA_DIR / "val.json", "r", encoding="utf-8") as f:
        val_samples = json.load(f)
        
    out_ckpt_dir = MISSION_DIR / "checkpoints" / f"candidate_{arm_name}"
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    model = load_parent_model()
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    
    pre_val = evaluate_val_loss(model, val_samples, tok)
    print(f"Pre-training Val Loss: {pre_val:.4f}")
    
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
            print(f"Step {step:3d}/{total_steps} | Loss: {loss.item():.4f} | GradNorm: {gnorm.item():.4f} | Val Loss: {val_loss:.4f}")
            history.append({
                "step": step,
                "train_loss": round(loss.item(), 4),
                "grad_norm": round(gnorm.item(), 4),
                "val_loss": round(val_loss, 4)
            })
            
            if val_loss < best_val:
                best_val = val_loss
                best_path = out_ckpt_dir / f"cr_exp_10_{arm_name}_best_val.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "model_class": "ChakrMicro",
                    "training_arm": f"cr_exp_10_{arm_name}"
                }, best_path)
                print(f"  --> Saved new best validation checkpoint (Val Loss: {val_loss:.4f})")
                
    final_path = out_ckpt_dir / f"cr_exp_10_{arm_name}_final.pt"
    torch.save({
        "step": total_steps,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "model_class": "ChakrMicro",
        "training_arm": f"cr_exp_10_{arm_name}_final"
    }, final_path)
    
    res_path = MISSION_DIR / f"{arm_name}_training_results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump({
            "arm": arm_name,
            "pre_val": pre_val,
            "best_val": best_val,
            "history": history
        }, f, indent=2)
    print(f"Arm {arm_name} finished. Best Val Loss: {best_val:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, required=True, choices=["arm_a", "arm_b"])
    args = parser.parse_args()
    
    t_file = DATA_DIR / f"train_{args.arm}.json"
    train_arm(args.arm, t_file)

