"""
CR-EXP-09 Candidate 1 Training Engine: Multi-Stream Curriculum Optimization
Trains Candidate 1 initialized from verified copy of CR-EXP-08 Arm B:
- Objective: Assistant-only cross-entropy with relational answer token weighting (weight 2.0 on relational targets, 1.0 on conversational & abstention sentences)
- Max 250 optimizer updates
- Learning rate: 3e-5 (conservative fine-tuning to preserve established single-hop representations)
- Batch size: 8
- AdamW, grad clipping 0.5
- Saves best validation checkpoint and outputs training log.
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
CKPT_DIR = MISSION_DIR / "checkpoints" / "candidate_1"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

INIT_CKPT = MISSION_DIR / "checkpoints" / "baseline" / "cr_exp_08_arm_b_best_val_copy.pt"
EXPECTED_SHA = "98bd76f99d43dedd10c31a1d290c95e452eba299f511ecf72e5ecc4075fb0956"

def load_initial_candidate():
    h = hashlib.sha256(INIT_CKPT.read_bytes()).hexdigest()
    assert h == EXPECTED_SHA, f"Hash mismatch on baseline! Expected {EXPECTED_SHA}, got {h}"
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
        
        # Weights: 2.0 on relational targets, 1.0 on conversational & abstention tokens, 0.0 on prompt
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
    print("=== CR-EXP-09: Training Candidate 1 (Multi-Stream Curriculum) ===")
    random.seed(42)
    torch.manual_seed(42)
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    with open(DATA_DIR / "train.json", encoding="utf-8") as f:
        train_samples = json.load(f)
    with open(DATA_DIR / "val.json", encoding="utf-8") as f:
        val_samples = json.load(f)
        
    model = load_initial_candidate()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    
    initial_val_loss = evaluate_val_loss(model, val_samples, tok)
    print(f"Pre-training Val Loss on Curriculum Dataset: {initial_val_loss:.4f}")
    
    best_val_loss = initial_val_loss
    best_step = 0
    best_ckpt_path = CKPT_DIR / "cr_exp_09_candidate_1_best_val.pt"
    torch.save(model.state_dict(), best_ckpt_path)
    
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    train_history = []
    max_updates = 200
    effective_bs = 8
    sample_idx = 0
    step = 0
    
    model.train()
    while step < max_updates:
        batch_samples = []
        for _ in range(effective_bs):
            batch_samples.append(train_samples[sample_idx % len(train_samples)])
            sample_idx += 1
            
        b_data = prepare_batch(batch_samples, tok)
        inp = b_data["input_ids"]
        tgt = b_data["target_ids"]
        weights = b_data["token_weights"][:, 1:].contiguous()
        
        optimizer.zero_grad()
        logits = model(inp)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = tgt[:, 1:].contiguous()
        
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1)).view(shift_targets.size())
        loss = (token_losses * weights).sum() / (weights.sum() + 1e-8)
        
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 0.5).item()
        optimizer.step()
        step += 1
        
        record = {
            "step": step,
            "loss": round(loss.item(), 4),
            "grad_norm": round(grad_norm, 4)
        }
        train_history.append(record)
        
        if step % 25 == 0 or step == max_updates:
            val_loss = evaluate_val_loss(model, val_samples, tok)
            record["val_loss"] = round(val_loss, 4)
            print(f"Step {step:3d}/{max_updates} | Loss: {loss.item():.4f} | GradNorm: {grad_norm:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  --> Saved new best validation checkpoint (Val Loss: {val_loss:.4f})")
            model.train()
            
    final_ckpt_path = CKPT_DIR / "cr_exp_09_candidate_1_final.pt"
    torch.save(model.state_dict(), final_ckpt_path)
    
    payload = {
        "candidate": "cr_exp_09_candidate_1",
        "parent": "cr_exp_08_arm_b_best_val_copy.pt",
        "parent_sha256": EXPECTED_SHA,
        "max_updates": max_updates,
        "learning_rate": 3e-5,
        "initial_val_loss": round(initial_val_loss, 4),
        "best_val_loss": round(best_val_loss, 4),
        "best_step": best_step,
        "final_train_loss": round(train_history[-1]["loss"], 4),
        "checkpoint_hashes": {
            "best_val": hashlib.sha256(best_ckpt_path.read_bytes()).hexdigest(),
            "final": hashlib.sha256(final_ckpt_path.read_bytes()).hexdigest()
        }
    }
    
    out_result = MISSION_DIR / "candidate_1_training_results.json"
    with open(out_result, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Candidate 1 training completed. Results written to {out_result}")

if __name__ == "__main__":
    main()

