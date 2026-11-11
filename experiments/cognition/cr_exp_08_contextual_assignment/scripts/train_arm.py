"""
CR-EXP-08 Autonomous Training Engine
Supports Arms A, B, and C under strictly controlled identical conditions:
- Same initialization checkpoint (conversational_pilot_best_val.pt)
- Same dataset order (seed 42)
- Same effective batch size (8)
- Same AdamW optimizer & learning rate (5e-5)
- Same gradient clipping (0.5)
- Max 150 optimizer updates
- Saves initial, best validation, and final checkpoints independently.
- Output logs & metrics written to results/arm_{a,b,c}_results.json.
"""

import sys
import os
import json
import random
import argparse
import hashlib
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
DATA_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_08_contextual_assignment" / "data"
EXP_DIR = REPO_ROOT / "experiments" / "cognition" / "cr_exp_08_contextual_assignment"

INIT_CKPT_PATH = REPO_ROOT / "experiments/cognition/autonomous_12h/conversational_pilot/checkpoints/conversational_pilot_best_val.pt"
EXPECTED_INIT_SHA = "ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd"

def load_initial_model():
    h = hashlib.sha256(INIT_CKPT_PATH.read_bytes()).hexdigest()
    assert h == EXPECTED_INIT_SHA, f"Initialization hash mismatch! Expected {EXPECTED_INIT_SHA}, got {h}"
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    ckpt = torch.load(INIT_CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    return model

def prepare_batch(samples, tok, arm):
    """
    Tokenizes samples and builds tensors for autoregressive loss and pairwise ranking.
    """
    input_ids_list = []
    loss_masks_list = []
    token_weights_list = []
    pairwise_triplets = [] # (prompt_ids, pos_ids, neg_ids)
    
    for s in samples:
        prompt = s["prompt"]
        completion = s["completion"]
        is_relational = s["is_relational"]
        
        enc_p = tok.encode(prompt)
        enc_full = tok.encode(prompt + completion)
        
        input_ids_list.append(enc_full)
        
        # Loss mask (1 on completion, 0 on prompt)
        m = [0.0] * len(enc_p) + [1.0] * (len(enc_full) - len(enc_p))
        loss_masks_list.append(m)
        
        # Token weights: 2.0 on relational completion, 1.0 on conversational completion, 0.0 on prompt
        w_val = 2.0 if is_relational else 1.0
        w = [0.0] * len(enc_p) + [w_val] * (len(enc_full) - len(enc_p))
        token_weights_list.append(w)
        
        if arm == "arm_c" and is_relational and s.get("negative_candidate"):
            neg = s["negative_candidate"]
            enc_neg_full = tok.encode(prompt + neg)
            pos_ids = enc_full[len(enc_p):]
            neg_ids = enc_neg_full[len(enc_p):]
            if len(pos_ids) == len(neg_ids):
                pairwise_triplets.append((enc_p, pos_ids, neg_ids))

    # Pad sequences
    max_len = max(len(ids) for ids in input_ids_list)
    padded_inputs = []
    padded_targets = []
    padded_masks = []
    padded_weights = []
    
    for ids, m, w in zip(input_ids_list, loss_masks_list, token_weights_list):
        pad_len = max_len - len(ids)
        padded_inputs.append(ids + [0] * pad_len)
        padded_targets.append(ids + [0] * pad_len)
        padded_masks.append(m + [0.0] * pad_len)
        padded_weights.append(w + [0.0] * pad_len)
        
    return {
        "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
        "target_ids": torch.tensor(padded_targets, dtype=torch.long),
        "loss_mask": torch.tensor(padded_masks, dtype=torch.float),
        "token_weights": torch.tensor(padded_weights, dtype=torch.float),
        "pairwise_triplets": pairwise_triplets
    }

def evaluate_val_loss(model, val_samples, tok, arm, batch_size=8):
    model.eval()
    total_loss = 0.0
    total_rel_loss = 0.0
    total_conv_loss = 0.0
    rel_count = 0
    conv_count = 0
    batches = [val_samples[i : i + batch_size] for i in range(0, len(val_samples), batch_size)]
    
    with torch.no_grad():
        for b_samples in batches:
            b_data = prepare_batch(b_samples, tok, arm)
            inp = b_data["input_ids"]
            tgt = b_data["target_ids"]
            logits = model(inp)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = tgt[:, 1:].contiguous()
            shift_mask = b_data["loss_mask"][:, 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            t_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1)).view(shift_targets.size())
            
            m_loss = (t_losses * shift_mask).sum() / (shift_mask.sum() + 1e-8)
            total_loss += m_loss.item()
            
    avg_loss = total_loss / len(batches) if batches else 0.0
    return avg_loss

def train_arm(arm_name, max_updates=150, lr=5e-5, effective_bs=8, eval_interval=25):
    print(f"\n==========================================")
    print(f"Starting Training: CR-EXP-08 [{arm_name.upper()}]")
    print(f"==========================================")
    
    random.seed(42)
    torch.manual_seed(42)
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    with open(DATA_DIR / "train.json", encoding="utf-8") as f:
        train_samples = json.load(f)
    with open(DATA_DIR / "val.json", encoding="utf-8") as f:
        val_samples = json.load(f)
        
    ckpt_dir = EXP_DIR / "checkpoints" / arm_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    model = load_initial_model()
    # Save independent initialization checkpoint
    init_save_path = ckpt_dir / f"{arm_name}_initial.pt"
    torch.save(model.state_dict(), init_save_path)
    print(f"Saved initial model copy to {init_save_path}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Pre-training validation loss
    initial_val_loss = evaluate_val_loss(model, val_samples, tok, arm_name)
    print(f"Pre-training Val Loss: {initial_val_loss:.4f}")
    
    best_val_loss = initial_val_loss
    best_step = 0
    best_ckpt_path = ckpt_dir / f"{arm_name}_best_val.pt"
    torch.save(model.state_dict(), best_ckpt_path)
    
    train_history = []
    step = 0
    sample_idx = 0
    model.train()
    
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    
    while step < max_updates:
        # Collect effective_bs samples
        batch_samples = []
        for _ in range(effective_bs):
            batch_samples.append(train_samples[sample_idx % len(train_samples)])
            sample_idx += 1
            
        b_data = prepare_batch(batch_samples, tok, arm_name)
        inp = b_data["input_ids"]
        tgt = b_data["target_ids"]
        
        optimizer.zero_grad()
        logits = model(inp)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = tgt[:, 1:].contiguous()
        
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1)).view(shift_targets.size())
        
        if arm_name == "arm_a":
            # Standard SFT Loss
            shift_mask = b_data["loss_mask"][:, 1:].contiguous()
            loss = (token_losses * shift_mask).sum() / (shift_mask.sum() + 1e-8)
            lm_loss_val = loss.item()
            pair_loss_val = 0.0
            
        elif arm_name == "arm_b":
            # Relational Token Weighted Loss (weight 2.0 on relational targets)
            shift_weights = b_data["token_weights"][:, 1:].contiguous()
            loss = (token_losses * shift_weights).sum() / (shift_weights.sum() + 1e-8)
            lm_loss_val = loss.item()
            pair_loss_val = 0.0
            
        elif arm_name == "arm_c":
            # Counterfactual Pairwise Assignment Objective
            shift_mask = b_data["loss_mask"][:, 1:].contiguous()
            lm_loss = (token_losses * shift_mask).sum() / (shift_mask.sum() + 1e-8)
            lm_loss_val = lm_loss.item()
            
            # Pairwise loss over relational samples in the batch
            triplets = b_data["pairwise_triplets"]
            if triplets:
                pair_losses = []
                for p_ids, pos_ids, neg_ids in triplets:
                    prompt_len = len(p_ids)
                    cand_len = len(pos_ids)
                    
                    full_p = torch.tensor([p_ids + pos_ids], dtype=torch.long)
                    full_n = torch.tensor([p_ids + neg_ids], dtype=torch.long)
                    
                    l_p = model(full_p)
                    l_n = model(full_n)
                    
                    pred_p = l_p[0, prompt_len - 1 : prompt_len + cand_len - 1, :]
                    pred_n = l_n[0, prompt_len - 1 : prompt_len + cand_len - 1, :]
                    
                    lp_p = F.log_softmax(pred_p, dim=-1).gather(1, full_p[0, prompt_len:prompt_len+cand_len].unsqueeze(1)).squeeze(1)
                    lp_n = F.log_softmax(pred_n, dim=-1).gather(1, full_n[0, prompt_len:prompt_len+cand_len].unsqueeze(1)).squeeze(1)
                    
                    s_pos = lp_p.mean()
                    s_neg = lp_n.mean()
                    # softplus(0.5 - (s_pos - s_neg))
                    pair_losses.append(F.softplus(0.5 - (s_pos - s_neg)))
                    
                mean_pair_loss = torch.stack(pair_losses).mean()
                pair_loss_val = mean_pair_loss.item()
                loss = lm_loss + 0.2 * mean_pair_loss
            else:
                pair_loss_val = 0.0
                loss = lm_loss
                
        # Optimization step
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 0.5).item()
        optimizer.step()
        step += 1
        
        # Stop condition checks
        if not torch.isfinite(loss):
            print(f"STOP TRIGGERED: Non-finite loss at step {step}: {loss.item()}")
            break
            
        record = {
            "step": step,
            "total_loss": round(loss.item(), 4),
            "lm_loss": round(lm_loss_val, 4),
            "pair_loss": round(pair_loss_val, 4),
            "grad_norm": round(grad_norm, 4)
        }
        train_history.append(record)
        
        if step % 25 == 0 or step == max_updates:
            val_loss = evaluate_val_loss(model, val_samples, tok, arm_name)
            record["val_loss"] = round(val_loss, 4)
            print(f"Step {step:3d}/{max_updates} | Loss: {loss.item():.4f} (LM: {lm_loss_val:.4f}, Pair: {pair_loss_val:.4f}) | GradNorm: {grad_norm:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  --> New best validation checkpoint saved! (Val loss: {val_loss:.4f})")
            model.train()
            
    # Save final checkpoint
    final_ckpt_path = ckpt_dir / f"{arm_name}_final.pt"
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Training completed for {arm_name}. Final checkpoint saved to {final_ckpt_path}")
    print(f"Best Val Loss: {best_val_loss:.4f} at Step {best_step}")
    
    # Save results payload
    payload = {
        "arm": arm_name,
        "max_updates": max_updates,
        "learning_rate": lr,
        "effective_batch_size": effective_bs,
        "initial_val_loss": round(initial_val_loss, 4),
        "best_val_loss": round(best_val_loss, 4),
        "best_step": best_step,
        "final_train_loss": round(train_history[-1]["total_loss"], 4),
        "training_history": train_history,
        "checkpoint_hashes": {
            "initial": hashlib.sha256(init_save_path.read_bytes()).hexdigest(),
            "best_val": hashlib.sha256(best_ckpt_path.read_bytes()).hexdigest(),
            "final": hashlib.sha256(final_ckpt_path.read_bytes()).hexdigest()
        }
    }
    
    results_path = EXP_DIR / f"{arm_name}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=str, required=True, choices=["arm_a", "arm_b", "arm_c"])
    args = parser.parse_args()
    train_arm(args.arm)

