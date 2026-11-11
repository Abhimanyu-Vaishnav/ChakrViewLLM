"""CR-EXP-16B 2x2 Factorial Training Engine
Executes controlled training for:
- Arm A: Full Loss + Single Template
- Arm B: Target-Only Loss + Single Template
- Arm C: Full Loss + Multi-Template
- Arm D: Target-Only Loss + Multi-Template

Across independent random seeds (e.g. seeds 42, 101).
All hyper-parameters (steps=300, batch_size=16, lr=3e-4, Stage B parent) are 100% MATCHED.
"""
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
DATA_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16b_copying_generalization/data"
CKPT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16b_copying_generalization/checkpoints"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
STAGE_B_CKPT = REPO_ROOT / "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"

sys.path.insert(0, str(REPO_ROOT))
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

class FactorialDataset(Dataset):
    def __init__(self, data_path, tokenizer, target_only_loss=True, max_len=64):
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.target_only = target_only_loss
        self.max_len = max_len
        self.pad_id = 2

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt_text = item["prompt"]
        target_text = item["target"]
        
        prompt_ids = self.tokenizer.encode(prompt_text)
        target_ids = self.tokenizer.encode(target_text)
        
        input_ids = prompt_ids + target_ids
        
        if self.target_only:
            # Mask prompt tokens with -100
            labels = [-100] * len(prompt_ids) + list(target_ids)
        else:
            # Full sequence loss: all tokens labeled
            labels = list(prompt_ids) + list(target_ids)
            
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]
            
        pad_len = self.max_len - len(input_ids)
        input_ids = input_ids + [self.pad_id] * pad_len
        labels = labels + [-100] * pad_len
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_len": len(prompt_ids),
            "target_len": len(target_ids)
        }

def train_arm(arm_name, target_only, data_file, seed=42, total_steps=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==========================================")
    print(f"TRAINING {arm_name} | Target-Only: {target_only} | Seed: {seed} | Steps: {total_steps}")
    print(f"==========================================")
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))
    
    # Load canonical Stage B checkpoint
    ckpt = torch.load(STAGE_B_CKPT, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = ChakrMicro(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    train_dataset = FactorialDataset(DATA_DIR / data_file, tokenizer, target_only_loss=target_only)
    val_dataset = FactorialDataset(DATA_DIR / "factorial_val.json", tokenizer, target_only_loss=target_only)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    arm_ckpt_dir = CKPT_DIR / f"{arm_name}_seed{seed}"
    arm_ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    step = 0
    best_val_loss = float("inf")
    history = []
    
    model.train()
    while step < total_steps:
        for batch in train_loader:
            step += 1
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids[:, :-1])
            if isinstance(logits, tuple):
                logits = logits[0]
            targets = labels[:, 1:].contiguous()
            
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if step % 25 == 0:
                print(f"Step {step:3d}/{total_steps} | Train Loss: {loss.item():.4f}")
                
            if step % 50 == 0 or step == total_steps:
                model.eval()
                val_loss = 0.0
                val_tokens = 0
                correct_target_tokens = 0
                total_target_tokens = 0
                first_target_correct = 0
                total_val_samples = 0
                
                with torch.no_grad():
                    for vbatch in val_loader:
                        v_ids = vbatch["input_ids"].to(device)
                        v_labels = vbatch["labels"].to(device)
                        p_lens = vbatch["prompt_len"]
                        
                        v_logits = model(v_ids[:, :-1])
                        if isinstance(v_logits, tuple):
                            v_logits = v_logits[0]
                        v_targets = v_labels[:, 1:].contiguous()
                        
                        v_loss = criterion(v_logits.reshape(-1, v_logits.size(-1)), v_targets.reshape(-1))
                        val_loss += v_loss.item() * (v_targets != -100).sum().item()
                        val_tokens += (v_targets != -100).sum().item()
                        
                        preds = v_logits.argmax(dim=-1)
                        mask = (v_targets != -100)
                        correct_target_tokens += ((preds == v_targets) & mask).sum().item()
                        total_target_tokens += mask.sum().item()
                        
                        for b_idx in range(v_ids.size(0)):
                            p_len = p_lens[b_idx].item()
                            pred_first = preds[b_idx, p_len - 1].item()
                            target_first = v_targets[b_idx, p_len - 1].item()
                            if pred_first == target_first:
                                first_target_correct += 1
                            total_val_samples += 1
                            
                avg_val_loss = val_loss / max(1, val_tokens)
                target_acc = correct_target_tokens / max(1, total_target_tokens)
                first_acc = first_target_correct / max(1, total_val_samples)
                
                print(f"[{arm_name} s{seed}] Step {step} | Val Loss: {avg_val_loss:.4f} | Target Acc: {target_acc*100:.2f}% | First Token Acc: {first_acc*100:.2f}%")
                
                history.append({
                    "step": step,
                    "train_loss": loss.item(),
                    "val_loss": avg_val_loss,
                    "target_acc": target_acc,
                    "first_token_acc": first_acc
                })
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_ckpt_path = arm_ckpt_dir / f"{arm_name}_best.pt"
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "model_config": config,
                        "val_loss": avg_val_loss,
                        "history": history
                    }, best_ckpt_path)
                    
                model.train()
                
            if step >= total_steps:
                break

    final_ckpt_path = arm_ckpt_dir / f"{arm_name}_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "val_loss": avg_val_loss,
        "history": history
    }, final_ckpt_path)
    
    with open(arm_ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Finished {arm_name} s{seed}. Saved to {arm_ckpt_dir}")

def run_all_factorial():
    arms = [
        ("arm_a_full_single", False, "factorial_single_template_train.json"),
        ("arm_b_target_single", True, "factorial_single_template_train.json"),
        ("arm_c_full_multi", False, "factorial_multi_template_train.json"),
        ("arm_d_target_multi", True, "factorial_multi_template_train.json"),
    ]
    
    seeds = [42, 101]
    for arm_name, target_only, data_file in arms:
        for seed in seeds:
            train_arm(arm_name, target_only, data_file, seed=seed, total_steps=300)

if __name__ == "__main__":
    run_all_factorial()

