"""CR-EXP-16A Arm D: Generalization-Preserved Induction Head Pretraining
Trains ChakrMicro starting strictly from Stage B checkpoint (96357705a...) on:
- High-diversity single-token and sequence copying
- Cross-category representation in 'Repeat the name' format (prevents category prior collapse)
- Target-only loss masking (-100 on prompt tokens)
- 500 steps, batch size 16, lr=3e-4, grad clip 1.0
"""
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
CR16A_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16a_copying_recovery"
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
STAGE_B_CKPT = REPO_ROOT / "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"
DATA_DIR = CR16A_DIR / "data"
CKPT_DIR = CR16A_DIR / "checkpoints" / "arm_d_generalization"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

class CopyDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=64):
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
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
        labels = [-100] * len(prompt_ids) + list(target_ids)
        
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

def train_arm_d():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Arm D on device: {device}")
    
    tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))
    
    print(f"Loading Stage B baseline checkpoint from: {STAGE_B_CKPT}")
    ckpt = torch.load(STAGE_B_CKPT, map_location=device, weights_only=False)
    config = ckpt["config"]
    
    model = ChakrMicro(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    train_dataset = CopyDataset(DATA_DIR / "cr16a_arm_d_train.json", tokenizer)
    val_dataset = CopyDataset(DATA_DIR / "cr16a_curriculum_val.json", tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    total_steps = 500
    eval_interval = 50
    step = 0
    best_val_loss = float("inf")
    history = []
    
    model.train()
    print(f"Starting Arm D training ({total_steps} steps, 5000 diverse samples)...")
    
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
                print(f"Step {step:4d}/{total_steps} | Train Loss: {loss.item():.4f}")
                
            if step % eval_interval == 0 or step == total_steps:
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
                
                print(f"=== EVAL Step {step} | Val Loss: {avg_val_loss:.4f} | Target Acc: {target_acc*100:.2f}% | First Token Acc: {first_acc*100:.2f}% ===")
                
                history.append({
                    "step": step,
                    "train_loss": loss.item(),
                    "val_loss": avg_val_loss,
                    "target_acc": target_acc,
                    "first_token_acc": first_acc
                })
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_ckpt_path = CKPT_DIR / "arm_d_generalization_best.pt"
                    torch.save({
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "model_config": config,
                        "val_loss": avg_val_loss,
                        "history": history
                    }, best_ckpt_path)
                    print(f"Saved new best checkpoint to {best_ckpt_path}")
                    
                model.train()
                
            if step >= total_steps:
                break

    final_ckpt_path = CKPT_DIR / "arm_d_generalization_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "val_loss": avg_val_loss,
        "history": history
    }, final_ckpt_path)
    print(f"Saved final checkpoint to {final_ckpt_path}")
    
    with open(CKPT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    train_arm_d()

