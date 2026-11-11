"""CR-EXP-16A Controlled Training Engine
Executes controlled arms:
- Arm A: Full-sequence loss control (Reproduction)
- Arm B: Target-only loss masking
- Arm C: Target-only loss with explicit copy serialization
All arms start strictly from canonical Stage B checkpoint (hash 96357705a...).
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
CKPT_DIR = CR16A_DIR / "checkpoints"

class MaskedCopyDataset(Dataset):
    def __init__(self, data_path, tokenizer, mask_prompt=True, max_seq_len=128):
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.mask_prompt = mask_prompt
        self.max_seq_len = max_seq_len
        self.items = []
        
        for s in self.samples:
            prompt_ids = self.tokenizer.encode(s["prompt"])
            target_ids = self.tokenizer.encode(s["target"])
            full_ids = prompt_ids + target_ids
            
            if len(full_ids) > self.max_seq_len:
                full_ids = full_ids[:self.max_seq_len]
                
            input_ids = full_ids[:-1]
            labels = full_ids[1:]
            
            if self.mask_prompt:
                # Mask prompt tokens with -100 so loss is computed ONLY on target tokens
                prompt_len = len(prompt_ids)
                # First target token is predicted from input_ids[prompt_len - 1]
                # Its label is labels[prompt_len - 1]
                masked_labels = []
                for idx, lbl in enumerate(labels):
                    if idx < prompt_len - 1:
                        masked_labels.append(-100)
                    else:
                        masked_labels.append(lbl)
                labels = masked_labels
                
            self.items.append((input_ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        inp, lbl = self.items[idx]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(lbl, dtype=torch.long)

def collate_fn(batch):
    max_len = max(len(x[0]) for x in batch)
    padded_inputs = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    
    for i, (inp, lbl) in enumerate(batch):
        padded_inputs[i, :len(inp)] = inp
        padded_labels[i, :len(lbl)] = lbl
        
    return padded_inputs, padded_labels

def evaluate_val_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            valid = (targets != -100).sum().item()
            total_loss += loss.item() * valid
            total_tokens += valid
    return total_loss / max(1, total_tokens)

def train_arm(arm_name, mask_prompt, steps=200, lr=1e-4, device=torch.device("cpu")):
    print(f"\n=======================================================")
    print(f"TRAINING ARM: {arm_name} (mask_prompt={mask_prompt}, steps={steps}, lr={lr})")
    print(f"=======================================================")
    
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    train_ds = MaskedCopyDataset(DATA_DIR / "cr16a_train.json", tokenizer, mask_prompt=mask_prompt)
    val_ds = MaskedCopyDataset(DATA_DIR / "cr16a_val.json", tokenizer, mask_prompt=mask_prompt)
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, hidden_dim=512)
    model = ChakrMicro(cfg)
    st = torch.load(STAGE_B_CKPT, map_location="cpu", weights_only=False)
    st_dict = st["model_state_dict"] if "model_state_dict" in st else st
    model.load_state_dict(st_dict, strict=True)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    init_val = evaluate_val_loss(model, val_loader, device)
    print(f"[{arm_name}] Initial Val Loss: {init_val:.4f}")
    
    history = []
    best_val = init_val
    arm_dir = CKPT_DIR / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    
    train_iter = iter(train_loader)
    model.train()
    for step in range(1, steps + 1):
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)
            
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 25 == 0:
            val_loss = evaluate_val_loss(model, val_loader, device)
            print(f"[{arm_name}] Step {step}/{steps} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
            history.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model_state_dict": model.state_dict(), "step": step, "val_loss": val_loss}, arm_dir / f"{arm_name}_best.pt")
            model.train()
            
    torch.save({"model_state_dict": model.state_dict(), "step": step, "val_loss": val_loss}, arm_dir / f"{arm_name}_final.pt")
    with open(arm_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[{arm_name}] Finished. Best Val Loss: {best_val:.4f}")
    return arm_dir / f"{arm_name}_best.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Arm A: Full-sequence loss control
    train_arm("arm_a_full_seq", mask_prompt=False, steps=150, lr=5e-5, device=device)
    
    # Arm B: Target-only loss masking
    train_arm("arm_b_target_masked", mask_prompt=True, steps=150, lr=1e-4, device=device)

if __name__ == "__main__":
    main()

