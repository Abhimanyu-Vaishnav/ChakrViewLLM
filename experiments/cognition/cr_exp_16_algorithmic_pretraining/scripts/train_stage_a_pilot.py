"""CR-EXP-16 Stage A Pilot Trainer & Evaluator
Trains a small isolated pilot model on Stage A Sequence Copying initialized from Stage B checkpoint.
Checks loss progression and evaluates on a subset of copying tasks to measure improvement.
"""
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
EXP16_DIR = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining"
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
STAGE_B_CKPT = REPO_ROOT / "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"
TRAIN_PATH = EXP16_DIR / "data/stage_a_train.json"
VAL_PATH = EXP16_DIR / "data/stage_a_val.json"
CKPT_DIR = EXP16_DIR / "checkpoints/pilot"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

class CopyDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len=128):
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.encoded = []
        for s in self.samples:
            ids = self.tokenizer.encode(s["text"])
            if len(ids) > self.max_seq_len:
                ids = ids[:self.max_seq_len]
            self.encoded.append(ids)

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded[idx], dtype=torch.long)

def collate_fn(batch):
    max_len = max(len(x) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
    return padded

def evaluate_val_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            non_pad = (targets != 0).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
    return total_loss / max(1, total_tokens)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Pilot using device: {device}")
    
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    train_ds = CopyDataset(TRAIN_PATH, tokenizer)
    val_ds = CopyDataset(VAL_PATH, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, hidden_dim=512)
    model = ChakrMicro(cfg)
    st = torch.load(STAGE_B_CKPT, map_location="cpu", weights_only=False)
    st_dict = st["model_state_dict"] if "model_state_dict" in st else st
    model.load_state_dict(st_dict, strict=True)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    init_val = evaluate_val_loss(model, val_loader, device)
    print(f"[Pilot] Initial Validation Loss: {init_val:.4f}")
    
    steps = 100
    model.train()
    step = 0
    history = []
    best_val = init_val
    
    train_iter = iter(train_loader)
    while step < steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        step += 1
        if step % 20 == 0:
            val_loss = evaluate_val_loss(model, val_loader, device)
            print(f"[Pilot] Step {step}/{steps} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
            history.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss})
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model_state_dict": model.state_dict(), "step": step, "val_loss": val_loss}, CKPT_DIR / "stage_a_pilot_best.pt")
            model.train()
            
    final_ckpt = CKPT_DIR / "stage_a_pilot_final.pt"
    torch.save({"model_state_dict": model.state_dict(), "step": step, "val_loss": val_loss}, final_ckpt)
    with open(CKPT_DIR / "pilot_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    print(f"[Pilot] Completed. Best Val Loss: {best_val:.4f}")

if __name__ == "__main__":
    main()

