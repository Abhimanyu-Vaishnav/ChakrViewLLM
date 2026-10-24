"""
train_pointer.py
ChakrView Experiment 15 — Field-Conditioned Pointer Attention Training.

Trains FieldConditionedSpanPointer:
- 100% frozen ChakrMicro 3.44M backbone.
- Trains only:
    1. field_embedding: [10, 192]
    2. query_start, key_start: 192 -> 192
    3. query_end, key_end: 192 -> 192
- Total trainable parameters: ~149,376.
- Cross-entropy loss on start and end logits.
- Validates on pointer_val.jsonl at each epoch and saves checkpoints/pointer_best.pt.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP15_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP15_DIR / "data"
CHECKPOINT_DIR = EXP15_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

from experiments.research_v09_conversation_sft.exp03_field_conditioned_pointer.pointer_attention import (
    FieldConditionedSpanPointer,
    build_field_conditioned_pointer
)

class PointerDataset(Dataset):
    def __init__(self, data_path: Path):
        self.records = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

def collate_fn(batch: List[Dict[str, Any]]):
    max_len = max(len(b["input_ids"]) for b in batch)
    
    input_ids_list = []
    candidate_mask_list = []
    field_ids_list = []
    start_pos_list = []
    end_pos_list = []
    example_ids = []

    for b in batch:
        cur_len = len(b["input_ids"])
        pad_len = max_len - cur_len
        # 0 is BOS/pad token
        padded_ids = b["input_ids"] + [0] * pad_len
        # mask is true for context, false for pad tokens
        # Position 0 is eligible for abstention
        mask = [True] * cur_len + [False] * pad_len
        
        input_ids_list.append(padded_ids)
        candidate_mask_list.append(mask)
        field_ids_list.append(b["field_id"])
        start_pos_list.append(b["start_pos"])
        end_pos_list.append(b["end_pos"])
        example_ids.append(b["example_id"])

    return {
        "example_ids": example_ids,
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "candidate_mask": torch.tensor(candidate_mask_list, dtype=torch.bool),
        "field_ids": torch.tensor(field_ids_list, dtype=torch.long),
        "start_pos": torch.tensor(start_pos_list, dtype=torch.long),
        "end_pos": torch.tensor(end_pos_list, dtype=torch.long),
    }

def evaluate_val(model: FieldConditionedSpanPointer, val_loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_exact_spans = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            candidate_mask = batch["candidate_mask"].to(device)
            field_ids = batch["field_ids"].to(device)
            start_targets = batch["start_pos"].to(device)
            end_targets = batch["end_pos"].to(device)

            start_logits, end_logits = model(input_ids, field_ids, candidate_mask)
            
            loss_start = criterion(start_logits, start_targets)
            loss_end = criterion(end_logits, end_targets)
            loss = (loss_start + loss_end) / 2.0

            total_loss += loss.item() * input_ids.size(0)

            # Joint decoding
            for i in range(input_ids.size(0)):
                pred_s, pred_e, _, _ = model.predict_span(
                    input_ids[i], field_ids[i].item(), candidate_mask[i]
                )
                gold_s = start_targets[i].item()
                gold_e = end_targets[i].item()
                if pred_s == gold_s and pred_e == gold_e:
                    total_exact_spans += 1
                total_samples += 1

    avg_loss = total_loss / max(1, total_samples)
    exact_acc = total_exact_spans / max(1, total_samples)
    return avg_loss, exact_acc

def train_pointer():
    print("=" * 70)
    print("CHAKRVIEW EXP15: FIELD-CONDITIONED POINTER TRAINING")
    print("=" * 70)

    torch.manual_seed(42)
    device = torch.device("cpu")

    train_dataset = PointerDataset(DATA_DIR / "pointer_train.jsonl")
    val_dataset = PointerDataset(DATA_DIR / "pointer_val.jsonl")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    print(f"Loaded Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples")

    model = build_field_conditioned_pointer(device=str(device))
    
    # Verify parameter counts
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Frozen Backbone Params: {num_frozen:,}")
    print(f"Trainable Pointer Params: {num_trainable:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_val_loss = float("inf")
    history = []

    num_epochs = 12
    step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            step += 1
            input_ids = batch["input_ids"].to(device)
            candidate_mask = batch["candidate_mask"].to(device)
            field_ids = batch["field_ids"].to(device)
            start_targets = batch["start_pos"].to(device)
            end_targets = batch["end_pos"].to(device)

            optimizer.zero_grad()
            start_logits, end_logits = model(input_ids, field_ids, candidate_mask)

            loss_start = criterion(start_logits, start_targets)
            loss_end = criterion(end_logits, end_targets)
            loss = (loss_start + loss_end) / 2.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, val_acc = evaluate_val(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch:2d}/{num_epochs:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Exact Span Acc: {val_acc * 100:.2f}%")

        record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_exact_span_acc": val_acc
        }
        history.append(record)

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            ckpt_path = CHECKPOINT_DIR / "pointer_best.pt"
            torch.save({
                "epoch": epoch,
                "step": step,
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "field_embedding": model.field_embedding.state_dict(),
                "query_start": model.query_start.state_dict(),
                "key_start": model.key_start.state_dict(),
                "query_end": model.query_end.state_dict(),
                "key_end": model.key_end.state_dict(),
                "d_model": model.d_model,
                "num_fields": model.num_fields
            }, ckpt_path)
            print(f"  --> Saved new best checkpoint (Val Acc: {val_acc*100:.2f}%) to {ckpt_path.name}")

    with open(CHECKPOINT_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Training completed. Best Val Acc: {best_val_acc * 100:.2f}%")

if __name__ == "__main__":
    train_pointer()
