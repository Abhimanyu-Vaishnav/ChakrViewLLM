import os
import sys
import json
import time
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW

from chakrview.brain.model import ModelConfig, ChakrMicro, initialize_weights
from scripts.run_m3_evaluations import load_tokenizer_artifacts

def load_data(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def collate_dialogue(dialogue, tokenizer, max_len=256):
    prompt = dialogue["prompt"]
    target = dialogue["target"]
    
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    target_ids = tokenizer.encode(target, add_bos=False, add_eos=True)
    
    # Assert target has at least 1 actual token + EOS
    assert len(target_ids) >= 2, f"Target too short: {target_ids} for target: '{target}'"
    
    input_ids = prompt_ids + target_ids
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        
    labels = [-100] * len(prompt_ids) + target_ids
    if len(labels) > max_len:
        labels = labels[:max_len]
        
    return torch.tensor([input_ids], dtype=torch.long), torch.tensor([labels], dtype=torch.long)

def train_recovered_candidate():
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    train_records = load_data("experiments/cognition/ucip_01r/data/ucip01r_train.jsonl")
    val_records = load_data("experiments/cognition/ucip_01r/data/ucip01r_validation.jsonl")
    
    print(f"Loaded {len(train_records)} clean train records, {len(val_records)} clean val records.")
    
    # 13M Architecture
    cfg = ModelConfig(
        vocab_size=4096,
        d_model=384,
        n_layers=7,
        n_heads=6,
        hidden_dim=896,
        max_seq_len=512,
        tie_embeddings=True
    )
    
    torch.manual_seed(42)
    model = ChakrMicro(cfg)
    initialize_weights(model, cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized ChakrMedium-13M model. Total unique parameters: {total_params:,}")
    
    # Warm-start embeddings from v0.11
    v011_path = REPO_ROOT / "releases/v0.11_conversational_model/chakrview_v011_model.pt"
    parent_ckpt = torch.load(v011_path, map_location="cpu")
    parent_sd = parent_ckpt.get("model_state_dict") or parent_ckpt
    with torch.no_grad():
        model.embedding.weight[:, :192].copy_(parent_sd["embedding.weight"])
    print("Warm-started embedding subspace from v0.11.")
    
    optimizer = AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.01, betas=(0.9, 0.98))
    
    save_dir = Path("experiments/cognition/ucip_01r/training")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.train()
    best_val_loss = float("inf")
    best_step = 0
    total_steps = 400
    batch_size = 4
    grad_accum_steps = 2
    step = 0
    train_iter = 0
    
    t0 = time.time()
    while step < total_steps:
        optimizer.zero_grad()
        accum_loss = 0.0
        
        for _ in range(grad_accum_steps):
            for _ in range(batch_size):
                rec = train_records[train_iter % len(train_records)]
                train_iter += 1
                inp, tgt = collate_dialogue(rec, tok)
                if inp.shape[1] < 2:
                    continue
                logits = model(inp[:, :-1])
                shift_logits = logits.view(-1, cfg.vocab_size)
                shift_targets = tgt[:, 1:].contiguous().view(-1)
                
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(shift_logits, shift_targets) / (grad_accum_steps * batch_size)
                loss.backward()
                accum_loss += loss.item() * (grad_accum_steps * batch_size)
                
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        
        if step % 50 == 0 or step == total_steps:
            model.eval()
            val_loss = 0.0
            val_tokens = 0
            val_correct = 0
            
            with torch.no_grad():
                for rec in val_records[:50]:
                    inp, tgt = collate_dialogue(rec, tok)
                    logits = model(inp[:, :-1])
                    shift_logits = logits.view(-1, cfg.vocab_size)
                    shift_targets = tgt[:, 1:].contiguous().view(-1)
                    
                    mask = (shift_targets != -100)
                    if mask.sum() > 0:
                        l = nn.CrossEntropyLoss(ignore_index=-100)(shift_logits, shift_targets)
                        val_loss += l.item() * mask.sum().item()
                        val_tokens += mask.sum().item()
                        preds = shift_logits[mask].argmax(dim=-1)
                        val_correct += (preds == shift_targets[mask]).sum().item()
                        
            mean_val_loss = val_loss / max(1, val_tokens)
            val_acc = val_correct / max(1, val_tokens)
            
            print(f"Step {step}/{total_steps} | Train Loss: {accum_loss:.4f} | Val Loss: {mean_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_step = step
                best_path = save_dir / "ucip01r_13m_recovered_best.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "val_acc": val_acc,
                    "config": cfg.__dict__
                }, best_path)
                
            model.train()
            
    print(f"\nRecovered Training Completed in {time.time() - t0:.1f}s. Best Val Loss: {best_val_loss:.4f} at Step {best_step}")

if __name__ == "__main__":
    train_recovered_candidate()
