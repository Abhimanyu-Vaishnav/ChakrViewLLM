"""
train_conversation_model.py
ChakrView Experiment 13: Native Human Conversation SFT Training Runner.

Trains candidate checkpoints from the canonical Step-10000 parent checkpoint:
- Base architecture: ChakrMicro (3.44M params, ModelConfig(max_seq_len=512))
- Pure CPU execution with AdamW optimizer and cosine learning rate decay
- Strict prompt-masked causal loss (gradients computed strictly on assistant response tokens)
- Bounded steps (120 steps) to preserve CPU compute budget
- Saves candidate checkpoint isolated under checkpoints/
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import build_causal_sequence

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
CKPT_DIR = EXP_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_PARENT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
EXPECTED_PARENT_SHA = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

CANONICAL_TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
EXPECTED_VOCAB_SHA = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"
EXPECTED_MERGES_SHA = "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7"

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

class ConversationDataset:
    def __init__(self, jsonl_path: Path, tokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.records = []
        
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))
                    
        self.encoded_samples = []
        for item in self.records:
            prompt_str = item["prompt"] if item["prompt"].endswith("\n") else item["prompt"] + "\n"
            resp_str = item["response"].strip()
            
            p_toks = self.tokenizer.encode(prompt_str)
            r_toks = self.tokenizer.encode(resp_str)
            
            in_seq, tgt_seq = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100)
            
            # Truncate if exceeds max_seq_len
            if len(in_seq) > max_seq_len:
                in_seq = in_seq[:max_seq_len]
                tgt_seq = tgt_seq[:max_seq_len]
                
            self.encoded_samples.append((
                torch.tensor(in_seq, dtype=torch.long),
                torch.tensor(tgt_seq, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.encoded_samples)

    def get_batch(self, batch_size: int, rng) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = [rng.randint(0, len(self.encoded_samples) - 1) for _ in range(batch_size)]
        samples = [self.encoded_samples[i] for i in indices]
        
        max_len = max(s[0].shape[0] for s in samples)
        in_batch = torch.full((batch_size, max_len), fill_value=2, dtype=torch.long) # PAD_ID=2
        tgt_batch = torch.full((batch_size, max_len), fill_value=-100, dtype=torch.long)
        
        for b_idx, (in_s, tgt_s) in enumerate(samples):
            seq_len = in_s.shape[0]
            in_batch[b_idx, :seq_len] = in_s
            tgt_batch[b_idx, :seq_len] = tgt_s
            
        return in_batch, tgt_batch

def main():
    print("=" * 70)
    print("CHAKRVIEW EXPERIMENT 13: NATIVE CONVERSATION SFT TRAINING")
    print("=" * 70)
    
    # 1. Preflight integrity assertions
    print("\n[Preflight] Verifying Canonical Hashes...")
    actual_parent_sha = compute_sha256(CANONICAL_PARENT_PATH)
    assert actual_parent_sha == EXPECTED_PARENT_SHA, f"Parent mismatch: {actual_parent_sha}"
    
    actual_vocab_sha = compute_sha256(CANONICAL_TOK_DIR / "vocab.json")
    actual_merges_sha = compute_sha256(CANONICAL_TOK_DIR / "merges.json")
    assert actual_vocab_sha == EXPECTED_VOCAB_SHA, "Vocab mismatch"
    assert actual_merges_sha == EXPECTED_MERGES_SHA, "Merges mismatch"
    print("  [PASS] Canonical parent and tokenizer verified bit-identical.")
    
    # 2. Setup Tokenizer, Data, and Model
    tokenizer, _ = load_tokenizer_artifacts(CANONICAL_TOK_DIR, verify_checksums=False)
    
    train_dataset = ConversationDataset(DATA_DIR / "train.jsonl", tokenizer, max_seq_len=512)
    val_dataset = ConversationDataset(DATA_DIR / "val.jsonl", tokenizer, max_seq_len=512)
    print(f"  Loaded Train Dataset: {len(train_dataset)} examples")
    print(f"  Loaded Val Dataset:   {len(val_dataset)} examples")
    
    device = torch.device("cpu")
    model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    
    parent_ckpt = torch.load(CANONICAL_PARENT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(parent_ckpt["model_state_dict"], strict=True)
    print("  Loaded ChakrMicro from canonical Step-10000 checkpoint.")
    
    initial_weights = {name: param.detach().clone() for name, param in model.named_parameters()}
    
    # Hyperparameters
    learning_rate = 5e-5
    min_learning_rate = 5e-6
    weight_decay = 0.01
    grad_clip = 1.0
    batch_size = 4
    total_steps = 120
    val_interval = 20
    seed = 42
    
    import random
    rng = random.Random(seed)
    
    # Segregate decay / no-decay parameters
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optimizer = optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=learning_rate)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    print("\n[Training] Beginning 120 steps SFT on local CPU...")
    history = []
    best_val_loss = float("inf")
    best_ckpt_path = CKPT_DIR / "candidate_best.pt"
    
    start_time = time.time()
    
    for step in range(1, total_steps + 1):
        model.train()
        optimizer.zero_grad()
        
        # Cosine learning rate schedule
        progress = (step - 1) / max(1, total_steps - 1)
        curr_lr = min_learning_rate + 0.5 * (learning_rate - min_learning_rate) * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr
            
        in_batch, tgt_batch = train_dataset.get_batch(batch_size, rng)
        
        logits = model(in_batch)
        loss = criterion(logits.view(-1, 4096), tgt_batch.view(-1))
        
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"Step {step}: Loss is NaN or Inf!")
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        if step % val_interval == 0 or step == total_steps:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(5):
                    v_in, v_tgt = val_dataset.get_batch(batch_size, rng)
                    v_logits = model(v_in)
                    v_loss = criterion(v_logits.view(-1, 4096), v_tgt.view(-1))
                    val_losses.append(v_loss.item())
            mean_val_loss = sum(val_losses) / len(val_losses)
            
            # Compute parameter drift
            total_delta_sq = 0.0
            total_orig_sq = 0.0
            with torch.no_grad():
                for name, param in model.named_parameters():
                    orig = initial_weights[name]
                    delta = param - orig
                    total_delta_sq += torch.sum(delta ** 2).item()
                    total_orig_sq += torch.sum(orig ** 2).item()
            drift = math.sqrt(total_delta_sq) / math.sqrt(total_orig_sq)
            
            elapsed = time.time() - start_time
            print(f"  Step {step:3d}/{total_steps} | Train Loss: {loss.item():.4f} | Val Loss: {mean_val_loss:.4f} | Drift: {drift * 100:.2f}% | Elapsed: {elapsed:.1f}s")
            
            history.append({
                "step": step,
                "train_loss": loss.item(),
                "val_loss": mean_val_loss,
                "drift_pct": drift * 100,
                "lr": curr_lr
            })
            
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "val_loss": best_val_loss,
                    "drift_pct": drift * 100
                }, best_ckpt_path)
                
    # Save final checkpoint
    final_ckpt_path = CKPT_DIR / "candidate_step_120.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": total_steps,
        "final_loss": loss.item()
    }, final_ckpt_path)
    
    # Save training history
    history_path = CKPT_DIR / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    
    total_time = time.time() - start_time
    print(f"\n[Complete] Finished training in {total_time:.2f}s.")
    print(f"  Best Val Loss: {best_val_loss:.4f} (Saved to {best_ckpt_path})")
    print(f"  Final Checkpoint saved to {final_ckpt_path}")

if __name__ == "__main__":
    main()
