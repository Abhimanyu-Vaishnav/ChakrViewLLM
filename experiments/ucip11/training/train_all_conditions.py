"""
UCIP-11 Controlled Training Harness.
Executes Conditions B, C, D, and E in isolated subdirectories under checkpoints/.
Logs loss progression, validation loss, and rollout samples.
"""

import sys
import os
import time
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

REPO_ROOT = Path(r"D:\Project\ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from scripts.run_m3_evaluations import DirectNeuralRunner

tokenizer_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
tokenizer, _ = load_tokenizer_artifacts(tokenizer_dir)

def collate_cpt_batch(lines: List[str], max_len: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_inps = []
    batch_tgts = []
    for line in lines:
        toks = tokenizer.encode(line, add_bos=True, add_eos=True)
        if len(toks) > max_len:
            toks = toks[:max_len]
        batch_inps.append(toks)
        batch_tgts.append(toks)
        
    m_len = max(len(s) for s in batch_inps)
    pad_inps = [s + [2] * (m_len - len(s)) for s in batch_inps]
    pad_tgts = [t + [-100] * (m_len - len(t)) for t in batch_tgts]
    
    return torch.tensor(pad_inps, dtype=torch.long), torch.tensor(pad_tgts, dtype=torch.long)

def collate_sft_batch(items: List[Dict[str, str]], max_len: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_inps = []
    batch_tgts = []
    for it in items:
        p = it["prompt"]
        r = it.get("target", it.get("response", ""))
        p_ids = tokenizer.encode(f"User: {p}\nAssistant:", add_bos=True, add_eos=False)
        r_ids = tokenizer.encode(f" {r}", add_bos=False, add_eos=True)
        
        seq = p_ids + r_ids
        lbl = [-100] * len(p_ids) + r_ids
        if len(seq) > max_len:
            seq = seq[:max_len]
            lbl = lbl[:max_len]
        batch_inps.append(seq)
        batch_tgts.append(lbl)
        
    m_len = max(len(s) for s in batch_inps)
    pad_inps = [s + [2] * (m_len - len(s)) for s in batch_inps]
    pad_tgts = [l + [-100] * (m_len - len(l)) for l in batch_tgts]
    
    return torch.tensor(pad_inps, dtype=torch.long), torch.tensor(pad_tgts, dtype=torch.long)

def run_training_condition(
    condition_id: str,
    parent_checkpoint: Optional[Path],
    is_cpt: bool,
    train_data_raw,
    val_data_raw,
    steps: int = 100,
    batch_size: int = 4,
    lr: float = 2e-5,
    label_smoothing: float = 0.0,
    eval_every: int = 20,
    seed: int = 42
) -> Dict:
    print(f"\n=======================================================")
    print(f"Executing UCIP-11: {condition_id}")
    print(f"Steps: {steps} | Batch Size: {batch_size} | LR: {lr} | Seed: {seed}")
    print(f"=======================================================")
    
    random.seed(seed)
    torch.manual_seed(seed)
    
    save_dir = REPO_ROOT / "experiments" / "ucip11" / "checkpoints" / condition_id.lower()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Model initialization
    if parent_checkpoint is not None:
        print(f"Loading parent weights from: {parent_checkpoint.name}")
        runner = DirectNeuralRunner(parent_checkpoint)
        model = runner.model
        parent_raw = torch.load(parent_checkpoint, map_location="cpu")
        model_cfg = parent_raw.get("config", {})
    else:
        print(f"Initializing from scratch with random weights (Seed {seed})")
        # 12.9M Medium model configuration
        model_cfg = {
            'vocab_size': 4096, 'd_model': 384, 'n_layers': 7, 'n_heads': 6,
            'max_seq_len': 512, 'rms_norm_eps': 1e-05, 'rope_theta': 50000.0,
            'hidden_dim': 896, 'dropout': 0.0, 'initializer_range': 0.02,
            'dtype': 'float32', 'use_bias': False, 'tie_embeddings': True,
            'bos_token_id': 0, 'eos_token_id': 1, 'pad_token_id': 2
        }
        cfg_obj = ModelConfig(**model_cfg)
        model = ChakrMicro(cfg_obj)
        
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
    
    train_indices = list(range(len(train_data_raw)))
    random.shuffle(train_indices)
    data_cursor = 0
    
    logs = []
    best_val_loss = float("inf")
    best_ckpt_path = save_dir / f"{condition_id.lower()}_best.pt"
    
    t_start = time.time()
    accum_loss = 0.0
    
    for step in range(1, steps + 1):
        # Fetch batch
        batch_slice = []
        for _ in range(batch_size):
            if data_cursor >= len(train_indices):
                random.shuffle(train_indices)
                data_cursor = 0
            batch_slice.append(train_data_raw[train_indices[data_cursor]])
            data_cursor += 1
            
        if is_cpt:
            inps, tgts = collate_cpt_batch(batch_slice)
        else:
            inps, tgts = collate_sft_batch(batch_slice)
            
        optimizer.zero_grad()
        logits = model(inps)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = tgts[:, 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        accum_loss += loss.item()
        
        if step % eval_every == 0 or step == steps:
            model.eval()
            val_loss = 0.0
            val_samples = min(20, len(val_data_raw))
            val_slice = val_data_raw[:val_samples]
            
            with torch.no_grad():
                if is_cpt:
                    v_inps, v_tgts = collate_cpt_batch(val_slice)
                else:
                    v_inps, v_tgts = collate_sft_batch(val_slice)
                v_logits = model(v_inps)
                v_shift_logits = v_logits[:, :-1, :].contiguous()
                v_shift_targets = v_tgts[:, 1:].contiguous()
                v_loss = criterion(v_shift_logits.view(-1, v_shift_logits.size(-1)), v_shift_targets.view(-1))
                val_loss = v_loss.item()
                
            avg_train_loss = accum_loss / eval_every
            accum_loss = 0.0
            elapsed = time.time() - t_start
            
            log_entry = {
                "step": step,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(val_loss, 4),
                "elapsed_seconds": round(elapsed, 2)
            }
            logs.append(log_entry)
            print(f"Step {step:3d}/{steps} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Elapsed: {elapsed:.1f}s")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "condition": condition_id,
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "config": model_cfg,
                    "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else "scratch",
                    "seed": seed
                }, best_ckpt_path)
            model.train()
            
    final_ckpt_path = save_dir / f"{condition_id.lower()}_final.pt"
    torch.save({
        "condition": condition_id,
        "step": steps,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "config": model_cfg,
        "parent_checkpoint": str(parent_checkpoint) if parent_checkpoint else "scratch",
        "seed": seed
    }, final_ckpt_path)
    
    # Compute SHA-256
    best_sha = hashlib.sha256(best_ckpt_path.read_bytes()).hexdigest()
    final_sha = hashlib.sha256(final_ckpt_path.read_bytes()).hexdigest()
    
    return {
        "condition": condition_id,
        "best_checkpoint": str(best_ckpt_path.relative_to(REPO_ROOT)),
        "best_checkpoint_sha256": best_sha,
        "final_checkpoint": str(final_ckpt_path.relative_to(REPO_ROOT)),
        "final_checkpoint_sha256": final_sha,
        "best_val_loss": best_val_loss,
        "logs": logs
    }

if __name__ == "__main__":
    # Prepare datasets
    # CPT data: Stage B clean text
    cpt_train_lines = []
    cpt_val_lines = []
    stage_b_train_dir = REPO_ROOT / "data" / "processed" / "stage_b" / "train"
    stage_b_val_dir = REPO_ROOT / "data" / "processed" / "stage_b" / "validation"
    
    for f in stage_b_train_dir.glob("*.txt"):
        lines = [l.strip() for l in f.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
        cpt_train_lines.extend(lines)
    for f in stage_b_val_dir.glob("*.txt"):
        lines = [l.strip() for l in f.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
        cpt_val_lines.extend(lines)
        
    print(f"Prepared CPT Corpus: {len(cpt_train_lines)} train lines, {len(cpt_val_lines)} val lines.")
    
    # SFT data: Exp34 candidate v3
    sft_train_p = REPO_ROOT / "data" / "sft" / "exp34_candidate_v3" / "splits" / "exp34_train_split_900.jsonl"
    sft_val_p = REPO_ROOT / "data" / "sft" / "exp34_candidate_v3" / "splits" / "exp34_val_split_100.jsonl"
    
    sft_train_items = [json.loads(l) for l in sft_train_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    sft_val_items = [json.loads(l) for l in sft_val_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Prepared SFT Corpus: {len(sft_train_items)} train items, {len(sft_val_items)} val items.")
    
    parent_ucip09 = REPO_ROOT / "releases" / "ucip_09_candidates" / "chakrview_ucip09_medium_model.pt"
    
    experiment_results = {}
    
    # Condition B: CPT on Stage B from UCIP-09
    res_b = run_training_condition(
        "Condition_B_CPT", parent_ucip09, is_cpt=True,
        train_data_raw=cpt_train_lines, val_data_raw=cpt_val_lines,
        steps=100, batch_size=4, lr=2e-5, label_smoothing=0.0
    )
    experiment_results["Condition_B_CPT"] = res_b
    
    # Condition C: SFT on Exp34 from UCIP-09
    res_c = run_training_condition(
        "Condition_C_SFT", parent_ucip09, is_cpt=False,
        train_data_raw=sft_train_items, val_data_raw=sft_val_items,
        steps=100, batch_size=4, lr=1.8e-5, label_smoothing=0.05
    )
    experiment_results["Condition_C_SFT"] = res_c
    
    # Condition D: CPT + SFT (start from Condition B best ckpt)
    cond_b_best_path = REPO_ROOT / res_b["best_checkpoint"]
    res_d = run_training_condition(
        "Condition_D_CPT_SFT", cond_b_best_path, is_cpt=False,
        train_data_raw=sft_train_items, val_data_raw=sft_val_items,
        steps=60, batch_size=4, lr=1.2e-5, label_smoothing=0.05
    )
    experiment_results["Condition_D_CPT_SFT"] = res_d
    
    # Condition E: Scratch language learning from random weights
    res_e = run_training_condition(
        "Condition_E_Scratch", None, is_cpt=True,
        train_data_raw=cpt_train_lines, val_data_raw=cpt_val_lines,
        steps=100, batch_size=4, lr=5e-4, label_smoothing=0.0
    )
    experiment_results["Condition_E_Scratch"] = res_e
    
    summary_path = REPO_ROOT / "experiments" / "ucip11" / "results" / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)
    print(f"\nAll training conditions completed successfully. Summary saved to {summary_path}")
