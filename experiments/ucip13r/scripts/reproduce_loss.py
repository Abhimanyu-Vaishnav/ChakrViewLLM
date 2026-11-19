"""
Phase E: Independent Loss Reproduction Script for UCIP-13R.
Evaluates:
1. Initial Validation Loss on Step 0 checkpoint
2. Final Validation Loss on Step 800 checkpoint
3. Final Held-Out Test Loss on Step 800 checkpoint

Ensures:
- Exact evaluation mode (eval())
- torch.no_grad()
- Complete partition traversal (all batches)
- Explicit loss, token, and perplexity tracking
"""

import json
import math
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.training.loss import CausalLoss

DEVICE = torch.device("cpu")
EXP_DIR = ROOT / "experiments" / "ucip13"
CKPT_DIR = EXP_DIR / "checkpoints" / "primary_seed42"
OUT_FILE = ROOT / "experiments" / "ucip13r" / "evaluations" / "independent_loss_reproduction.json"


def evaluate_partition(model, shard_dir, sequence_length=512, batch_size=16, ignore_index=2):
    model.eval()
    ds = StreamingTokenDataset(
        shard_dir,
        sequence_length=sequence_length,
        loop=False,
        drop_remainder=True,
        pad_token_id=ignore_index
    )
    loader = DataLoader(ds, batch_size=batch_size)
    
    total_loss_sum = 0.0
    total_batches = 0
    total_valid_tokens = 0
    
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            target_ids = batch["target_ids"].to(DEVICE)
            
            logits = model(input_ids)
            
            # Independent cross-entropy verification
            B, T, V = logits.shape
            flat_logits = logits.view(-1, V)
            flat_targets = target_ids.view(-1)
            
            # Token counting (non-pad)
            valid_mask = flat_targets != ignore_index
            valid_count = valid_mask.sum().item()
            total_valid_tokens += valid_count
            
            batch_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=ignore_index, reduction="mean").item()
            total_loss_sum += batch_loss
            total_batches += 1
            
    duration = time.time() - t0
    mean_loss = total_loss_sum / max(1, total_batches)
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    
    return {
        "mean_loss": round(mean_loss, 4),
        "perplexity": round(perplexity, 2),
        "batches": total_batches,
        "valid_tokens": total_valid_tokens,
        "duration_sec": round(duration, 2)
    }


def run_evaluations():
    print("=" * 80)
    print("INDEPENDENT LOSS REPRODUCTION — UCIP-13R")
    print("=" * 80)
    
    val_dir = ROOT / "data" / "tokenized" / "stage_c" / "validation"
    test_dir = ROOT / "data" / "tokenized" / "stage_c" / "test"
    
    # 1. Step 0 on Validation
    print("Evaluating Step 0 Checkpoint on Validation Split...")
    ckpt0_path = CKPT_DIR / "checkpoint_step_000.pt"
    data0 = torch.load(ckpt0_path, map_location="cpu", weights_only=False)
    m0 = ChakrMicro(ModelConfig())
    m0.load_state_dict(data0["model_state_dict"])
    val_res_step0 = evaluate_partition(m0, val_dir)
    print(f"Step 0 Val Loss: {val_res_step0['mean_loss']} | PPL: {val_res_step0['perplexity']} in {val_res_step0['duration_sec']}s")
    
    # 2. Step 800 on Validation
    print("\nEvaluating Step 800 Checkpoint on Validation Split...")
    ckpt800_path = CKPT_DIR / "checkpoint_step_800.pt"
    data800 = torch.load(ckpt800_path, map_location="cpu", weights_only=False)
    m800 = ChakrMicro(ModelConfig())
    m800.load_state_dict(data800["model_state_dict"])
    val_res_step800 = evaluate_partition(m800, val_dir)
    print(f"Step 800 Val Loss: {val_res_step800['mean_loss']} | PPL: {val_res_step800['perplexity']} in {val_res_step800['duration_sec']}s")
    
    # 3. Step 800 on Test
    print("\nEvaluating Step 800 Checkpoint on Held-Out Test Split...")
    test_res_step800 = evaluate_partition(m800, test_dir)
    print(f"Step 800 Test Loss: {test_res_step800['mean_loss']} | PPL: {test_res_step800['perplexity']} in {test_res_step800['duration_sec']}s")
    
    # Also evaluate Step 0 on Test for baseline comparison
    print("\nEvaluating Step 0 Checkpoint on Held-Out Test Split (Baseline comparison)...")
    test_res_step0 = evaluate_partition(m0, test_dir)
    print(f"Step 0 Test Loss: {test_res_step0['mean_loss']} | PPL: {test_res_step0['perplexity']} in {test_res_step0['duration_sec']}s")
    
    summary = {
        "reported_metrics": {
            "initial_val_loss": 8.3084,
            "final_val_loss": 4.6859,
            "final_val_perplexity": 108.41,
            "final_test_loss": 4.6595,
            "final_test_perplexity": 105.59
        },
        "independent_metrics": {
            "initial_val_loss": val_res_step0["mean_loss"],
            "initial_val_perplexity": val_res_step0["perplexity"],
            "initial_val_batches": val_res_step0["batches"],
            "initial_val_tokens": val_res_step0["valid_tokens"],
            
            "final_val_loss": val_res_step800["mean_loss"],
            "final_val_perplexity": val_res_step800["perplexity"],
            "final_val_batches": val_res_step800["batches"],
            "final_val_tokens": val_res_step800["valid_tokens"],
            
            "initial_test_loss": test_res_step0["mean_loss"],
            "initial_test_perplexity": test_res_step0["perplexity"],
            
            "final_test_loss": test_res_step800["mean_loss"],
            "final_test_perplexity": test_res_step800["perplexity"],
            "final_test_batches": test_res_step800["batches"],
            "final_test_tokens": test_res_step800["valid_tokens"]
        },
        "discrepancies": {
            "initial_val_loss_diff": round(val_res_step0["mean_loss"] - 8.3084, 4),
            "final_val_loss_diff": round(val_res_step800["mean_loss"] - 4.6859, 4),
            "final_test_loss_diff": round(test_res_step800["mean_loss"] - 4.6595, 4)
        }
    }
    
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\nResults successfully saved to:", OUT_FILE)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_evaluations()
