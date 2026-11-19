"""
Phase B: Forensic Checkpoint Audit Script for UCIP-13R
Calculates file hashes, inspects state dict keys, tensor shapes, parameter counts,
checks for NaN/Inf, inspects optimizer/scheduler/metadata states.
"""

import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "ucip13"
CKPT_DIR = EXP_DIR / "checkpoints" / "primary_seed42"
MANIFEST_PATH = EXP_DIR / "checkpoints" / "checkpoint_manifest.json"
OUT_JSON = ROOT / "experiments" / "ucip13r" / "metrics" / "checkpoint_forensics.json"

with open(MANIFEST_PATH, "r") as f:
    manifest = json.load(f)

expected_steps = [0, 50, 100, 200, 400, 600, 800]
results = []

for step in expected_steps:
    ckpt_name = f"checkpoint_step_{step:03d}.pt"
    ckpt_path = CKPT_DIR / ckpt_name
    
    assert ckpt_path.is_file(), f"File does not exist: {ckpt_path}"
    raw_bytes = ckpt_path.read_bytes()
    file_size = len(raw_bytes)
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    
    # Check manifest match
    manifest_match = False
    for k, v in manifest.items():
        if ckpt_name in k and v == file_sha256:
            manifest_match = True
            break
            
    # Load safely
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Inspect contents
    top_keys = list(data.keys())
    state_dict = data.get("model_state_dict", {})
    
    total_params = 0
    nan_count = 0
    inf_count = 0
    param_summary = {}
    
    for p_name, tensor in state_dict.items():
        num_p = tensor.numel()
        total_params += num_p
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan:
            nan_count += 1
        if has_inf:
            inf_count += 1
        param_summary[p_name] = {
            "shape": list(tensor.shape),
            "numel": num_p,
            "dtype": str(tensor.dtype),
            "mean": round(tensor.float().mean().item(), 6),
            "std": round(tensor.float().std().item(), 6),
            "min": round(tensor.float().min().item(), 6),
            "max": round(tensor.float().max().item(), 6),
        }
        
    # Parameter hash for model parameters
    param_bytes = b"".join(state_dict[k].detach().cpu().numpy().tobytes() for k in sorted(state_dict.keys()))
    weights_sha256 = hashlib.sha256(param_bytes).hexdigest()
    
    # Optimizer & Scheduler inspection
    has_opt = "optimizer_state_dict" in data
    opt_state_count = len(data["optimizer_state_dict"]["state"]) if has_opt and "state" in data["optimizer_state_dict"] else 0
    has_sched = "scheduler_state_dict" in data
    
    step_meta = data.get("step")
    tokens_meta = data.get("tokens_processed")
    train_loss = data.get("train_loss")
    val_loss = data.get("val_loss")
    val_ppl = data.get("val_perplexity")
    
    rec = {
        "step": step,
        "filename": ckpt_name,
        "file_size_bytes": file_size,
        "file_sha256": file_sha256,
        "manifest_match": manifest_match,
        "top_level_keys": top_keys,
        "total_parameters": total_params,
        "weights_sha256": weights_sha256,
        "nan_param_count": nan_count,
        "inf_param_count": inf_count,
        "has_optimizer_state": has_opt,
        "optimizer_param_states_saved": opt_state_count,
        "has_scheduler_state": has_sched,
        "metadata": {
            "step": step_meta,
            "tokens_processed": tokens_meta,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
        },
        "sample_param_shapes": {k: param_summary[k]["shape"] for k in list(param_summary.keys())[:5]}
    }
    results.append(rec)
    print(f"Verified {ckpt_name}: Size {file_size} bytes | SHA {file_sha256[:12]}... | Match: {manifest_match} | Params: {total_params} | NaNs: {nan_count}")

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nPhase B Checkpoint Verification complete. Saved to:", OUT_JSON)
