import hashlib
import json
import os
import torch

files = {
    "canonical_stage_b": "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
    "cr16_benchmark": "experiments/cognition/cr_exp_16_algorithmic_pretraining/benchmark_data/cr16_synthetic_benchmark_1000.json",
    "session_py": "chakrview/runtime/session.py",
    "v011": "releases/v0.11_conversational_model/chakrview_v011_model.pt",
    "v0101": "releases/v0.10.1_conversational_model/chakrview_v0101_model.pt"
}

results = {}
for k, path in files.items():
    if os.path.exists(path):
        with open(path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        results[k] = {"path": path, "sha256": h, "exists": True}
    else:
        results[k] = {"path": path, "exists": False}

print("Artifact hashes:")
print(json.dumps(results, indent=2))

# Load v0.11 strictly
print("\nInspecting v0.11 Checkpoint:")
ckpt = torch.load("releases/v0.11_conversational_model/chakrview_v011_model.pt", map_location="cpu")
print("Top-level keys:", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    sd = ckpt["model_state_dict"]
elif isinstance(ckpt, dict) and "state_dict" in ckpt:
    sd = ckpt["state_dict"]
else:
    sd = ckpt

total_params = sum(p.numel() for p in sd.values())
print(f"Total parameters in state dict: {total_params:,}")
print("Layer prefixes:")
prefixes = set(k.split(".")[0] for k in sd.keys())
print(prefixes)
for k, v in list(sd.items())[:8]:
    print(f"  {k}: {tuple(v.shape)}")
