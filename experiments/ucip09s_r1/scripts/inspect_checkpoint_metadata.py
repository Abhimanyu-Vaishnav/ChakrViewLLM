import torch
import json
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")

ckpts = {
    "ucip08_medium": REPO_ROOT / "releases" / "ucip_08_candidates" / "chakrview_ucip08_medium_model.pt",
    "ucip09_medium": REPO_ROOT / "releases" / "ucip_09_candidates" / "chakrview_ucip09_medium_model.pt",
    "ucip09r_medium": REPO_ROOT / "releases" / "ucip_09r_candidates" / "chakrview_ucip09r_medium_model.pt",
    "ucip09s_pilot": REPO_ROOT / "experiments" / "ucip09s" / "checkpoints" / "ucip09s_balanced_pilot.pt"
}

info = {}
for name, p in ckpts.items():
    if not p.exists():
        info[name] = "NOT FOUND"
        continue
    c = torch.load(p, map_location="cpu")
    meta = {
        "path": str(p),
        "keys": list(c.keys()),
        "step": c.get("step"),
        "val_loss": c.get("val_loss"),
        "architecture": c.get("architecture"),
        "parent_checkpoint": c.get("parent_checkpoint"),
        "config": c.get("config"),
        "state_dict_keys_count": len(c.get("model_state_dict", c.get("model_state", {})))
    }
    info[name] = meta
    print(f"=== {name} ===")
    print(f"  Architecture: {meta['architecture']}")
    print(f"  Parent:       {meta['parent_checkpoint']}")
    print(f"  Step:         {meta['step']}")
    print(f"  Val Loss:     {meta['val_loss']}")
    print(f"  Config:       {meta['config']}")

with open(REPO_ROOT / "experiments" / "ucip09s_r1" / "evaluations" / "checkpoint_metadata.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)
