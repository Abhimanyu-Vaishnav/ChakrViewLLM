import os
import sys
import json
import hashlib
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")

def get_sha256(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

def check_files():
    targets = {
        "session.py": REPO_ROOT / "chakrview" / "runtime" / "session.py",
        "ucip09_medium_ckpt": REPO_ROOT / "releases" / "ucip_09_candidates" / "chakrview_ucip09_medium_model.pt",
        "ucip09r_medium_ckpt": REPO_ROOT / "releases" / "ucip_09r_candidates" / "chakrview_ucip09r_medium_model.pt",
        "ucip09s_pilot_ckpt": REPO_ROOT / "experiments" / "ucip09s" / "checkpoints" / "ucip09s_balanced_pilot.pt",
        "ucip08_medium_ckpt": REPO_ROOT / "releases" / "ucip_08_candidates" / "chakrview_ucip08_medium_model.pt",
        "ucip09_train_jsonl": REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "data" / "ucip09_train.jsonl",
        "ucip09_val_jsonl": REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "data" / "ucip09_validation.jsonl",
        "ucip09r_train_jsonl": REPO_ROOT / "experiments" / "cognition" / "ucip_09r" / "data" / "ucip09r_train.jsonl",
        "ucip09r_val_jsonl": REPO_ROOT / "experiments" / "cognition" / "ucip_09r" / "data" / "ucip09r_val.jsonl",
        "ucip09s_balanced_train": REPO_ROOT / "experiments" / "ucip09s" / "datasets" / "ucip09s_balanced_train.jsonl",
        "ucip09s_balanced_val": REPO_ROOT / "experiments" / "ucip09s" / "datasets" / "ucip09s_balanced_val.jsonl",
        "tokenizer_config": REPO_ROOT / "data" / "experiments" / "vocab_4096" / "tokenizer_config.json",
        "tokenizer_vocab": REPO_ROOT / "data" / "experiments" / "vocab_4096" / "vocab.json",
        "tokenizer_merges": REPO_ROOT / "data" / "experiments" / "vocab_4096" / "merges.txt",
    }
    
    res = {}
    for name, path in targets.items():
        sha = get_sha256(path)
        size = path.stat().st_size if path.exists() else 0
        res[name] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha, "size_bytes": size}
        print(f"{name:25s}: {sha} ({size:,} bytes)")
    
    with open(REPO_ROOT / "experiments" / "ucip09s_r1" / "evaluations" / "file_hashes.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    check_files()
