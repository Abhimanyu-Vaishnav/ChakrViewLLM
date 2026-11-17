import hashlib
import sys
import platform
import torch
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")

def get_hash(path):
    p = Path(path)
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

print("=== SYSTEM ENVIRONMENT ===")
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("OS:", platform.platform())
print("Machine:", platform.machine(), platform.processor())

print("\n=== ARTIFACT PREFLIGHT AUDIT ===")
targets = [
    ("Protected session.py", REPO_ROOT / "chakrview" / "runtime" / "session.py", "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"),
    ("Primary Baseline: UCIP-09 Medium", REPO_ROOT / "releases" / "ucip_09_candidates" / "chakrview_ucip09_medium_model.pt", "5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499"),
    ("Experimental: UCIP-09R Medium", REPO_ROOT / "releases" / "ucip_09r_candidates" / "chakrview_ucip09r_medium_model.pt", "e1b943222e70fce2bac1b198e656bfea294209de1291f0042a2ebd677ccef9f0"),
    ("UCIP-09 Training Dataset", REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "data" / "ucip09_train.jsonl", "2b251762e8216cfd956acdbd343e757e64bf2c33309cc24fa0f653e4ef37108d"),
    ("UCIP-09R Training Dataset", REPO_ROOT / "experiments" / "cognition" / "ucip_09r" / "data" / "ucip09r_train.jsonl", "d53f14e2fecd4f6370081a0f12c89792d8e7db47ec3b70fc9cff50f93101bcec"),
    ("Tokenizer Config", REPO_ROOT / "data" / "experiments" / "vocab_4096" / "config.json", "0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589"),
    ("Tokenizer Merges", REPO_ROOT / "data" / "experiments" / "vocab_4096" / "merges.json", "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7"),
    ("Tokenizer Vocab", REPO_ROOT / "data" / "experiments" / "vocab_4096" / "vocab.json", "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"),
]

for label, path, expected in targets:
    actual = get_hash(path)
    match = (actual.lower() == expected.lower())
    print(f"[{'MATCH' if match else 'FAIL'}] {label}:")
    print(f"  Path: {path}")
    print(f"  Actual:   {actual}")
    print(f"  Expected: {expected}")
