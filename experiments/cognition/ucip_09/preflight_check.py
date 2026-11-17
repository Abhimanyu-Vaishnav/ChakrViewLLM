import hashlib
import sys
import os
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

print("\n=== ARTIFACT VERIFICATION ===")
targets = [
    ("Protected session.py", REPO_ROOT / "chakrview" / "runtime" / "session.py", "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"),
    ("UCIP-07 Medium", REPO_ROOT / "releases" / "ucip_07_candidates" / "chakrview_ucip07_medium_model.pt", "554c503e058311d8732216a3ad200cf1c7d60a98baa39f78c26328c66516d5b2"),
    ("UCIP-08 Medium", REPO_ROOT / "releases" / "ucip_08_candidates" / "chakrview_ucip08_medium_model.pt", "fe6f381acf05662bcc6570e2ae4a06f3ecc89b669811e5340eebd84bb014f631"),
    ("Canonical Stage B", REPO_ROOT / "models" / "canonical" / "stage_b" / "chakrview_stage_b.pt", "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"),
    ("Owner Reality 100", REPO_ROOT / "experiments" / "cognition" / "ucip_08" / "sealed_owner_reality_100.json", "7607eaf4635232822cdd0b1faad15543664e7c7b339ecd0fda5a7a8e3766ae15"),
]

for label, path, expected in targets:
    actual = get_hash(path)
    match = (actual.lower() == expected.lower())
    print(f"[{'MATCH' if match else 'FAIL'}] {label}:")
    print(f"  Path: {path}")
    print(f"  Actual:   {actual}")
    print(f"  Expected: {expected}")

# Find UCIP-06 Medium
print("\n=== SEARCHING UCIP-06 MEDIUM ===")
ucip06_candidates = list((REPO_ROOT / "experiments" / "cognition" / "ucip_06").rglob("*.pt")) + list((REPO_ROOT / "releases" / "ucip_06_candidate").rglob("*.pt"))
for c in ucip06_candidates:
    print(f"Candidate: {c} -> {get_hash(c)}")

# Check canonical model hash
sys.path.insert(0, str(REPO_ROOT))
try:
    from tests.test_checkpoint_integrity import compute_canonical_model_hash
    from chakrview.model import ChakrMicro
    m = ChakrMicro(seed=42)
    h = compute_canonical_model_hash(m)
    expected_c = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
    print(f"\nCanonical ChakrMicro(seed=42) Hash: {h} (Expected: {expected_c}) -> {'MATCH' if h.lower() == expected_c.lower() else 'FAIL'}")
except Exception as e:
    print(f"\nCanonical model hash check error: {e}")
