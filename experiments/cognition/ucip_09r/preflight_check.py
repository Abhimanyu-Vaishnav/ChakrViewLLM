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

print("\n=== ARTIFACT INTEGRITY AUDIT ===")
targets = [
    ("Protected session.py", REPO_ROOT / "chakrview" / "runtime" / "session.py", "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"),
    ("UCIP-09 Medium Candidate", REPO_ROOT / "releases" / "ucip_09_candidates" / "chakrview_ucip09_medium_model.pt", "5513fefe878ca85a637aa8401b3bebf5b8e3ca93ab362e07722d629e34294499"),
    ("UCIP-08 Medium Parent", REPO_ROOT / "releases" / "ucip_08_candidates" / "chakrview_ucip08_medium_model.pt", "fe6f381acf05662bcc6570e2ae4a06f3ecc89b669811e5340eebd84bb014f631"),
    ("UCIP-07 Medium", REPO_ROOT / "releases" / "ucip_07_candidates" / "chakrview_ucip07_medium_model.pt", "554c503e058311d8732216a3ad200cf1c7d60a98baa39f78c26328c66516d5b2"),
    ("Sealed Language Intelligence 500", REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "sealed_language_intelligence_500.json", "3958d3839c0e8ef38fa056cffb24594243cbbe59623dd7460a340a0189981eba"),
    ("Sealed Owner Reality 100", REPO_ROOT / "experiments" / "cognition" / "ucip_08" / "sealed_owner_reality_100.json", "7607eaf4635232822cdd0b1faad15543664e7c7b339ecd0fda5a7a8e3766ae15"),
]

for label, path, expected in targets:
    actual = get_hash(path)
    # Check if JSON has embedded sha
    if label.startswith("Sealed") and actual != expected and path.exists():
        import json
        d = json.load(open(path, encoding="utf-8"))
        embedded = d.get("sha256", "")
        match = (embedded.lower() == expected.lower())
        print(f"[{'MATCH' if match else 'FAIL'}] {label} (embedded sha):")
        print(f"  Path: {path}")
        print(f"  Embedded: {embedded}")
        print(f"  Expected: {expected}")
        print(f"  File SHA: {actual}")
    else:
        match = (actual.lower() == expected.lower())
        print(f"[{'MATCH' if match else 'FAIL'}] {label}:")
        print(f"  Path: {path}")
        print(f"  Actual:   {actual}")
        print(f"  Expected: {expected}")
