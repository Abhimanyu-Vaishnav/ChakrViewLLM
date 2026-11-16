import hashlib
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.canonical_hash import verify_canonical_baseline

def get_hash(path):
    full_p = os.path.join(REPO_ROOT, path)
    if not os.path.exists(full_p):
        return "MISSING"
    h = hashlib.sha256()
    with open(full_p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()

targets = {
    "UCIP-06 Medium": ("releases/ucip_06_candidate/chakrview_ucip06_medium_model.pt", "be843bc2cbc08f3f8459b85d6eec2cdce534441ac5ccc797f0bc0d2be8aa1b72"),
    "UCIP-06 Micro": ("releases/ucip_06_candidate/chakrview_ucip06_micro_model.pt", "7706e19ba568946c527cc15d90dc7b50180b7c754eb2b5b6dd7a5cdb3c55fd71"),
    "UCIP-05 Medium Parent": ("releases/ucip_05_candidate/chakrview_ucip05_medium_model.pt", "089335f838c3e9999600144edccbb1caf1a4a741f43f17d493625b457e90e234"),
    "Canonical Stage B": ("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt", "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"),
    "Protected Runtime": ("chakrview/runtime/session.py", "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67"),
}

all_match = True
for name, (path, exp) in targets.items():
    act = get_hash(path)
    match = (act == exp.lower())
    if not match:
        all_match = False
    print(f"{name:<24}: {act} | {'MATCH' if match else 'MISMATCH'}")

torch.manual_seed(42)
canon_model = ChakrMicro(ModelConfig())
res = verify_canonical_baseline(canon_model)
canon_h = res["digest"]
canon_exp = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da".lower()
match_canon = res["is_bit_exact"] and (canon_h == canon_exp)
if not match_canon:
    all_match = False
print(f"Canonical Architecture  : {canon_h} | {'MATCH' if match_canon else 'MISMATCH'}")

print(f"\nALL PREFLIGHT HASHES VERIFIED: {all_match}")
