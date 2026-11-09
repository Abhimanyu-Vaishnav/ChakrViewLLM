"""
CR-EXP-15 Phase 4: Tokenizer and Representation Diagnostics
Evaluates:
- Subword fragmentation across 1, 2, 3, 4, 5+ token entity names
- Round-trip lossless byte reconstruction
- Name copying accuracy across subword length
- Name identification / selection across subword length
Using CR-12 Parent checkpoint.
"""

import sys
import json
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
PARENT_CKPT = REPO_ROOT / "experiments" / "cognition" / "cr_exp_14_entity_binding" / "checkpoints" / "baseline_cr12_b" / "cr_exp_12_candidate_b_best_val_copy.pt"

tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
cfg = ModelConfig()
model = ChakrMicro(cfg)
ck = torch.load(PARENT_CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ck.get("model_state_dict", ck))
model.eval()

# Broad test suite of names across length and familiarity
TEST_NAMES = [
    # 1-2 token names
    "Dev", "Om", "Jai", "Veer", "Raj", "Karan", "Aarav", "Pooja", "Rohan", "Maya",
    # 3 token names
    "Abhay", "Bimla", "Chetan", "Ekta", "Girish", "Hema", "Indu", "Kavita", "Mina", "Radha",
    # 4-5 token names
    "Damodar", "Hansmukh", "Kailashpati", "Yadunath", "Vatsala", "Bhavaniprasad", "Venkateshwara", "Trilok", "Pushpa", "Subhash"
]

results = []

for name in TEST_NAMES:
    enc = tok.encode(name)
    subwords = [tok.decode([t]) for t in enc]
    reconstructed = tok.decode(enc)
    is_lossless = (reconstructed == name)
    
    # 1. Direct copy probe
    prompt_copy = f"Repeat the name: {name}.\nName:"
    enc_prompt = tok.encode(prompt_copy)
    curr = list(enc_prompt)
    with torch.no_grad():
        for _ in range(12):
            x = torch.tensor([curr], dtype=torch.long)
            nxt = model(x)[0, -1].argmax().item()
            curr.append(nxt)
            if nxt == 13: break
    gen_copy = tok.decode(curr[len(enc_prompt):]).strip()
    copy_success = name.lower() in gen_copy.lower()
    
    # 2. In-context selection probe (Select between name and distractor 'Sohan')
    prompt_sel = f"Between {name} and Sohan, the person mentioned is:"
    enc_sel = tok.encode(prompt_sel)
    curr_s = list(enc_sel)
    with torch.no_grad():
        for _ in range(10):
            x = torch.tensor([curr_s], dtype=torch.long)
            nxt = model(x)[0, -1].argmax().item()
            curr_s.append(nxt)
            if nxt == 13: break
    gen_sel = tok.decode(curr_s[len(enc_sel):]).strip()
    sel_success = (name.lower() in gen_sel.lower()) and ("sohan" not in gen_sel.lower())
    
    results.append({
        "name": name,
        "token_count": len(enc),
        "tokens": enc,
        "subwords": subwords,
        "is_lossless": is_lossless,
        "copy_prompt_output": gen_copy,
        "copy_success": copy_success,
        "sel_prompt_output": gen_sel,
        "sel_success": sel_success
    })

out_path = REPO_ROOT / "experiments" / "cognition" / "cr_exp_15_architecture_binding" / "eval_outputs" / "tokenizer_diagnostics.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"Diagnostics complete. Analyzed {len(results)} entity names.")
copy_acc = sum(1 for r in results if r["copy_success"]) / len(results)
sel_acc = sum(1 for r in results if r["sel_success"]) / len(results)
print(f"Baseline Direct Copy Accuracy: {copy_acc*100:.1f}%")
print(f"Baseline In-Context Selection Accuracy: {sel_acc*100:.1f}%")
