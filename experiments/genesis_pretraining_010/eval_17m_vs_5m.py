"""
Benchmark evaluation comparing 5.18M (NR-005 best) vs 17.06M (NR-010 SFT best) across 300 standardized blind prompts.
"""

import json
import os
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.genesis.chakr_genesis import ChakrGenesis, GenesisConfig
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tok, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load 5.18M model
cfg_5m = GenesisConfig()
m_5m = ChakrGenesis(cfg_5m).to(device)
ckpt_5m = torch.load("experiments/genesis_conversational_foundation_005/checkpoints/genesis_sft_best.pt", map_location=device, weights_only=False)
m_5m.load_state_dict(ckpt_5m.get("model_state_dict", ckpt_5m))
m_5m.eval()

# Load 17.06M model
cfg_17m = Genesis17MConfig(max_seq_len=256)
m_17m = ChakrGenesis17M(cfg_17m).to(device)
ckpt_17m = torch.load("experiments/genesis_pretraining_010/sft_checkpoints/genesis_17m_sft_best.pt", map_location=device, weights_only=False)
m_17m.load_state_dict(ckpt_17m["model_state_dict"])
m_17m.eval()

manifest_path = Path("reports/NR006_BLIND_DATASET_MANIFEST.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest_data = json.load(f)
eval_prompts = manifest_data.get("prompts_inventory", [])

def generate(model, prompt: str) -> str:
    p_ids = tok.encode(prompt)
    idx = torch.tensor([p_ids], dtype=torch.long, device=device)
    generated = []
    with torch.no_grad():
        for _ in range(40):
            idx_cond = idx[:, -256:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / 0.7
            v, _ = torch.topk(logits, min(20, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            tok_val = idx_next.item()
            if tok_val == 1:
                break
            generated.append(tok_val)
            idx = torch.cat((idx, idx_next), dim=1)
    return tok.decode(generated, errors="replace").strip()

def score(resp: str, prompt_meta: dict) -> str:
    if not resp:
        return "Incomplete"
    words = resp.split()
    if len(words) >= 6:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        if len(trigrams) > len(set(trigrams)) * 2:
            return "Repetitive"
    if len(resp) >= 8 and any(c.isalpha() for c in resp):
        return "Coherent"
    return "Incorrect"

res_5m = {"Coherent": 0, "Incorrect": 0, "Incomplete": 0, "Repetitive": 0}
res_17m = {"Coherent": 0, "Incorrect": 0, "Incomplete": 0, "Repetitive": 0}

for item in eval_prompts:
    p = item["prompt"]
    r5 = generate(m_5m, p)
    r17 = generate(m_17m, p)
    res_5m[score(r5, item)] += 1
    res_17m[score(r17, item)] += 1

total = len(eval_prompts)
print("=== 300-PROMPT BENCHMARK: 5.18M VS 17.06M SFT ===")
print(f"ChakrGenesis-5M (NR-005 Best):   {res_5m['Coherent']}/{total} ({res_5m['Coherent']/total*100:.1f}%) Coherent, Repetition: {res_5m['Repetitive']}")
print(f"ChakrGenesis-17M (NR-010 Best):  {res_17m['Coherent']}/{total} ({res_17m['Coherent']/total*100:.1f}%) Coherent, Repetition: {res_17m['Repetitive']}")

out_file = Path("experiments/genesis_pretraining_010/eval_300_comparison.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump({"5m": res_5m, "17m": res_17m}, f, indent=2)
