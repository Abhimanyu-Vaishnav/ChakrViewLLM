"""Inspect logits at structural branch points in JSON generation."""

import json
import torch
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TOK_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
CKPT_PATH = REPO_ROOT / "experiments" / "batch3111_3150" / "json_sft_v2_pilot_run" / "sft_best_checkpoint.pt"

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def main():
    tok, _ = load_tokenizer_artifacts(str(TOK_DIR))
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    with open(REPO_ROOT / "experiments" / "batch3111_3150" / "val.jsonl", "r", encoding="utf-8") as f:
        val_recs = [json.loads(line) for line in f if line.strip()]
    r = [x for x in val_recs if x["category"].startswith("structured")][0]

    p = r["prompt"]
    inp = [0] + tok.encode(p)

    # Branch point 1: After {"
    curr = list(inp) + [126, 37]
    x = torch.tensor([curr], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    top10 = torch.topk(logits[0, -1, :], 10)
    print("Top 10 tokens after {\":")
    for sc, idx in zip(top10.values.tolist(), top10.indices.tolist()):
        print(f"  {idx:4d} {repr(tok.decode([idx])):12s} logit={sc:.2f}")

    # Branch point 2: After {"status": "
    curr = list(inp) + tok.encode('{"status": "')
    x = torch.tensor([curr], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    top10 = torch.topk(logits[0, -1, :], 10)
    print("\nTop 10 tokens after {\"status\": \":")
    for sc, idx in zip(top10.values.tolist(), top10.indices.tolist()):
        print(f"  {idx:4d} {repr(tok.decode([idx])):12s} logit={sc:.2f}")

    # Branch point 3: After {"status": "queued", "code": 
    curr = list(inp) + tok.encode('{"status": "queued", "code": ')
    x = torch.tensor([curr], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    top10 = torch.topk(logits[0, -1, :], 10)
    print("\nTop 10 tokens after {\"status\": \"queued\", \"code\": :")
    for sc, idx in zip(top10.values.tolist(), top10.indices.tolist()):
        print(f"  {idx:4d} {repr(tok.decode([idx])):12s} logit={sc:.2f}")

    # Branch point 4: After {"status": "queued", "code": 202
    curr = list(inp) + tok.encode('{"status": "queued", "code": 202')
    x = torch.tensor([curr], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    top10 = torch.topk(logits[0, -1, :], 10)
    print("\nTop 10 tokens after {\"status\": \"queued\", \"code\": 202:")
    for sc, idx in zip(top10.values.tolist(), top10.indices.tolist()):
        print(f"  {idx:4d} {repr(tok.decode([idx])):12s} logit={sc:.2f}")

if __name__ == "__main__":
    main()
