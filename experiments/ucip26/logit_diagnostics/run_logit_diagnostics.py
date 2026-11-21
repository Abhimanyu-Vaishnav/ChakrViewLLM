"""
UCIP-26 Phase 3: Next-Token Logit Diagnostics at the Generation Boundary.
Inspects the probability distribution at the assistant generation boundary across
a representative frozen sample from Tasks A-H for both Step-800 and UCIP-20.
Records:
- Top-1, Top-5, Top-20 tokens with probabilities
- EOS probability
- Distribution entropy
- Probability and rank of expected target's first token
- Probability of dominant attractor tokens (e.g. 68 ['A'], 38 ['#'])
"""

from __future__ import annotations
import glob
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def load_model(checkpoint_path: Path) -> ChakrMicro:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = ChakrMicro(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_logit_diagnostics():
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")

    # Select 5 representative records per task (40 records total)
    sample_records = []
    for p in sorted(glob.glob("experiments/ucip23/datasets/task_*.jsonl")):
        with open(p, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        for line in lines[:5]:
            sample_records.append(json.loads(line))

    print(f"Loaded {len(sample_records)} representative frozen records across Tasks A-H.")

    checkpoints = {
        "step800": Path("experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt"),
        "ucip20_best": Path("experiments/ucip20/checkpoints/checkpoint_best.pt")
    }

    diagnostic_results = {}

    for ckpt_key, ckpt_path in checkpoints.items():
        print(f"\nAnalyzing Generation Boundary Logits for {ckpt_key}...")
        model = load_model(ckpt_path)

        ckpt_data = []
        for rec in sample_records:
            rid = rec["id"]
            task = rec["task_type"]
            prompt = rec["prompt"]
            target = rec["target_response"].strip()

            # Encode prompt
            inp = [0] + tok.encode(prompt)
            target_tokens = tok.encode(target)
            expected_first_tok = target_tokens[0] if target_tokens else 1

            curr = torch.tensor([inp], dtype=torch.long)
            with torch.no_grad():
                logits = model(curr)[0, -1, :]
                probs = F.softmax(logits, dim=-1)

            # Metrics
            top20_vals, top20_indices = torch.topk(probs, 20)
            top20_list = [
                {"rank": r + 1, "token_id": int(top20_indices[r].item()), "token_str": tok.decode([int(top20_indices[r].item())]), "prob": float(top20_vals[r].item())}
                for r in range(20)
            ]

            # Expected token rank
            sorted_indices = torch.argsort(probs, descending=True)
            expected_rank = int((sorted_indices == expected_first_tok).nonzero(as_tuple=True)[0].item()) + 1
            expected_prob = float(probs[expected_first_tok].item())

            # Entropy
            entropy = float((-probs * torch.log(probs + 1e-12)).sum().item())
            eos_prob = float(probs[1].item())

            ckpt_data.append({
                "id": rid,
                "task_type": task,
                "target": target,
                "expected_first_token": {
                    "id": expected_first_tok,
                    "str": tok.decode([expected_first_tok]),
                    "prob": expected_prob,
                    "rank": expected_rank
                },
                "entropy": entropy,
                "eos_prob": eos_prob,
                "top1_token": top20_list[0],
                "top5_tokens": top20_list[:5],
                "top20_tokens": top20_list
            })

        diagnostic_results[ckpt_key] = ckpt_data

    # Save summary
    out_file = "experiments/ucip26/logit_diagnostics/generation_boundary_logits.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(diagnostic_results, f, indent=2)

    print(f"Logit Diagnostics Complete. Results saved to {out_file}")


if __name__ == "__main__":
    run_logit_diagnostics()
