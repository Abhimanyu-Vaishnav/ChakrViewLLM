"""
UCIP-26 Phase 5: Controlled Prompt Conditioning Experiments.
Uses matched-length prompt variants to isolate semantic conditioning from positional effects:
Condition 1: Original factual prompt (Context: NovaTrack founded_in 1981)
Condition 2: Entity replacement (Context: SolarPulse founded_in 1981)
Condition 3: Value replacement (Context: NovaTrack founded_in 2015)
Condition 4: Neutral replacement (Context: Item-A feature-B value-C)
Condition 5: Distractor competitor insertion
Condition 6: Length-matched random permutation

Measures:
- Logit changes and cosine similarity of logits at generation boundary
- Probability and rank of original target value
- Free generation output
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def run_conditioning_experiments():
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")
    ckpt_path = Path("experiments/ucip20/checkpoints/checkpoint_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ChakrMicro(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Base factual prompt
    prompt_base = "User: Context statement: NovaTrack has founded in recorded as 1981.\nQuestion: What is the founded in of NovaTrack?\nAnswer concisely:\nAssistant: "
    target_base = "1981"

    # Variant 1: Entity replacement (matched length)
    prompt_entity_rep = "User: Context statement: SolarPulse has founded in recorded as 1981.\nQuestion: What is the founded in of SolarPulse?\nAnswer concisely:\nAssistant: "

    # Variant 2: Value replacement (matched length: 1981 -> 2025)
    prompt_value_rep = "User: Context statement: NovaTrack has founded in recorded as 2025.\nQuestion: What is the founded in of NovaTrack?\nAnswer concisely:\nAssistant: "

    # Variant 3: Unrelated fact context
    prompt_neutral = "User: Context statement: Geography study observes rivers flowing to seas.\nQuestion: What is the founded in of NovaTrack?\nAnswer concisely:\nAssistant: "

    variants = {
        "base_factual": prompt_base,
        "entity_replaced": prompt_entity_rep,
        "value_replaced": prompt_value_rep,
        "neutral_context": prompt_neutral
    }

    results = {}
    base_logits = None

    for name, p in variants.items():
        inp = [0] + tok.encode(p)
        curr = torch.tensor([inp], dtype=torch.long)
        with torch.no_grad():
            logits = model(curr)[0, -1, :]
            probs = F.softmax(logits, dim=-1)

            # Generate 8 tokens
            gen_toks = []
            c_gen = curr.clone()
            for _ in range(8):
                nxt = int(torch.argmax(model(c_gen)[0, -1, :]).item())
                if nxt == 1:
                    break
                gen_toks.append(nxt)
                c_gen = torch.cat([c_gen, torch.tensor([[nxt]])], dim=1)

            gen_text = tok.decode(gen_toks)

        if base_logits is None:
            base_logits = logits.clone()
            cos_sim = 1.0
        else:
            cos_sim = float(F.cosine_similarity(base_logits.unsqueeze(0), logits.unsqueeze(0)).item())

        # Target token stats (1981 vs 2025)
        t_1981 = tok.encode("1981")[0]
        t_2025 = tok.encode("2025")[0]

        results[name] = {
            "prompt": p,
            "cosine_similarity_to_base": cos_sim,
            "top1_token": tok.decode([int(torch.argmax(logits).item())]),
            "generated_text": gen_text,
            "prob_1981": float(probs[t_1981].item()),
            "prob_2025": float(probs[t_2025].item()),
            "rank_1981": int((torch.argsort(probs, descending=True) == t_1981).nonzero(as_tuple=True)[0].item()) + 1,
            "rank_2025": int((torch.argsort(probs, descending=True) == t_2025).nonzero(as_tuple=True)[0].item()) + 1
        }

    out_file = "experiments/ucip26/conditioning/conditioning_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Phase 5 Conditioning Experiments Complete.")
    for k, v in results.items():
        print(f"Condition: {k}")
        print(f"  Cosine Sim: {v['cosine_similarity_to_base']:.4f}")
        print(f"  Generated: {repr(v['generated_text'])}")
        print(f"  Prob 1981: {v['prob_1981']:.6f} (Rank {v['rank_1981']}) | Prob 2025: {v['prob_2025']:.6f} (Rank {v['rank_2025']})")


if __name__ == "__main__":
    run_conditioning_experiments()
