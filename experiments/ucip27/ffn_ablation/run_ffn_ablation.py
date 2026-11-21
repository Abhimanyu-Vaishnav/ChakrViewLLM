"""
UCIP-27 Phase 4: Isolated Layer-Wise FFN Ablation Experiments.
For each layer l in [0, 1, 2, 3, 4, 5]:
- Forward pass with layer l FFN (MLP) output ablated (zeroed out)
- Forward pass with all FFN outputs ablated (attention only)
- Measures impact on:
  1. Final boundary logits and JS divergence from unablated baseline
  2. Top-1 generated token
  3. Probability and rank of expected target token
  4. Attractor persistence ('#' in Step-800, 'A'/'Analyst' in UCIP-20)
  5. Free autoregressive generation across 8 steps
"""

from __future__ import annotations
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


def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(torch.log(m + 1e-12), p, reduction="sum")
    kl_qm = F.kl_div(torch.log(m + 1e-12), q, reduction="sum")
    js = 0.5 * (kl_pm + kl_qm)
    return float(math.sqrt(max(0.0, js.item())))


def forward_with_ffn_ablation(
    model: ChakrMicro,
    input_ids: List[int],
    ablate_layer: int | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor([input_ids], dtype=torch.long)
    h = model.embedding(x)

    for l_idx, layer in enumerate(model.layers):
        norm_h1 = layer.norm_1(h)
        attn_out = layer.attn(norm_h1, layer_idx=l_idx)
        h = h + attn_out

        norm_h2 = layer.norm_2(h)
        mlp_out = layer.ffn(norm_h2)
        if ablate_layer == l_idx:
            mlp_out = torch.zeros_like(mlp_out)
        h = h + mlp_out

    h_final = model.final_norm(h)
    logits = model.lm_head(h_final)[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    return logits, probs


def safe_decode(tok, token_ids: List[int]) -> str:
    try:
        return tok.decode(token_ids, errors="replace")
    except Exception:
        return str(token_ids)


def run_ffn_ablation():
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")
    checkpoints = {
        "step800": Path("experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt"),
        "ucip20_best": Path("experiments/ucip20/checkpoints/checkpoint_best.pt")
    }

    test_prompts = [
        ("task_c_1981", "User: Context statement: NovaTrack has founded in recorded as 1981.\nQuestion: What is the founded in of NovaTrack?\nAnswer concisely:\nAssistant: ", "1981"),
        ("task_c_pune", "User: Context statement: BharatArmour has headquartered in recorded as Pune.\nQuestion: What is the headquartered in of BharatArmour?\nAnswer concisely:\nAssistant: ", "Pune"),
        ("task_a_yes", "User: Text snippet 1 discusses NovaTrack. Text snippet 2 discusses NovaTrack.\nQuestion: Do both snippets refer to the identical entity name?\nAnswer concisely with Yes or No:\nAssistant: ", "Yes"),
        ("task_h_abstain", "User: Recorded information covers NovaTrack.\nQuestion: What is the details for UnregisteredEntity_999?\nAnswer concisely:\nAssistant: ", "Information not provided in context.")
    ]

    ablation_results = {}

    for ckpt_key, ckpt_path in checkpoints.items():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ChakrMicro(ModelConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        ckpt_cases = []
        for pid, prompt, target in test_prompts:
            tgt_tok = tok.encode(target)[0]
            inp_ids = [0] + tok.encode(prompt)

            # Unablated baseline
            base_logits, base_probs = forward_with_ffn_ablation(model, inp_ids, ablate_layer=None)
            base_top1 = int(torch.argmax(base_probs).item())
            base_tgt_rank = int((torch.argsort(base_probs, descending=True) == tgt_tok).nonzero(as_tuple=True)[0].item()) + 1

            # Layer-wise FFN ablations
            layer_ablations = []
            for l in range(6):
                abl_logits, abl_probs = forward_with_ffn_ablation(model, inp_ids, ablate_layer=l)
                abl_top1 = int(torch.argmax(abl_probs).item())
                abl_tgt_rank = int((torch.argsort(abl_probs, descending=True) == tgt_tok).nonzero(as_tuple=True)[0].item()) + 1
                js_div = jensen_shannon_divergence(base_probs, abl_probs)
                cos_sim = float(F.cosine_similarity(base_logits.unsqueeze(0), abl_logits.unsqueeze(0)).item())

                # Generate 6 tokens under FFN ablation
                curr_inp = torch.tensor([inp_ids], dtype=torch.long)
                gen_toks = []
                for _ in range(6):
                    with torch.no_grad():
                        l_step, _ = forward_with_ffn_ablation(model, curr_inp[0].tolist(), ablate_layer=l)
                        nxt = int(torch.argmax(l_step).item())
                        if nxt == 1:
                            break
                        gen_toks.append(nxt)
                        curr_inp = torch.cat([curr_inp, torch.tensor([[nxt]])], dim=1)

                layer_ablations.append({
                    "ablated_layer": l,
                    "top1_token": safe_decode(tok, [abl_top1]),
                    "top1_token_id": abl_top1,
                    "target_token_rank": abl_tgt_rank,
                    "target_token_prob": float(abl_probs[tgt_tok].item()),
                    "target_rank_delta": abl_tgt_rank - base_tgt_rank,
                    "cosine_similarity_to_unablated": cos_sim,
                    "js_divergence_from_unablated": js_div,
                    "generated_text": safe_decode(tok, gen_toks)
                })

            # Full FFN Ablation (Attention only)
            x = torch.tensor([inp_ids], dtype=torch.long)
            h = model.embedding(x)
            for l_idx, layer in enumerate(model.layers):
                norm_h1 = layer.norm_1(h)
                attn_out = layer.attn(norm_h1, layer_idx=l_idx)
                h = h + attn_out
            h_final = model.final_norm(h)
            all_abl_logits = model.lm_head(h_final)[0, -1, :]
            all_abl_probs = F.softmax(all_abl_logits, dim=-1)
            all_top1 = int(torch.argmax(all_abl_probs).item())
            all_tgt_rank = int((torch.argsort(all_abl_probs, descending=True) == tgt_tok).nonzero(as_tuple=True)[0].item()) + 1

            ckpt_cases.append({
                "prompt_id": pid,
                "target": target,
                "base_top1_token": safe_decode(tok, [base_top1]),
                "base_target_rank": base_tgt_rank,
                "base_target_prob": float(base_probs[tgt_tok].item()),
                "layer_ablations": layer_ablations,
                "all_ffn_ablated_top1": safe_decode(tok, [all_top1]),
                "all_ffn_ablated_target_rank": all_tgt_rank,
                "all_ffn_ablated_cos_sim": float(F.cosine_similarity(base_logits.unsqueeze(0), all_abl_logits.unsqueeze(0)).item()),
                "all_ffn_ablated_js_div": jensen_shannon_divergence(base_probs, all_abl_probs)
            })

        ablation_results[ckpt_key] = ckpt_cases

    out_file = "experiments/ucip27/ffn_ablation/ffn_ablation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    print(f"FFN Ablation Complete. Saved to {out_file}")


if __name__ == "__main__":
    run_ffn_ablation()
