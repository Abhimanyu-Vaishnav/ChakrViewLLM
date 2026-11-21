"""
UCIP-27 Phase 2: Layer-Wise Baseline Tracing Script.
Instruments all 6 transformer layers across representative records from Tasks A-G under:
1. Original factual prompt
2. Matched entity replacement
3. Matched relation replacement
4. Matched value replacement
5. Matched neutral context

For each layer and condition, records:
- Hidden state norm entering layer
- Attention output norm
- FFN output norm
- Hidden state norm leaving layer
- Cosine similarity of attention output across conditions
- Cosine similarity of FFN output across conditions
- Final norm output
- LM-head logits, Jensen-Shannon divergence, and expected answer token margins
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

import numpy as np
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


def trace_forward(model: ChakrMicro, input_ids: List[int]) -> Dict[str, Any]:
    x = torch.tensor([input_ids], dtype=torch.long)
    seq_len = x.shape[1]

    layer_records = []
    h = model.embedding(x)

    for l_idx, layer in enumerate(model.layers):
        h_in_norm = float(torch.norm(h[0, -1, :]).item())

        # Attention
        norm_h1 = layer.norm_1(h)
        attn_out = layer.attn(norm_h1, layer_idx=l_idx)
        attn_out_norm = float(torch.norm(attn_out[0, -1, :]).item())
        h_post_attn = h + attn_out

        # FFN
        norm_h2 = layer.norm_2(h_post_attn)
        mlp_out = layer.ffn(norm_h2)
        mlp_out_norm = float(torch.norm(mlp_out[0, -1, :]).item())
        h = h_post_attn + mlp_out

        h_out_norm = float(torch.norm(h[0, -1, :]).item())

        layer_records.append({
            "layer_idx": l_idx,
            "h_in_boundary": h[0, -1, :].clone(),
            "attn_out_boundary": attn_out[0, -1, :].clone(),
            "mlp_out_boundary": mlp_out[0, -1, :].clone(),
            "h_in_norm": h_in_norm,
            "attn_norm": attn_out_norm,
            "mlp_norm": mlp_out_norm,
            "h_out_norm": h_out_norm
        })

    h_final = model.final_norm(h)
    logits = model.lm_head(h_final)[0, -1, :]
    probs = F.softmax(logits, dim=-1)

    return {
        "layer_records": layer_records,
        "h_final_boundary": h_final[0, -1, :].clone(),
        "logits": logits.clone(),
        "probs": probs.clone()
    }


def run_layer_tracing():
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")
    checkpoints = {
        "step800": Path("experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt"),
        "ucip20_best": Path("experiments/ucip20/checkpoints/checkpoint_best.pt")
    }

    # Test cases: matched length prompt variations
    test_suite = [
        {
            "case_id": "tech_founded_in",
            "base": "User: Context statement: NovaTrack has founded in recorded as 1981.\nQuestion: What is the founded in of NovaTrack?\nAnswer concisely:\nAssistant: ",
            "entity_rep": "User: Context statement: SolarPulse has founded in recorded as 1981.\nQuestion: What is the founded in of SolarPulse?\nAnswer concisely:\nAssistant: ",
            "value_rep": "User: Context statement: NovaTrack has founded in recorded as 2025.\nQuestion: What is the founded in of NovaTrack?\nAnswer concisely:\nAssistant: ",
            "neutral": "User: Context statement: Modern history registers events systematically.\nQuestion: What is the founded in of NovaTrack?\nAnswer concisely:\nAssistant: ",
            "target_val": "1981"
        },
        {
            "case_id": "defense_headquartered_in",
            "base": "User: Context statement: BharatArmour has headquartered in recorded as Pune.\nQuestion: What is the headquartered in of BharatArmour?\nAnswer concisely:\nAssistant: ",
            "entity_rep": "User: Context statement: SagarMarine has headquartered in recorded as Pune.\nQuestion: What is the headquartered in of SagarMarine?\nAnswer concisely:\nAssistant: ",
            "value_rep": "User: Context statement: BharatArmour has headquartered in recorded as Agra.\nQuestion: What is the headquartered in of BharatArmour?\nAnswer concisely:\nAssistant: ",
            "neutral": "User: Context statement: Coastal regions feature prominent harbors naturally.\nQuestion: What is the headquartered in of BharatArmour?\nAnswer concisely:\nAssistant: ",
            "target_val": "Pune"
        }
    ]

    all_tracing_results = {}

    for ckpt_key, ckpt_path in checkpoints.items():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ChakrMicro(ModelConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        case_results = []
        for case in test_suite:
            cid = case["case_id"]
            tgt = case["target_val"]
            tgt_tok = tok.encode(tgt)[0]

            # Trace base
            base_trace = trace_forward(model, [0] + tok.encode(case["base"]))
            ent_trace = trace_forward(model, [0] + tok.encode(case["entity_rep"]))
            val_trace = trace_forward(model, [0] + tok.encode(case["value_rep"]))
            neu_trace = trace_forward(model, [0] + tok.encode(case["neutral"]))

            # Compare layers across base vs value_rep
            layer_comparisons = []
            for l in range(6):
                b_l = base_trace["layer_records"][l]
                v_l = val_trace["layer_records"][l]

                attn_cos = float(F.cosine_similarity(b_l["attn_out_boundary"].unsqueeze(0), v_l["attn_out_boundary"].unsqueeze(0)).item())
                mlp_cos = float(F.cosine_similarity(b_l["mlp_out_boundary"].unsqueeze(0), v_l["mlp_out_boundary"].unsqueeze(0)).item())
                h_cos = float(F.cosine_similarity(b_l["h_in_boundary"].unsqueeze(0), v_l["h_in_boundary"].unsqueeze(0)).item())

                layer_comparisons.append({
                    "layer": l,
                    "hidden_cos_sim": h_cos,
                    "attn_out_cos_sim": attn_cos,
                    "mlp_out_cos_sim": mlp_cos,
                    "attn_norm_base": b_l["attn_norm"],
                    "mlp_norm_base": b_l["mlp_norm"]
                })

            # Compare final outputs
            raw_logit_cos = float(F.cosine_similarity(base_trace["logits"].unsqueeze(0), val_trace["logits"].unsqueeze(0)).item())
            centered_base = base_trace["logits"] - base_trace["logits"].mean()
            centered_val = val_trace["logits"] - val_trace["logits"].mean()
            centered_logit_cos = float(F.cosine_similarity(centered_base.unsqueeze(0), centered_val.unsqueeze(0)).item())

            js_div = jensen_shannon_divergence(base_trace["probs"], val_trace["probs"])

            # Target token rank and margin
            base_probs = base_trace["probs"]
            top1_tok = int(torch.argmax(base_probs).item())
            top1_prob = float(base_probs[top1_tok].item())
            tgt_prob = float(base_probs[tgt_tok].item())
            margin = tgt_prob - top1_prob
            rank = int((torch.argsort(base_probs, descending=True) == tgt_tok).nonzero(as_tuple=True)[0].item()) + 1

            case_results.append({
                "case_id": cid,
                "target_val": tgt,
                "target_tok_id": tgt_tok,
                "target_tok_str": tok.decode([tgt_tok]),
                "layer_comparisons": layer_comparisons,
                "raw_logit_cosine_sim": raw_logit_cos,
                "centered_logit_cosine_sim": centered_logit_cos,
                "jensen_shannon_divergence": js_div,
                "target_rank": rank,
                "target_prob": tgt_prob,
                "top1_prob": top1_prob,
                "target_margin_over_top1": margin,
                "top1_token_str": tok.decode([top1_tok])
            })

        all_tracing_results[ckpt_key] = case_results

    out_file = "experiments/ucip27/layer_tracing/layer_tracing_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_tracing_results, f, indent=2)

    print(f"Layer Tracing Complete. Saved to {out_file}")


if __name__ == "__main__":
    run_layer_tracing()
