"""
UCIP-28 Phase 2 & Phase 3:
- Phase 2: Individual FFN Layer Attribution across all 6 layers across multiple benchmark items.
- Phase 3: Matched-Norm Controls (Zero, Matched-Norm Neutral, Matched Random, Sham, Baseline).
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


def load_dataset_prompts(n_samples_per_task: int = 5) -> List[Dict[str, Any]]:
    dataset_files = sorted(glob.glob("experiments/ucip23/datasets/task_*.jsonl"))
    items = []
    for fpath in dataset_files:
        with open(fpath, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        for rec in lines[:n_samples_per_task]:
            items.append({
                "task_id": rec["id"],
                "task_type": rec["task_type"],
                "prompt": rec["prompt"],
                "target": rec["target_response"],
                "entity": rec.get("query_entity", "unknown")
            })
    return items


def run_phase2_and_phase3():
    print("=== UCIP-28 Phase 2 & Phase 3: FFN Layer Attribution & Matched-Norm Controls ===")
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")
    items = load_dataset_prompts(n_samples_per_task=4)  # 8 tasks * 4 = 32 distinct items
    print(f"Loaded {len(items)} evaluation benchmark items across all tasks.")

    checkpoints = {
        "step800": "experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt",
        "ucip20_best": "experiments/ucip20/checkpoints/checkpoint_best.pt"
    }

    layer_attribution_results = {}
    matched_controls_results = {}

    for ckpt_key, ckpt_path in checkpoints.items():
        print(f"\nEvaluating checkpoint: {ckpt_key}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ChakrMicro(ModelConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        ckpt_layer_attr = []
        ckpt_matched_ctrl = []

        with torch.no_grad():
            for item in items:
                prompt = item["prompt"]
                target = item["target"]
                tgt_toks = tok.encode(target)
                tgt_id = tgt_toks[0] if len(tgt_toks) > 0 else 0
                input_ids = [0] + tok.encode(prompt)
                x = torch.tensor([input_ids], dtype=torch.long)

                # 1. Baseline forward & capture residual stream norm and FFN outputs
                h = model.embedding(x)
                layer_ffn_outs = []
                layer_res_norms_before_ffn = []
                layer_res_norms_after_ffn = []

                for l_idx, layer in enumerate(model.layers):
                    h = h + layer.attn(layer.norm_1(h), layer_idx=l_idx)
                    layer_res_norms_before_ffn.append(float(torch.norm(h[0, -1, :]).item()))
                    mlp_out = layer.ffn(layer.norm_2(h))
                    layer_ffn_outs.append(mlp_out)
                    h = h + mlp_out
                    layer_res_norms_after_ffn.append(float(torch.norm(h[0, -1, :]).item()))

                base_logits = model.lm_head(model.final_norm(h))[0, -1, :]
                base_probs = F.softmax(base_logits, dim=-1)
                base_top1_id = int(torch.argmax(base_logits).item())
                base_top1_str = tok.decode([base_top1_id], errors="replace")
                base_ranks = torch.argsort(base_logits, descending=True)
                base_tgt_rank = int((base_ranks == tgt_id).nonzero(as_tuple=True)[0].item()) + 1
                base_entropy = float(-torch.sum(base_probs * torch.log(base_probs + 1e-12)).item())

                # --- Phase 2: Layer-Wise FFN Ablations (Layers 0..5) ---
                item_layer_records = []
                for ablate_l in range(6):
                    h = model.embedding(x)
                    res_norms = []
                    for l_idx, layer in enumerate(model.layers):
                        h = h + layer.attn(layer.norm_1(h), layer_idx=l_idx)
                        mlp_out = layer.ffn(layer.norm_2(h))
                        if l_idx == ablate_l:
                            mlp_out = torch.zeros_like(mlp_out)
                        h = h + mlp_out
                        res_norms.append(float(torch.norm(h[0, -1, :]).item()))

                    logits = model.lm_head(model.final_norm(h))[0, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    top5_ids = torch.topk(logits, 5).indices.tolist()
                    top5_strs = [tok.decode([tid], errors="replace") for tid in top5_ids]
                    ranks = torch.argsort(logits, descending=True)
                    tgt_rank = int((ranks == tgt_id).nonzero(as_tuple=True)[0].item()) + 1
                    js = jensen_shannon_divergence(base_probs, probs)
                    entropy = float(-torch.sum(probs * torch.log(probs + 1e-12)).item())
                    cos_sim = float(F.cosine_similarity(base_logits.unsqueeze(0), logits.unsqueeze(0)).item())

                    item_layer_records.append({
                        "ablated_layer": ablate_l,
                        "top1_str": top5_strs[0],
                        "top5_strs": top5_strs,
                        "target_rank": tgt_rank,
                        "target_rank_delta": tgt_rank - base_tgt_rank,
                        "entropy": entropy,
                        "js_divergence": js,
                        "cosine_similarity": cos_sim,
                        "final_res_norm": res_norms[-1]
                    })

                ckpt_layer_attr.append({
                    "task_id": item["task_id"],
                    "entity": item["entity"],
                    "base_top1_str": base_top1_str,
                    "base_tgt_rank": base_tgt_rank,
                    "base_entropy": base_entropy,
                    "layer_records": item_layer_records
                })

                # --- Phase 3: Matched-Norm Controls at Layers 4 and 5 ---
                # Compare for Layer 4 & Layer 5:
                # 1. Baseline
                # 2. Zero ablation
                # 3. Matched-norm neutral replacement (unit vector along mean activation scaled to original norm)
                # 4. Random replacement with matched mean and variance (Gaussian with same mean & std)
                # 5. Sham intervention (replaces activation with identical activation)
                item_ctrl_records = {}
                for target_l in [4, 5]:
                    layer_ctrls = {}
                    orig_act = layer_ffn_outs[target_l] # [1, T, 192]
                    orig_norm = torch.norm(orig_act[0, -1, :]).item()
                    mean_val = float(orig_act[0, -1, :].mean().item())
                    std_val = float(orig_act[0, -1, :].std().item())

                    # Types of interventions
                    intervention_types = ["baseline", "zero", "matched_norm_neutral", "matched_random", "sham"]
                    for itype in intervention_types:
                        torch.manual_seed(42 + target_l)
                        h = model.embedding(x)
                        for l_idx, layer in enumerate(model.layers):
                            h = h + layer.attn(layer.norm_1(h), layer_idx=l_idx)
                            mlp_out = layer.ffn(layer.norm_2(h))
                            if l_idx == target_l:
                                if itype == "baseline":
                                    pass
                                elif itype == "zero":
                                    mlp_out = torch.zeros_like(mlp_out)
                                elif itype == "matched_norm_neutral":
                                    # Neutral constant direction with identical L2 norm
                                    ones = torch.ones_like(mlp_out)
                                    neutral = ones / torch.norm(ones[0, -1, :]) * orig_norm
                                    mlp_out = neutral
                                elif itype == "matched_random":
                                    # Random Gaussian with matched mean and std
                                    rand = torch.randn_like(mlp_out) * std_val + mean_val
                                    # Ensure exact norm match
                                    rand = rand / torch.norm(rand[0, -1, :]) * orig_norm
                                    mlp_out = rand
                                elif itype == "sham":
                                    mlp_out = mlp_out.clone()
                            h = h + mlp_out

                        logits = model.lm_head(model.final_norm(h))[0, -1, :]
                        probs = F.softmax(logits, dim=-1)
                        top1_id = int(torch.argmax(logits).item())
                        top1_str = tok.decode([top1_id], errors="replace")
                        ranks = torch.argsort(logits, descending=True)
                        tgt_rank = int((ranks == tgt_id).nonzero(as_tuple=True)[0].item()) + 1
                        js = jensen_shannon_divergence(base_probs, probs)
                        cos_sim = float(F.cosine_similarity(base_logits.unsqueeze(0), logits.unsqueeze(0)).item())
                        res_norm = float(torch.norm(h[0, -1, :]).item())

                        layer_ctrls[itype] = {
                            "top1_str": top1_str,
                            "target_rank": tgt_rank,
                            "cosine_sim": cos_sim,
                            "js_divergence": js,
                            "residual_norm": res_norm
                        }
                    item_ctrl_records[f"layer_{target_l}"] = layer_ctrls

                ckpt_matched_ctrl.append({
                    "task_id": item["task_id"],
                    "entity": item["entity"],
                    "controls": item_ctrl_records
                })

        layer_attribution_results[ckpt_key] = ckpt_layer_attr
        matched_controls_results[ckpt_key] = ckpt_matched_ctrl

    with open("experiments/ucip28/layer_ablation/layer_attribution_results.json", "w", encoding="utf-8") as f:
        json.dump(layer_attribution_results, f, indent=2)
    with open("experiments/ucip28/matched_controls/matched_controls_results.json", "w", encoding="utf-8") as f:
        json.dump(matched_controls_results, f, indent=2)

    print("Phase 2 & Phase 3 data generated and saved successfully.")


if __name__ == "__main__":
    run_phase2_and_phase3()
