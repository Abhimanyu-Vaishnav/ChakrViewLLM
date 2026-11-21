"""
UCIP-28 Phase 4: FFN Channel Attribution.
Instruments SwiGLU intermediate activations:
- gate = SiLU(x * W_gate)   [512]
- up = x * W_up             [512]
- gated_product = gate * up [512]
- down contribution = gated_product[j] * W_down[:, j]  [d_model=192]
Measures correlation and causal patching of top channels to attractor logit contributions.
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


def load_dataset_prompts(n_samples: int = 32) -> List[Dict[str, Any]]:
    dataset_files = sorted(glob.glob("experiments/ucip23/datasets/task_*.jsonl"))
    items = []
    for fpath in dataset_files:
        with open(fpath, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        for rec in lines:
            items.append({
                "task_id": rec["id"],
                "prompt": rec["prompt"],
                "target": rec["target_response"],
                "entity": rec.get("query_entity", "unknown")
            })
            if len(items) >= n_samples:
                return items
    return items


def run_phase4():
    print("=== UCIP-28 Phase 4: FFN Channel Attribution (SwiGLU Channels) ===")
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")
    items = load_dataset_prompts(n_samples=32)

    checkpoints = {
        "step800": ("experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt", "#"),
        "ucip20_best": ("experiments/ucip20/checkpoints/checkpoint_best.pt", "A")
    }

    channel_analysis_results = {}

    for ckpt_key, (ckpt_path, dominant_attractor_char) in checkpoints.items():
        print(f"\nAnalyzing FFN channels for: {ckpt_key}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ChakrMicro(ModelConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        attractor_token_id = tok.encode(dominant_attractor_char)[0]
        # Vocabulary unembedding vector for the attractor token: W_u[attractor_token_id, :]
        # For tied weights, W_u = embedding.weight: [vocab_size, d_model]
        w_u_attractor = model.embedding.weight[attractor_token_id, :].clone().detach() # [192]

        ckpt_results = {}

        # Focus in depth on Layer 4 and Layer 5 (implicated in UCIP-27), plus Layer 2 as control
        for layer_idx in [2, 4, 5]:
            print(f"  Profiling Layer {layer_idx} SwiGLU 512 channels...")
            ffn_module = model.layers[layer_idx].ffn
            # W_down weight shape: [d_model, hidden_dim] = [192, 512]
            w_down = ffn_module.down_proj.weight.detach() # [192, 512]

            # Direct projection of each channel's output basis vector into attractor unembedding:
            # direct_attractor_weight[j] = dot(w_down[:, j], w_u_attractor)
            direct_attractor_weights = torch.mv(w_down.t(), w_u_attractor) # [512]

            # Collect activations across benchmark items at generation position
            all_gated_products = [] # [N, 512]
            all_gates = []
            all_ups = []
            all_channel_logit_contribs = [] # [N, 512]

            with torch.no_grad():
                for item in items:
                    input_ids = [0] + tok.encode(item["prompt"])
                    x = torch.tensor([input_ids], dtype=torch.long)
                    h = model.embedding(x)
                    for l in range(layer_idx):
                        h = h + model.layers[l].attn(model.layers[l].norm_1(h), layer_idx=l)
                        h = h + model.layers[l].ffn(model.layers[l].norm_2(h))

                    # Enter layer_idx
                    h = h + model.layers[layer_idx].attn(model.layers[layer_idx].norm_1(h), layer_idx=layer_idx)
                    norm_h = model.layers[layer_idx].norm_2(h)
                    pos = -1

                    # SwiGLU internal activations at boundary token
                    x_norm = norm_h[0, pos, :] # [192]
                    gate = F.silu(ffn_module.gate_proj(x_norm)) # [512]
                    up = ffn_module.up_proj(x_norm) # [512]
                    gated = gate * up # [512]

                    all_gates.append(gate.cpu().numpy())
                    all_ups.append(up.cpu().numpy())
                    all_gated_products.append(gated.cpu().numpy())

                    # Channel logit contribution: gated[j] * dot(w_down[:, j], w_u_attractor)
                    channel_contrib = gated * direct_attractor_weights # [512]
                    all_channel_logit_contribs.append(channel_contrib.cpu().numpy())

            gated_arr = np.array(all_gated_products) # [N, 512]
            contrib_arr = np.array(all_channel_logit_contribs) # [N, 512]

            mean_contribs = np.mean(contrib_arr, axis=0) # [512]
            std_contribs = np.std(contrib_arr, axis=0) # [512]
            mean_activations = np.mean(gated_arr, axis=0)

            # Top positive and negative channels by mean logit contribution
            sorted_indices = np.argsort(mean_contribs)[::-1]
            top_10_attractor_promoters = [int(idx) for idx in sorted_indices[:10]]
            top_10_attractor_suppressors = [int(idx) for idx in sorted_indices[-10:]]

            # CAUSAL TEST: Knock out the top 10 attractor promoter channels vs 10 random channels
            # Test whether zeroing top-10 promoter channels breaks the attractor
            causal_patch_results = []
            for item in items[:10]:
                input_ids = [0] + tok.encode(item["prompt"])
                x = torch.tensor([input_ids], dtype=torch.long)

                def forward_ablating_channels(channels_to_zero):
                    h = model.embedding(x)
                    for l in range(model.config.n_layers):
                        h = h + model.layers[l].attn(model.layers[l].norm_1(h), layer_idx=l)
                        norm_h2 = model.layers[l].norm_2(h)
                        if l == layer_idx and channels_to_zero is not None:
                            # Forward through SwiGLU with specific channels zeroed
                            gate_val = F.silu(model.layers[l].ffn.gate_proj(norm_h2))
                            up_val = model.layers[l].ffn.up_proj(norm_h2)
                            gated_val = gate_val * up_val
                            for c in channels_to_zero:
                                gated_val[:, :, c] = 0.0
                            mlp_out = model.layers[l].ffn.down_proj(gated_val)
                        else:
                            mlp_out = model.layers[l].ffn(norm_h2)
                        h = h + mlp_out
                    logits = model.lm_head(model.final_norm(h))[0, -1, :]
                    return logits

                # Baseline
                base_logits = forward_ablating_channels(None)
                base_top1 = tok.decode([int(torch.argmax(base_logits).item())], errors="replace")
                base_attractor_logit = float(base_logits[attractor_token_id].item())

                # Ablate top 1 promoter channel
                top1_abl_logits = forward_ablating_channels([top_10_attractor_promoters[0]])
                top1_abl_top1 = tok.decode([int(torch.argmax(top1_abl_logits).item())], errors="replace")
                top1_abl_attractor_logit = float(top1_abl_logits[attractor_token_id].item())

                # Ablate top 5 promoter channels
                top5_abl_logits = forward_ablating_channels(top_10_attractor_promoters[:5])
                top5_abl_top1 = tok.decode([int(torch.argmax(top5_abl_logits).item())], errors="replace")
                top5_abl_attractor_logit = float(top5_abl_logits[attractor_token_id].item())

                # Ablate top 10 promoter channels
                top10_abl_logits = forward_ablating_channels(top_10_attractor_promoters)
                top10_abl_top1 = tok.decode([int(torch.argmax(top10_abl_logits).item())], errors="replace")
                top10_abl_attractor_logit = float(top10_abl_logits[attractor_token_id].item())

                # Ablate 10 random control channels (fixed seed)
                rng = np.random.RandomState(42 + layer_idx)
                random_10_channels = [int(c) for c in rng.choice(512, size=10, replace=False)]
                rand10_abl_logits = forward_ablating_channels(random_10_channels)
                rand10_abl_top1 = tok.decode([int(torch.argmax(rand10_abl_logits).item())], errors="replace")
                rand10_abl_attractor_logit = float(rand10_abl_logits[attractor_token_id].item())

                causal_patch_results.append({
                    "task_id": item["task_id"],
                    "base_top1": base_top1,
                    "base_attractor_logit": base_attractor_logit,
                    "top1_channel_abl_top1": top1_abl_top1,
                    "top1_channel_delta": top1_abl_attractor_logit - base_attractor_logit,
                    "top5_channel_abl_top1": top5_abl_top1,
                    "top5_channel_delta": top5_abl_attractor_logit - base_attractor_logit,
                    "top10_channel_abl_top1": top10_abl_top1,
                    "top10_channel_delta": top10_abl_attractor_logit - base_attractor_logit,
                    "rand10_channel_abl_top1": rand10_abl_top1,
                    "rand10_channel_delta": rand10_abl_attractor_logit - base_attractor_logit,
                })

            ckpt_results[f"layer_{layer_idx}"] = {
                "top_10_attractor_promoter_channels": top_10_attractor_promoters,
                "top_10_promoter_mean_contribs": [float(mean_contribs[c]) for c in top_10_attractor_promoters],
                "top_10_attractor_suppressor_channels": top_10_attractor_suppressors,
                "top_10_suppressor_mean_contribs": [float(mean_contribs[c]) for c in top_10_attractor_suppressors],
                "mean_channel_contribution_abs": float(np.mean(np.abs(mean_contribs))),
                "causal_channel_interventions": causal_patch_results
            }

        channel_analysis_results[ckpt_key] = ckpt_results

    with open("experiments/ucip28/ffn_channels/ffn_channel_attribution_results.json", "w", encoding="utf-8") as f:
        json.dump(channel_analysis_results, f, indent=2)

    print("Phase 4 FFN channel attribution completed and saved successfully.")


if __name__ == "__main__":
    run_phase4()
