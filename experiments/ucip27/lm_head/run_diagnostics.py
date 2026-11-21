"""
UCIP-27 Phase 5 & 6: Residual Interventions, Normalization and LM-Head Diagnostics.
Evaluates:
1. Residual Stream Activation Patching:
   - Does replacing the generation-boundary residual state with the hidden state from the target token position restore the target token to rank 1?
   - What is the margin of the target token before and after final RMSNorm?
2. Final RMSNorm Scaling and Attractor Amplification:
   - Evaluates input norm, scale parameter, and output norm.
3. LM-Head Logit Lens & Unembedding Projections:
   - Computes unembedding dot products across intermediate layer hidden states.
   - Determines at which layer the target token was most salient, and where the attractor overtakes it.
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


def run_residual_and_lm_head_diagnostics():
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")
    ckpt_path = Path("experiments/ucip20/checkpoints/checkpoint_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = ChakrMicro(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prompt = "User: Context statement: NovaTrack has founded in recorded as 1981.\nQuestion: What is the founded in of NovaTrack?\nAnswer concisely:\nAssistant: "
    target = "1981"
    target_tok = tok.encode(target)[0]  # token for 1981

    input_ids = [0] + tok.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long)
    seq_len = x.shape[1]

    # Find position of '1981' in prompt
    pos_target = [i for i, t in enumerate(input_ids) if t == target_tok][0]
    print(f"Target token '{target}' (ID {target_tok}) located at prompt index {pos_target} of {seq_len}")

    # Forward pass collecting hidden states at each layer
    h = model.embedding(x)
    layer_states = [h]

    for l_idx, layer in enumerate(model.layers):
        norm_h1 = layer.norm_1(h)
        attn_out = layer.attn(norm_h1, layer_idx=l_idx)
        h = h + attn_out

        norm_h2 = layer.norm_2(h)
        mlp_out = layer.ffn(norm_h2)
        h = h + mlp_out
        layer_states.append(h)

    # 1. Logit Lens: Project intermediate layer representations at boundary (index -1) to logits
    logit_lens_data = []
    for l_idx, h_l in enumerate(layer_states):
        # Apply final norm and lm_head
        h_boundary = h_l[:, -1, :]
        normed = model.final_norm(h_boundary)
        logits = model.lm_head(normed)[0]
        probs = F.softmax(logits, dim=-1)

        tgt_rank = int((torch.argsort(probs, descending=True) == target_tok).nonzero(as_tuple=True)[0].item()) + 1
        tgt_prob = float(probs[target_tok].item())
        top1_tok = int(torch.argmax(probs).item())
        top1_prob = float(probs[top1_tok].item())
        top1_str = tok.decode([top1_tok], errors="replace")

        logit_lens_data.append({
            "stage": f"layer_{l_idx}" if l_idx > 0 else "embedding",
            "target_rank": tgt_rank,
            "target_prob": tgt_prob,
            "top1_token": top1_str,
            "top1_prob": top1_prob,
            "margin_target_minus_top1": tgt_prob - top1_prob
        })

    # 2. Causal Activation Patching Intervention:
    # What happens if we copy the hidden representation of the target token position (pos_target)
    # into the generation boundary position (-1) at layer 3, 4, 5?
    patching_results = []
    for patch_layer in [2, 3, 4, 5]:
        h_patched = layer_states[patch_layer].clone()
        # Patch boundary with target position vector
        h_patched[0, -1, :] = h_patched[0, pos_target, :]

        # Continue forward from patch_layer to output
        curr_h = h_patched
        for l_idx in range(patch_layer, len(model.layers)):
            layer = model.layers[l_idx]
            norm_h1 = layer.norm_1(curr_h)
            attn_out = layer.attn(norm_h1, layer_idx=l_idx)
            curr_h = curr_h + attn_out

            norm_h2 = layer.norm_2(curr_h)
            mlp_out = layer.ffn(norm_h2)
            curr_h = curr_h + mlp_out

        h_final = model.final_norm(curr_h)
        logits = model.lm_head(h_final)[0, -1, :]
        probs = F.softmax(logits, dim=-1)

        p_tgt_rank = int((torch.argsort(probs, descending=True) == target_tok).nonzero(as_tuple=True)[0].item()) + 1
        p_tgt_prob = float(probs[target_tok].item())
        p_top1 = int(torch.argmax(probs).item())
        p_top1_str = tok.decode([p_top1], errors="replace")

        patching_results.append({
            "patched_at_layer": patch_layer,
            "resulting_top1": p_top1_str,
            "target_rank": p_tgt_rank,
            "target_prob": p_tgt_prob,
            "restored_target_as_top1": (p_top1 == target_tok)
        })

    # 3. Final RMSNorm vs LM-Head Scaling
    h_pre_norm = layer_states[-1][0, -1, :]
    pre_norm_mag = float(torch.norm(h_pre_norm).item())
    h_post_norm = model.final_norm(layer_states[-1])[0, -1, :]
    post_norm_mag = float(torch.norm(h_post_norm).item())

    # Raw dot products with lm_head weights (target vs attractor)
    # lm_head weight shape: (vocab_size, d_model)
    lm_weights = model.lm_head.weight
    dot_target = float(torch.dot(h_post_norm, lm_weights[target_tok]).item())
    dot_attractor = float(torch.dot(h_post_norm, lm_weights[68]).item())  # token 68 is 'A'

    norm_analysis = {
        "pre_norm_l2_norm": pre_norm_mag,
        "post_norm_l2_norm": post_norm_mag,
        "dot_product_with_target_weight_1981": dot_target,
        "dot_product_with_attractor_weight_A": dot_attractor,
        "raw_logit_margin": dot_target - dot_attractor
    }

    full_results = {
        "logit_lens": logit_lens_data,
        "activation_patching": patching_results,
        "norm_and_head_analysis": norm_analysis
    }

    out_file = "experiments/ucip27/lm_head/residual_and_lm_head_diagnostics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    print("Phase 5 & 6 Residual and LM-Head Diagnostics Complete.")
    print("Logit Lens Trajectory across Layers (Target: '1981'):")
    for l in logit_lens_data:
        print(f"  {l['stage']}: Top-1={repr(l['top1_token'])} ({l['top1_prob']*100:.1f}%), Target Rank={l['target_rank']}, Target Prob={l['target_prob']*100:.3f}%")
    print("\nActivation Patching Results (Copying '1981' context hidden state to boundary):")
    for p in patching_results:
        print(f"  Patched at Layer {p['patched_at_layer']}: Top-1={repr(p['resulting_top1'])}, Target Rank={p['target_rank']}, Target Prob={p['target_prob']*100:.2f}%")


if __name__ == "__main__":
    run_residual_and_lm_head_diagnostics()
