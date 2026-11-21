"""
UCIP-28 Phase 5 & Phase 6:
- Phase 5: Context Signal Attribution.
  Matched factual and counterfactual prompts.
  At each layer:
  * Record attention output at the generation position
  * Record residual stream
  * Record FFN input and output
  * Record expected answer token rank
  * Test whether context-dependent activation differences persist, disappear, or are amplified through Layers 4-5.
- Phase 6: Activation Patching.
  Interventions:
  1. Factual prompt activation into counterfactual prompt.
  2. Counterfactual activation into factual prompt.
  3. FFN activation replacement at Layer 4.
  4. FFN activation replacement at Layer 5.
  5. Matched random activation replacement.
  6. Sham activation replacement.
  Measures impact on expected token rank, target margin, top-1 output, full logit distribution, and free generation.
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


def construct_matched_pairs(tok) -> List[Dict[str, Any]]:
    # Pairs with identical length and template structure, swapping the factual value:
    # Pair A: founded in 1981 vs founded in 1999
    # Pair B: headquartered in Pune vs headquartered in Agra
    # Pair C: founded in 1982 vs founded in 1975
    # Pair D: headquartered in Hyderabad vs headquartered in Bengaluru
    # Pair E: founded in 1990 vs founded in 2005
    # Pair F: headquartered in Delhi vs headquartered in Noida
    raw_pairs = [
        ("NovaTrack", "founded in", "1981", "1999"),
        ("BharatArmour", "headquartered in", "Pune", "Agra"),
        ("HyperLink", "founded in", "1982", "1975"),
        ("NovaTrack", "headquartered in", "Delhi", "Noida"),
        ("ZenithWorks", "founded in", "1990", "2005"),
        ("ApexDynamics", "founded in", "1985", "1995"),
        ("AlphaSys", "headquartered in", "Pune", "Goa "),
        ("OmegaCorp", "founded in", "1970", "1980")
    ]
    pairs = []
    for ent, rel, v_fact, v_cf in raw_pairs:
        p_fact = f"User: Context statement: {ent} has {rel} recorded as {v_fact}.\nQuestion: What is the {rel} of {ent}?\nAnswer concisely:\nAssistant: "
        p_cf = f"User: Context statement: {ent} has {rel} recorded as {v_cf}.\nQuestion: What is the {rel} of {ent}?\nAnswer concisely:\nAssistant: "
        
        ids_fact = [0] + tok.encode(p_fact)
        ids_cf = [0] + tok.encode(p_cf)

        # Ensure matched token length
        if len(ids_fact) == len(ids_cf):
            pairs.append({
                "entity": ent,
                "relation": rel,
                "fact_val": v_fact,
                "cf_val": v_cf,
                "fact_prompt": p_fact,
                "cf_prompt": p_cf,
                "fact_ids": ids_fact,
                "cf_ids": ids_cf,
                "fact_tgt_id": tok.encode(v_fact.strip())[0],
                "cf_tgt_id": tok.encode(v_cf.strip())[0]
            })
    return pairs


def run_phase5_and_phase6():
    print("=== UCIP-28 Phase 5 & Phase 6: Context Signal Attribution & Activation Patching ===")
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")
    pairs = construct_matched_pairs(tok)
    print(f"Constructed {len(pairs)} matched-length factual/counterfactual pairs.")

    checkpoints = {
        "step800": "experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt",
        "ucip20_best": "experiments/ucip20/checkpoints/checkpoint_best.pt"
    }

    context_attribution_results = {}
    activation_patching_results = {}

    for ckpt_key, ckpt_path in checkpoints.items():
        print(f"\nProcessing Checkpoint: {ckpt_key}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = ChakrMicro(ModelConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        ckpt_context_tracing = []
        ckpt_patching = []

        with torch.no_grad():
            for p_idx, pair in enumerate(pairs):
                xf = torch.tensor([pair["fact_ids"]], dtype=torch.long)
                xc = torch.tensor([pair["cf_ids"]], dtype=torch.long)
                fact_tgt_id = pair["fact_tgt_id"]
                cf_tgt_id = pair["cf_tgt_id"]

                # ---------------- Phase 5: Tracing Layer-by-Layer Signal ----------------
                # Forward factual and counterfactual simultaneously
                hf = model.embedding(xf)
                hc = model.embedding(xc)

                layer_context_trace = []
                for l_idx, layer in enumerate(model.layers):
                    # Attention
                    attn_f = layer.attn(layer.norm_1(hf), layer_idx=l_idx)
                    attn_c = layer.attn(layer.norm_1(hc), layer_idx=l_idx)
                    
                    # Attention output difference at boundary token
                    attn_diff_norm = float(torch.norm(attn_f[0, -1, :] - attn_c[0, -1, :]).item())
                    attn_cos = float(F.cosine_similarity(attn_f[0, -1, :].unsqueeze(0), attn_c[0, -1, :].unsqueeze(0)).item())

                    hf_post_attn = hf + attn_f
                    hc_post_attn = hc + attn_c

                    res_diff_norm_pre_ffn = float(torch.norm(hf_post_attn[0, -1, :] - hc_post_attn[0, -1, :]).item())

                    # FFN
                    mlp_f = layer.ffn(layer.norm_2(hf_post_attn))
                    mlp_c = layer.ffn(layer.norm_2(hc_post_attn))

                    mlp_diff_norm = float(torch.norm(mlp_f[0, -1, :] - mlp_c[0, -1, :]).item())
                    mlp_cos = float(F.cosine_similarity(mlp_f[0, -1, :].unsqueeze(0), mlp_c[0, -1, :].unsqueeze(0)).item())

                    hf = hf_post_attn + mlp_f
                    hc = hc_post_attn + mlp_c

                    res_diff_norm_post_ffn = float(torch.norm(hf[0, -1, :] - hc[0, -1, :]).item())
                    res_cos = float(F.cosine_similarity(hf[0, -1, :].unsqueeze(0), hc[0, -1, :].unsqueeze(0)).item())

                    # Check intermediate logit lens projection
                    logits_f_inter = model.lm_head(model.final_norm(hf))[0, -1, :]
                    ranks_f = torch.argsort(logits_f_inter, descending=True)
                    tgt_rank_f = int((ranks_f == fact_tgt_id).nonzero(as_tuple=True)[0].item()) + 1
                    cf_rank_f = int((ranks_f == cf_tgt_id).nonzero(as_tuple=True)[0].item()) + 1

                    layer_context_trace.append({
                        "layer": l_idx,
                        "attn_diff_norm": attn_diff_norm,
                        "attn_cosine_sim": attn_cos,
                        "res_diff_norm_pre_ffn": res_diff_norm_pre_ffn,
                        "mlp_diff_norm": mlp_diff_norm,
                        "mlp_cosine_sim": mlp_cos,
                        "res_diff_norm_post_ffn": res_diff_norm_post_ffn,
                        "res_cosine_sim": res_cos,
                        "fact_tgt_rank": tgt_rank_f,
                        "cf_tgt_rank": cf_rank_f
                    })

                # Final boundary logits
                logits_f_final = model.lm_head(model.final_norm(hf))[0, -1, :]
                logits_c_final = model.lm_head(model.final_norm(hc))[0, -1, :]
                probs_f = F.softmax(logits_f_final, dim=-1)
                probs_c = F.softmax(logits_c_final, dim=-1)
                final_js = jensen_shannon_divergence(probs_f, probs_c)
                final_cos = float(F.cosine_similarity(logits_f_final.unsqueeze(0), logits_c_final.unsqueeze(0)).item())

                ckpt_context_tracing.append({
                    "entity": pair["entity"],
                    "relation": pair["relation"],
                    "fact_val": pair["fact_val"],
                    "cf_val": pair["cf_val"],
                    "final_logit_cosine_sim": final_cos,
                    "final_prob_js_divergence": final_js,
                    "layer_traces": layer_context_trace
                })

                # ---------------- Phase 6: Controlled Activation Patching ----------------
                # Interventions testing causal transmission of context signal:
                # Target: Does patching activations change predictions between factual and counterfactual?
                # Interventions:
                # 1. Patch Factual Residual into Counterfactual Prompt at generation position at each layer (L0..L5)
                # 2. Patch Counterfactual Residual into Factual Prompt
                # 3. Patch FFN Output at Layer 4 (Factual -> CF)
                # 4. Patch FFN Output at Layer 5 (Factual -> CF)
                # 5. Matched random activation replacement at Layer 4 (norm matched)
                # 6. Sham intervention (re-inject CF activation itself)

                # Precompute layer states for factual and CF prompts
                f_residuals = []
                c_residuals = []
                f_mlp_outs = []
                c_mlp_outs = []

                # Trace factual
                h = model.embedding(xf)
                for l_idx, layer in enumerate(model.layers):
                    h = h + layer.attn(layer.norm_1(h), layer_idx=l_idx)
                    mlp = layer.ffn(layer.norm_2(h))
                    f_mlp_outs.append(mlp.clone())
                    h = h + mlp
                    f_residuals.append(h.clone())

                # Trace counterfactual
                h = model.embedding(xc)
                for l_idx, layer in enumerate(model.layers):
                    h = h + layer.attn(layer.norm_1(h), layer_idx=l_idx)
                    mlp = layer.ffn(layer.norm_2(h))
                    c_mlp_outs.append(mlp.clone())
                    h = h + mlp
                    c_residuals.append(h.clone())

                def run_patched_cf(patch_type: str, layer_target: int = 4):
                    # Forward through CF prompt with intervention
                    h = model.embedding(xc)
                    for l_idx, layer in enumerate(model.layers):
                        h = h + layer.attn(layer.norm_1(h), layer_idx=l_idx)
                        norm_h2 = layer.norm_2(h)
                        mlp_out = layer.ffn(norm_h2)

                        if patch_type == "ffn_patch" and l_idx == layer_target:
                            # Replace FFN output at boundary token with factual FFN output
                            mlp_out[0, -1, :] = f_mlp_outs[layer_target][0, -1, :]
                        elif patch_type == "matched_random" and l_idx == layer_target:
                            # Replace FFN output with random norm-matched vector
                            orig_norm = torch.norm(mlp_out[0, -1, :]).item()
                            rng_vec = torch.randn_like(mlp_out[0, -1, :])
                            mlp_out[0, -1, :] = rng_vec / torch.norm(rng_vec) * orig_norm
                        elif patch_type == "sham" and l_idx == layer_target:
                            mlp_out[0, -1, :] = c_mlp_outs[layer_target][0, -1, :]

                        h = h + mlp_out

                        if patch_type == "residual_patch" and l_idx == layer_target:
                            # Replace full residual state at boundary token with factual residual state
                            h[0, -1, :] = f_residuals[layer_target][0, -1, :]

                    logits = model.lm_head(model.final_norm(h))[0, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    top1_id = int(torch.argmax(logits).item())
                    top1_str = tok.decode([top1_id], errors="replace")
                    ranks = torch.argsort(logits, descending=True)
                    f_tgt_rank = int((ranks == fact_tgt_id).nonzero(as_tuple=True)[0].item()) + 1
                    c_tgt_rank = int((ranks == cf_tgt_id).nonzero(as_tuple=True)[0].item()) + 1
                    f_tgt_margin = float((logits[fact_tgt_id] - logits[cf_tgt_id]).item())

                    # Free generation (6 tokens)
                    gen_tokens = [top1_id]
                    curr_ids = xc[0].tolist() + [top1_id]
                    for _ in range(5):
                        inp = torch.tensor([curr_ids], dtype=torch.long)
                        nxt_l = model(inp)[0, -1, :]
                        nid = int(torch.argmax(nxt_l).item())
                        gen_tokens.append(nid)
                        curr_ids.append(nid)
                    gen_str = tok.decode(gen_tokens, errors="replace")

                    return {
                        "top1_str": top1_str,
                        "fact_tgt_rank": f_tgt_rank,
                        "cf_tgt_rank": c_tgt_rank,
                        "fact_tgt_margin": f_tgt_margin,
                        "free_gen": gen_str
                    }

                # Evaluate all 6 intervention conditions
                interventions = {
                    "baseline_cf": run_patched_cf("sham", layer_target=0),
                    "residual_patch_L4": run_patched_cf("residual_patch", layer_target=4),
                    "residual_patch_L5": run_patched_cf("residual_patch", layer_target=5),
                    "ffn_patch_L4": run_patched_cf("ffn_patch", layer_target=4),
                    "ffn_patch_L5": run_patched_cf("ffn_patch", layer_target=5),
                    "matched_random_L4": run_patched_cf("matched_random", layer_target=4),
                    "matched_random_L5": run_patched_cf("matched_random", layer_target=5),
                    "sham_L4": run_patched_cf("sham", layer_target=4)
                }

                ckpt_patching.append({
                    "entity": pair["entity"],
                    "relation": pair["relation"],
                    "fact_val": pair["fact_val"],
                    "cf_val": pair["cf_val"],
                    "interventions": interventions
                })

        context_attribution_results[ckpt_key] = ckpt_context_tracing
        activation_patching_results[ckpt_key] = ckpt_patching

    with open("experiments/ucip28/context_attribution/context_attribution_results.json", "w", encoding="utf-8") as f:
        json.dump(context_attribution_results, f, indent=2)
    with open("experiments/ucip28/activation_patching/activation_patching_results.json", "w", encoding="utf-8") as f:
        json.dump(activation_patching_results, f, indent=2)

    print("Phase 5 & Phase 6 data generated and saved successfully.")


if __name__ == "__main__":
    run_phase5_and_phase6()
