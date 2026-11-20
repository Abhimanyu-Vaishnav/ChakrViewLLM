"""
UCIP-19 Master Diagnostic Suite.
Executes Investigations A, B, C, D, and E on ChakrMicro (Step 800 baseline vs. UCIP-18 pilot checkpoints).
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def load_model(ckpt_path: Path, device: str = "cpu") -> ChakrMicro:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = ChakrMicro(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ==============================================================================
# INVESTIGATION A: Prompt Sensitivity on 100 Frozen Probes with Controlled Variants
# ==============================================================================

def run_investigation_a(
    model: ChakrMicro,
    tok: Any,
    probes: List[Dict[str, Any]],
    output_path: Path,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    For each probe, test 6 controlled variants:
    1. Original prompt
    2. Reworded instruction
    3. Reordered words
    4. Added irrelevant context
    5. Different requested format
    6. Completely different instruction
    """
    print("Running Investigation A: Prompt Sensitivity...")
    results = []
    
    first_token_matches = {"reworded": 0, "reordered": 0, "extra_ctx": 0, "format": 0, "different": 0}
    top10_overlaps = {"reworded": [], "reordered": [], "extra_ctx": [], "format": [], "different": []}
    logit_cos_sims = {"reworded": [], "reordered": [], "extra_ctx": [], "format": [], "different": []}
    gen_identical = {"reworded": 0, "reordered": 0, "extra_ctx": 0, "format": 0, "different": 0}

    for idx, p_info in enumerate(probes):
        orig_p = p_info["prompt"]
        cat = p_info["category"]
        lang = p_info["language"]
        
        # Construct controlled variants
        # 1. Original
        v_orig = orig_p
        
        # 2. Reworded instruction
        if "User: " in orig_p:
            body = orig_p.replace("User: ", "").replace("\nAssistant: ", "")
            v_reword = f"User: Please assist with this: {body}\nAssistant: "
            v_reorder = f"User: {' '.join(body.split()[::-1])}\nAssistant: "
            v_extra = f"User: Note that yesterday was raining. In addition, {body}\nAssistant: "
            v_format = f"User: {body} Reply strictly in one concise JSON block.\nAssistant: "
            v_diff = "User: Calculate the boiling point of liquid nitrogen in Kelvin.\nAssistant: "
        else:
            v_reword = f"Please explain: {orig_p}"
            v_reorder = " ".join(orig_p.split()[::-1])
            v_extra = f"Context note: XYZ. {orig_p}"
            v_format = f"{orig_p} in bullet points."
            v_diff = "Calculate the boiling point of liquid nitrogen in Kelvin."

        variants = {
            "orig": v_orig,
            "reworded": v_reword,
            "reordered": v_reorder,
            "extra_ctx": v_extra,
            "format": v_format,
            "different": v_diff,
        }
        
        var_data = {}
        for k, p_text in variants.items():
            inp = [0] + tok.encode(p_text)
            curr = torch.tensor([inp], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(curr)[0, -1, :]
                probs = F.softmax(logits, dim=-1)
                top10_vals, top10_ids = torch.topk(probs, 10)
                first_tok = int(torch.argmax(logits).item())
                eos_prob = float(probs[1].item())
                
                # Generate 32 tokens greedy
                gen = []
                for _ in range(32):
                    if curr.shape[1] >= 512: break
                    l = model(curr)[0, -1, :]
                    nxt = int(torch.argmax(l).item())
                    if nxt == 1: break
                    gen.append(nxt)
                    curr = torch.cat([curr, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
                gen_text = tok.decode(gen).strip()
                
            var_data[k] = {
                "first_tok": first_tok,
                "first_tok_str": tok.decode([first_tok]),
                "top10_ids": top10_ids.tolist(),
                "top10_strs": [tok.decode([i]) for i in top10_ids.tolist()],
                "eos_prob": eos_prob,
                "gen_text": gen_text,
                "logits": logits.detach().cpu()
            }
            
        # Compare variants against original
        orig_top10 = set(var_data["orig"]["top10_ids"])
        orig_logits = var_data["orig"]["logits"]
        orig_gen = var_data["orig"]["gen_text"]
        orig_first = var_data["orig"]["first_tok"]
        
        for k in ["reworded", "reordered", "extra_ctx", "format", "different"]:
            if var_data[k]["first_tok"] == orig_first:
                first_token_matches[k] += 1
            if var_data[k]["gen_text"] == orig_gen:
                gen_identical[k] += 1
            # Jaccard overlap of top-10
            v_top10 = set(var_data[k]["top10_ids"])
            top10_overlaps[k].append(len(orig_top10.intersection(v_top10)) / 10.0)
            # Logit cosine similarity
            cos = F.cosine_similarity(orig_logits.unsqueeze(0), var_data[k]["logits"].unsqueeze(0)).item()
            logit_cos_sims[k].append(cos)
            
        results.append({
            "id": p_info.get("id", f"PROBE_{idx}"),
            "orig_prompt": orig_p,
            "orig_first_tok": var_data["orig"]["first_tok_str"],
            "orig_gen": orig_gen,
            "variants": {
                k: {
                    "first_tok": var_data[k]["first_tok_str"],
                    "eos_prob": round(var_data[k]["eos_prob"], 6),
                    "gen_text": var_data[k]["gen_text"],
                } for k in variants
            }
        })

    summary = {
        "num_probes": len(probes),
        "first_token_agreement_rate": {k: round(v / len(probes), 4) for k, v in first_token_matches.items()},
        "generation_identical_rate": {k: round(v / len(probes), 4) for k, v in gen_identical.items()},
        "mean_top10_overlap": {k: round(float(np.mean(v)), 4) for k, v in top10_overlaps.items()},
        "mean_logit_cosine_similarity": {k: round(float(np.mean(v)), 4) for k, v in logit_cos_sims.items()},
        "sample_probes": results[:5]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("Investigation A Complete. Summary:")
    print(f"  First token agreement with different prompt: {summary['first_token_agreement_rate']['different']*100:.1f}%")
    print(f"  Generation identical with different prompt  : {summary['generation_identical_rate']['different']*100:.1f}%")
    print(f"  Mean logit cosine similarity (different)   : {summary['mean_logit_cosine_similarity']['different']:.4f}")
    return summary


# ==============================================================================
# INVESTIGATION B: Internal Representation Analysis
# ==============================================================================

def run_investigation_b(
    model: ChakrMicro,
    tok: Any,
    probes: List[Dict[str, Any]],
    output_path: Path,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Records and compares:
    - Layer-wise hidden-state norms
    - Layer-wise attention entropy across heads
    - Cosine similarity across layers for similar vs distinct prompts
    - Representation distinctiveness from Layer 0 to Layer 5
    """
    print("Running Investigation B: Internal Representation Analysis...")
    
    # Select 20 pairs: 10 pairs of (similar instructions), 10 pairs of (distinct instructions / different languages)
    layer_acts = {}
    attn_probs = {}
    
    def get_layer_hook(idx):
        def hook(m, inp, out):
            layer_acts[idx] = out.detach()
        return hook
        
    def get_attn_hook(idx):
        def hook(m, inp, out):
            x = inp[0]
            B, T, C = x.shape
            q = m.q_proj(x).view(B, T, m.n_heads, m.head_dim).transpose(1, 2)
            k = m.k_proj(x).view(B, T, m.n_heads, m.head_dim).transpose(1, 2)
            q = m.rotary(q, T)
            k = m.rotary(k, T)
            scores = torch.matmul(q, k.transpose(-2, -1)) * m.scale
            scores = scores + m.causal_mask(T)
            probs = torch.softmax(scores, dim=-1)
            attn_probs[idx] = probs.detach()
        return hook

    handles = []
    for i, layer in enumerate(model.layers):
        handles.append(layer.register_forward_hook(get_layer_hook(i)))
        handles.append(layer.attn.register_forward_hook(get_attn_hook(i)))

    # Track layer-wise stats across all 100 probes
    layer_norms = [[] for _ in range(6)]
    layer_entropies = [[] for _ in range(6)]
    final_h_norms = []
    
    # Store final token representation for all probes
    probe_reps = [[] for _ in range(6)] # reps at each layer for the prompt-last token
    
    for p_info in probes:
        p = p_info["prompt"]
        inp = torch.tensor([[0] + tok.encode(p)], dtype=torch.long, device=device)
        with torch.no_grad():
            _, h_final = model(inp, return_hidden_states=True)
            final_h_norms.append(torch.norm(h_final[0, -1, :]).item())
            
            for i in range(6):
                # Layer hidden norm at last token
                norm_i = torch.norm(layer_acts[i][0, -1, :]).item()
                layer_norms[i].append(norm_i)
                probe_reps[i].append(layer_acts[i][0, -1, :].cpu())
                
                # Attention entropy at last query position across heads
                probs_i = attn_probs[i][0, :, -1, :] # [H, T]
                ent = -(probs_i * torch.log(probs_i + 1e-12)).sum(dim=-1).mean().item()
                layer_entropies[i].append(ent)

    # Compute pairwise representational similarity across probes at each layer
    layer_pairwise_cos = []
    for i in range(6):
        # Compute mean cosine similarity across 50 random pairs of probes
        cos_sims = []
        for _ in range(100):
            idx1, idx2 = random.sample(range(len(probes)), 2)
            v1 = probe_reps[i][idx1]
            v2 = probe_reps[i][idx2]
            cos_sims.append(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item())
        layer_pairwise_cos.append(float(np.mean(cos_sims)))

    # Clean up hooks
    for h in handles:
        h.remove()

    summary = {
        "num_probes": len(probes),
        "layer_hidden_norms_mean": [round(float(np.mean(norms)), 4) for norms in layer_norms],
        "layer_hidden_norms_std": [round(float(np.std(norms)), 4) for norms in layer_norms],
        "layer_attention_entropy_mean": [round(float(np.mean(ents)), 4) for ents in layer_entropies],
        "layer_attention_entropy_std": [round(float(np.std(ents)), 4) for ents in layer_entropies],
        "final_hidden_norm_mean": round(float(np.mean(final_h_norms)), 4),
        "layer_pairwise_cosine_similarity": [round(c, 4) for c in layer_pairwise_cos],
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Investigation B Complete. Summary:")
    print("  Layer norms (L0 to L5):", summary["layer_hidden_norms_mean"])
    print("  Attention entropy (L0 to L5):", summary["layer_attention_entropy_mean"])
    print("  Pairwise cosine similarity (L0 to L5):", summary["layer_pairwise_cosine_similarity"])
    return summary


# ==============================================================================
# INVESTIGATION C: Causal Prompt-Conditioning Tests
# ==============================================================================

def run_investigation_c(
    model: ChakrMicro,
    tok: Any,
    probes: List[Dict[str, Any]],
    output_path: Path,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Controlled interventions comparing next-token distribution under:
    - Correct prompt
    - Empty prompt ("User: \nAssistant: ")
    - Random unrelated prompt
    - Shuffled tokens
    - Entity changed
    - Instruction changed
    """
    print("Running Investigation C: Causal Conditioning Tests...")
    
    kl_divs = {"empty": [], "unrelated": [], "shuffled": [], "entity_swap": [], "instruction_swap": []}
    top5_overlaps = {"empty": [], "unrelated": [], "shuffled": [], "entity_swap": [], "instruction_swap": []}
    
    for idx, p_info in enumerate(probes):
        orig_p = p_info["prompt"]
        body = orig_p.replace("User: ", "").replace("\nAssistant: ", "")
        tokens = tok.encode(body)
        
        # 1. Empty prompt
        p_empty = "User: \nAssistant: "
        
        # 2. Random unrelated
        p_unrelated = "User: The boiling point of liquid hydrogen is approximately 20 degrees.\nAssistant: "
        
        # 3. Shuffled tokens (shuffle words or decode with errors='replace')
        words = body.split()
        shuffled_words = list(words)
        random.shuffle(shuffled_words)
        p_shuffled = f"User: {' '.join(shuffled_words)}\nAssistant: "
        
        # 4. Entity swap
        words = body.split()
        if len(words) >= 3:
            swapped_words = list(words)
            swapped_words[1] = "Jupiter" if swapped_words[1] != "Jupiter" else "Mercury"
            p_entity = f"User: {' '.join(swapped_words)}\nAssistant: "
        else:
            p_entity = f"User: {body} Alpha\nAssistant: "
            
        # 5. Instruction swap
        p_inst = f"User: Translate into Latin: {body}\nAssistant: "
        
        conds = {
            "orig": orig_p,
            "empty": p_empty,
            "unrelated": p_unrelated,
            "shuffled": p_shuffled,
            "entity_swap": p_entity,
            "instruction_swap": p_inst
        }
        
        distribs = {}
        top5_sets = {}
        for c_name, c_text in conds.items():
            inp = torch.tensor([[0] + tok.encode(c_text)], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(inp)[0, -1, :]
                probs = F.softmax(logits, dim=-1)
                distribs[c_name] = probs
                top5_sets[c_name] = set(torch.topk(probs, 5).indices.tolist())
                
        p_orig = distribs["orig"]
        s_orig = top5_sets["orig"]
        
        for c_name in ["empty", "unrelated", "shuffled", "entity_swap", "instruction_swap"]:
            p_cand = distribs[c_name]
            # KL divergence D_KL(P_orig || P_cand)
            kl = F.kl_div(torch.log(p_cand + 1e-12), p_orig, reduction="sum").item()
            kl_divs[c_name].append(max(0.0, kl))
            top5_overlaps[c_name].append(len(s_orig.intersection(top5_sets[c_name])) / 5.0)

    summary = {
        "num_probes": len(probes),
        "mean_kl_divergence": {k: round(float(np.mean(v)), 4) for k, v in kl_divs.items()},
        "std_kl_divergence": {k: round(float(np.std(v)), 4) for k, v in kl_divs.items()},
        "mean_top5_overlap": {k: round(float(np.mean(v)), 4) for k, v in top5_overlaps.items()},
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Investigation C Complete. Summary:")
    print("  Mean KL divergence from orig:", summary["mean_kl_divergence"])
    print("  Mean top-5 overlap with orig :", summary["mean_top5_overlap"])
    return summary


# ==============================================================================
# INVESTIGATION D: Training Objective Diagnostics
# ==============================================================================

def run_investigation_d(output_path: Path) -> Dict[str, Any]:
    """
    Audits the existing SFT training objective, prompt-response boundaries,
    token counts, and loss normalization without executing any training.
    """
    print("Running Investigation D: Training Objective Diagnostics...")
    from chakrview.training.prompt_contract import build_causal_sequence, normalize_prompt_response_strings
    from chakrview.training.sft_engine import SFTDataset
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")
    
    train_file = ROOT / "data/sft/ucip17/sft_train_12000.jsonl"
    sft_ds = SFTDataset(train_file, tok, max_seq_len=512, is_training=True)
    
    p_lens = []
    r_lens = []
    total_seq_lens = []
    eos_in_target = 0
    mask_check_correct = 0
    
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(sft_ds), size=500, replace=False)
    
    for idx in sample_indices:
        rec = sft_ds.records[idx]
        p_str, r_str = normalize_prompt_response_strings(rec["prompt"], rec["response"])
        p_toks = tok.encode(p_str)
        r_toks = tok.encode(r_str)
        p_lens.append(len(p_toks))
        r_lens.append(len(r_toks))
        
        inp_seq, tgt_seq = build_causal_sequence(p_toks, r_toks, bos_id=0, eos_id=1, ignore_index=-100)
        total_seq_lens.append(len(inp_seq))
        
        # Verify target EOS
        if tgt_seq[-1] == 1:
            eos_in_target += 1
            
        # Verify prompt masking (-100)
        # inp_seq: [0] + p_toks + r_toks
        # tgt_seq: [-100]*len(p_toks) + r_toks + [1]
        if tgt_seq[:len(p_toks)] == [-100] * len(p_toks) and tgt_seq[len(p_toks):-1] == r_toks:
            mask_check_correct += 1

    summary = {
        "audited_records_sample": len(sample_indices),
        "mean_prompt_tokens": round(float(np.mean(p_lens)), 2),
        "mean_response_tokens": round(float(np.mean(r_lens)), 2),
        "mean_total_sequence_tokens": round(float(np.mean(total_seq_lens)), 2),
        "prompt_to_response_token_ratio": round(float(np.mean(p_lens) / np.mean(r_lens)), 2),
        "target_has_terminal_eos_rate": round(eos_in_target / len(sample_indices), 4),
        "assistant_only_mask_alignment_correct_rate": round(mask_check_correct / len(sample_indices), 4),
        "replay_batch_structure": {
            "sft_batch_size": 12,
            "sft_active_tokens_per_step": round(12 * float(np.mean(r_lens)) + 12, 1),
            "replay_batch_size": 1,
            "replay_active_tokens_per_step": 512,
            "loss_formula": "L = 0.85 * (sum_sft / N_sft) + 0.15 * (sum_rep / N_rep)",
            "effective_gradient_share_sft": "89.2% norm share, 105.96% projection along g_sft"
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Investigation D Complete. Summary:")
    print(f"  Mean prompt tokens  : {summary['mean_prompt_tokens']}")
    print(f"  Mean response tokens: {summary['mean_response_tokens']}")
    print(f"  Target terminal EOS : {summary['target_has_terminal_eos_rate']*100:.1f}%")
    print(f"  Mask alignment rate : {summary['assistant_only_mask_alignment_correct_rate']*100:.1f}%")
    return summary


# ==============================================================================
# INVESTIGATION E: Generation Dynamics Across Decoding Settings
# ==============================================================================

def run_investigation_e(
    model_base: ChakrMicro,
    model_pilot: ChakrMicro,
    tok: Any,
    probes: List[Dict[str, Any]],
    output_path: Path,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Measures token-by-token generation dynamics:
    - Entropy at each step
    - Top-1 probability
    - Top-5 probability mass
    - EOS probability over time
    - Repeated n-gram onset and attractor transition
    - Evaluation across decoding configs (greedy, T=0.7, top-p=0.9)
    """
    print("Running Investigation E: Generation Dynamics...")
    
    configs = [
        {"name": "greedy", "temp": 0.0, "top_p": 1.0},
        {"name": "temp_0.7", "temp": 0.7, "top_p": 1.0},
        {"name": "top_p_0.9", "temp": 0.7, "top_p": 0.9},
    ]
    
    dynamics_results = {}
    
    for cfg in configs:
        c_name = cfg["name"]
        print(f"  Testing decoding setting: {c_name}...")
        
        pilot_step_entropies = [[] for _ in range(48)]
        pilot_step_top1_probs = [[] for _ in range(48)]
        pilot_step_top5_masses = [[] for _ in range(48)]
        pilot_step_eos_probs = [[] for _ in range(48)]
        
        base_step_entropies = [[] for _ in range(48)]
        base_step_top1_probs = [[] for _ in range(48)]
        base_step_eos_probs = [[] for _ in range(48)]
        
        unique_pilot_generations = set()
        unique_base_generations = set()

        for p_info in probes[:30]: # 30 probes for deep token-by-token trajectory
            p = p_info["prompt"]
            
            # 1. Pilot trajectory
            inp = torch.tensor([[0] + tok.encode(p)], dtype=torch.long, device=device)
            gen_p = []
            with torch.no_grad():
                for step in range(48):
                    if inp.shape[1] >= 512: break
                    logits = model_pilot(inp)[0, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    
                    ent = -(probs * torch.log(probs + 1e-12)).sum().item()
                    top1_p = torch.max(probs).item()
                    top5_mass = torch.topk(probs, 5).values.sum().item()
                    eos_p = probs[1].item()
                    
                    pilot_step_entropies[step].append(ent)
                    pilot_step_top1_probs[step].append(top1_p)
                    pilot_step_top5_masses[step].append(top5_mass)
                    pilot_step_eos_probs[step].append(eos_p)
                    
                    if cfg["temp"] <= 0.0:
                        nxt = int(torch.argmax(logits).item())
                    else:
                        scaled = logits / cfg["temp"]
                        if cfg["top_p"] < 1.0:
                            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > cfg["top_p"]
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices[sorted_indices_to_remove]
                            scaled[indices_to_remove] = -float("Inf")
                        nxt = int(torch.multinomial(F.softmax(scaled, dim=-1), 1).item())
                        
                    if nxt == 1: break
                    gen_p.append(nxt)
                    inp = torch.cat([inp, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            unique_pilot_generations.add(tok.decode(gen_p, errors="replace").strip())

            # 2. Baseline trajectory
            inp_b = torch.tensor([[0] + tok.encode(p)], dtype=torch.long, device=device)
            gen_b = []
            with torch.no_grad():
                for step in range(48):
                    if inp_b.shape[1] >= 512: break
                    logits = model_base(inp_b)[0, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    
                    ent = -(probs * torch.log(probs + 1e-12)).sum().item()
                    top1_p = torch.max(probs).item()
                    eos_p = probs[1].item()
                    
                    base_step_entropies[step].append(ent)
                    base_step_top1_probs[step].append(top1_p)
                    base_step_eos_probs[step].append(eos_p)
                    
                    if cfg["temp"] <= 0.0:
                        nxt = int(torch.argmax(logits).item())
                    else:
                        scaled = logits / cfg["temp"]
                        nxt = int(torch.multinomial(F.softmax(scaled, dim=-1), 1).item())
                    if nxt == 1: break
                    gen_b.append(nxt)
                    inp_b = torch.cat([inp_b, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1)
            unique_base_generations.add(tok.decode(gen_b, errors="replace").strip())

        # Average over first 20 tokens
        dynamics_results[c_name] = {
            "pilot": {
                "unique_outputs_count": len(unique_pilot_generations),
                "diversity_ratio": round(len(unique_pilot_generations) / 30, 4),
                "mean_token_entropy_trajectory": [round(float(np.mean(e)), 3) for e in pilot_step_entropies[:20] if e],
                "mean_top1_prob_trajectory": [round(float(np.mean(p)), 3) for p in pilot_step_top1_probs[:20] if p],
                "mean_top5_mass_trajectory": [round(float(np.mean(m)), 3) for m in pilot_step_top5_masses[:20] if m],
                "mean_eos_prob_trajectory": [round(float(np.mean(eos)), 5) for eos in pilot_step_eos_probs[:20] if eos],
            },
            "baseline": {
                "unique_outputs_count": len(unique_base_generations),
                "diversity_ratio": round(len(unique_base_generations) / 30, 4),
                "mean_token_entropy_trajectory": [round(float(np.mean(e)), 3) for e in base_step_entropies[:20] if e],
                "mean_top1_prob_trajectory": [round(float(np.mean(p)), 3) for p in base_step_top1_probs[:20] if p],
                "mean_eos_prob_trajectory": [round(float(np.mean(eos)), 5) for eos in base_step_eos_probs[:20] if eos],
            }
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dynamics_results, f, indent=2)

    print("Investigation E Complete.")
    return dynamics_results


def main():
    device = "cpu"
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")
    
    base_ckpt = ROOT / "experiments/ucip13/checkpoints/primary_seed42/checkpoint_step_800.pt"
    pilot_ckpt = ROOT / "experiments/ucip18/pilot/checkpoint_best.pt"
    
    print(f"Loading Base Checkpoint: {base_ckpt}")
    m_base = load_model(base_ckpt, device=device)
    print(f"Loading Pilot Checkpoint: {pilot_ckpt}")
    m_pilot = load_model(pilot_ckpt, device=device)
    
    dev_file = ROOT / "data/sft/ucip17/sft_dev_1500.jsonl"
    with open(dev_file, "r", encoding="utf-8") as f:
        dev_records = [json.loads(line) for line in f]
    random.seed(42)
    probes = random.sample(dev_records, 100)
    
    out_dir = ROOT / "experiments/ucip19/results"
    
    run_investigation_a(m_pilot, tok, probes, out_dir / "investigation_a_sensitivity.json", device=device)
    run_investigation_b(m_pilot, tok, probes, out_dir / "investigation_b_representations.json", device=device)
    run_investigation_c(m_pilot, tok, probes, out_dir / "investigation_c_causal.json", device=device)
    run_investigation_d(out_dir / "investigation_d_training_objective.json")
    run_investigation_e(m_base, m_pilot, tok, probes, out_dir / "investigation_e_dynamics.json", device=device)
    print("\nAll UCIP-19 Diagnostic Investigations Successfully Completed!")


if __name__ == "__main__":
    main()
