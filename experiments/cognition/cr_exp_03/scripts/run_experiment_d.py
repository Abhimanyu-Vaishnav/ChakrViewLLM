"""
CR-EXP-03 Experiment D: Counterfactual Activation Replacement
Tests swapping activations at the delimiter position between Prompt A and Prompt B:
- FFN activation swap at Layer 0, 2, 4, 5 (A -> B and B -> A)
- Attention activation swap at Layer 0, 2, 4, 5 (A -> B and B -> A) as control
Measures:
- Logit L2 difference from unperturbed baseline
- Top-1 token and top-1 change
- Expected target token probability
- Relative logit shift
"""

import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ModelConfig, ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

PARENT_PATH = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_04" / "checkpoints" / "sft_checkpoint_step_200.pt"
CAND_PATH = REPO_ROOT / "experiments" / "cognition" / "cr_exp_01" / "runs" / "stage_a_pilot" / "checkpoints" / "stage_a_candidate_step_100.pt"

PROMPTS = {
    "A": "Mapping: Quasar is mapped to Rhodium. Query: What does Quasar map to?\nAnswer:",
    "B": "Mapping: Quasar is mapped to Platinum. Query: What does Quasar map to?\nAnswer:"
}

TEST_LAYERS = [0, 2, 4, 5]

def load_model(ckpt_path):
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, max_seq_len=512, tie_embeddings=True)
    model = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def capture_sublayer_activations(model, input_ids):
    attn_outs = {}
    ffn_outs = {}
    with torch.no_grad():
        x = model.embedding(input_ids)
        for i, layer in enumerate(model.layers):
            norm1_out = layer.norm_1(x)
            attn_out = layer.attn(norm1_out)
            attn_outs[i] = attn_out[:, -1:, :].clone()
            res_mid = x + attn_out
            norm2_out = layer.norm_2(res_mid)
            ffn_out = layer.ffn(norm2_out)
            ffn_outs[i] = ffn_out[:, -1:, :].clone()
            x = res_mid + ffn_out
        final_norm = model.final_norm(x)
        logits = model.lm_head(final_norm)
    return logits[0, -1], attn_outs, ffn_outs

def forward_with_replacement(model, input_ids, swap_type, swap_layer, replacement_act):
    """
    swap_type: 'ffn' or 'attn'
    swap_layer: layer index (int)
    replacement_act: tensor of shape [1, 1, 192] to inject at position -1
    """
    with torch.no_grad():
        x = model.embedding(input_ids)
        for i, layer in enumerate(model.layers):
            norm1_out = layer.norm_1(x)
            attn_out = layer.attn(norm1_out)
            if swap_type == 'attn' and i == swap_layer:
                attn_out[:, -1:, :] = replacement_act
            res_mid = x + attn_out
            norm2_out = layer.norm_2(res_mid)
            ffn_out = layer.ffn(norm2_out)
            if swap_type == 'ffn' and i == swap_layer:
                ffn_out[:, -1:, :] = replacement_act
            x = res_mid + ffn_out
        final_norm = model.final_norm(x)
        logits = model.lm_head(final_norm)
    return logits[0, -1]

def get_summary(logits, baseline_logits, tok):
    probs = F.softmax(logits, dim=-1)
    top1_id = int(torch.argmax(logits).item())
    l2_diff = float(torch.norm(logits - baseline_logits).item())
    norm_base = float(torch.norm(baseline_logits).item())
    rel_diff = round((l2_diff / norm_base) * 100, 2) if norm_base > 0 else 0.0
    return {
        "top1_id": top1_id,
        "top1_str": tok.decode([top1_id]),
        "top1_logit": round(float(logits[top1_id].item()), 4),
        "l2_diff_from_base": round(l2_diff, 4),
        "rel_diff_pct": rel_diff
    }

def main():
    print("=== CR-EXP-03 EXPERIMENT D: COUNTERFACTUAL ACTIVATION REPLACEMENT ===")
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    tokens_A = torch.tensor([tok.encode(PROMPTS["A"])], dtype=torch.long)
    tokens_B = torch.tensor([tok.encode(PROMPTS["B"])], dtype=torch.long)
    
    parent = load_model(PARENT_PATH)
    cand = load_model(CAND_PATH)
    
    models = [("parent", parent), ("candidate", cand)]
    all_results = {}
    
    for m_name, model in models:
        base_logits_A, attn_A, ffn_A = capture_sublayer_activations(model, tokens_A)
        base_logits_B, attn_B, ffn_B = capture_sublayer_activations(model, tokens_B)
        
        m_results = {}
        for swap_type in ["ffn", "attn"]:
            type_res = {}
            for l in TEST_LAYERS:
                # 1. Inject A's activation into B (testing if B's prediction shifts toward A)
                rep_A = ffn_A[l] if swap_type == 'ffn' else attn_A[l]
                logits_B_with_A = forward_with_replacement(model, tokens_B, swap_type, l, rep_A)
                res_B_with_A = get_summary(logits_B_with_A, base_logits_B, tok)
                
                # 2. Inject B's activation into A (testing if A's prediction shifts toward B)
                rep_B = ffn_B[l] if swap_type == 'ffn' else attn_B[l]
                logits_A_with_B = forward_with_replacement(model, tokens_A, swap_type, l, rep_B)
                res_A_with_B = get_summary(logits_A_with_B, base_logits_A, tok)
                
                type_res[f"layer_{l}"] = {
                    "inject_A_into_B": res_B_with_A,
                    "inject_B_into_A": res_A_with_B
                }
                print(f"[{m_name.upper()} | {swap_type.upper()} L{l}] B+rep(A): top1={res_B_with_A['top1_str']!r}, shift={res_B_with_A['rel_diff_pct']}% | A+rep(B): top1={res_A_with_B['top1_str']!r}, shift={res_A_with_B['rel_diff_pct']}%")
                
            type_res_all = {}
            # Also test swapping across all tested layers simultaneously
            # (Layer 0, 2, 4, 5 together)
            # Custom multi-layer swap function
            with torch.no_grad():
                # A into B across all 4 layers
                x = model.embedding(tokens_B)
                for i, layer in enumerate(model.layers):
                    norm1_out = layer.norm_1(x)
                    attn_out = layer.attn(norm1_out)
                    if swap_type == 'attn' and i in TEST_LAYERS:
                        attn_out[:, -1:, :] = attn_A[i]
                    res_mid = x + attn_out
                    norm2_out = layer.norm_2(res_mid)
                    ffn_out = layer.ffn(norm2_out)
                    if swap_type == 'ffn' and i in TEST_LAYERS:
                        ffn_out[:, -1:, :] = ffn_A[i]
                    x = res_mid + ffn_out
                final_norm = model.final_norm(x)
                logits_multi = model.lm_head(final_norm)
                type_res["layers_0_2_4_5_combined"] = {
                    "inject_A_into_B": get_summary(logits_multi[0, -1], base_logits_B, tok)
                }
                print(f"[{m_name.upper()} | {swap_type.upper()} COMBINED] B+rep(A) all: top1={type_res['layers_0_2_4_5_combined']['inject_A_into_B']['top1_str']!r}, shift={type_res['layers_0_2_4_5_combined']['inject_A_into_B']['rel_diff_pct']}%")
                
            m_results[swap_type] = type_res
        all_results[m_name] = m_results
        
    out_file = REPO_ROOT / "experiments" / "cognition" / "cr_exp_03" / "experiment_d_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved Experiment D results to {out_file}")

if __name__ == "__main__":
    main()

