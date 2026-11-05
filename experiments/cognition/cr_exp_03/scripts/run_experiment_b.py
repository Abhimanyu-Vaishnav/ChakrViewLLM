"""
CR-EXP-03 Experiment B: FFN Causal Ablation
Tests zeroing FFN outputs at inference time:
1. No intervention
2. FFN zeroed at Layer 5
3. FFN zeroed at Layer 4
4. FFN zeroed at Layers 4 and 5
5. FFN zeroed across all six layers (0..5)

Measured for Parent and Candidate on Prompts A & B:
- Top-1 token & text
- Top-5 tokens & texts
- Top-1 margin (z_top1 - z_top2)
- Entropy H(p)
- Vocabulary logit L2 difference ||z_A - z_B||_2
- Relative logit divergence ||z_A - z_B|| / ||z_A||
- Target token probabilities for ' Rhodium', ' Platinum'
- Residual stream norm at final norm
"""

import sys
import json
import math
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

CONDITIONS = [
    ("none", []),
    ("layer_5", [5]),
    ("layer_4", [4]),
    ("layer_4_5", [4, 5]),
    ("all_layers", [0, 1, 2, 3, 4, 5])
]

def load_model(ckpt_path):
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, max_seq_len=512, tie_embeddings=True)
    model = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def forward_with_ffn_ablation(model, input_ids, zero_ffn_layers):
    """
    Forward pass applying temporary zeroing of FFN output before residual addition
    at the specified layers. Does NOT mutate model weights.
    """
    with torch.no_grad():
        x = model.embedding(input_ids)
        for i, layer in enumerate(model.layers):
            res_in = x
            norm1_out = layer.norm_1(x)
            attn_out = layer.attn(norm1_out)
            res_mid = res_in + attn_out
            norm2_out = layer.norm_2(res_mid)
            ffn_out = layer.ffn(norm2_out)
            if i in zero_ffn_layers:
                ffn_out = torch.zeros_like(ffn_out)
            x = res_mid + ffn_out
        final_norm = model.final_norm(x)
        logits = model.lm_head(final_norm)
    return logits[0, -1], final_norm[0, -1]

def compute_metrics(logits, tok):
    probs = F.softmax(logits, dim=-1)
    top5_vals, top5_ids = torch.topk(logits, k=5)
    
    top1_id = int(top5_ids[0].item())
    top1_logit = float(top5_vals[0].item())
    top2_logit = float(top5_vals[1].item())
    margin = top1_logit - top2_logit
    
    # Entropy: - sum p * log(p)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = float(-torch.sum(probs * log_probs).item())
    
    top5_details = []
    for rank in range(5):
        tid = int(top5_ids[rank].item())
        tval = float(top5_vals[rank].item())
        prob = float(probs[tid].item())
        token_str = tok.decode([tid])
        top5_details.append({
            "rank": rank + 1,
            "token_id": tid,
            "token_str": token_str,
            "logit": round(tval, 4),
            "prob": round(prob, 6)
        })
        
    return {
        "top1_id": top1_id,
        "top1_str": tok.decode([top1_id]),
        "top1_logit": round(top1_logit, 4),
        "top1_margin": round(margin, 4),
        "entropy": round(entropy, 4),
        "top5": top5_details,
        "logits_raw": logits
    }

def main():
    print("=== CR-EXP-03 EXPERIMENT B: FFN CAUSAL ABLATION ===")
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    tokens_A = torch.tensor([tok.encode(PROMPTS["A"])], dtype=torch.long)
    tokens_B = torch.tensor([tok.encode(PROMPTS["B"])], dtype=torch.long)
    
    parent = load_model(PARENT_PATH)
    cand = load_model(CAND_PATH)
    
    models = [("parent", parent), ("candidate", cand)]
    all_results = {}
    
    for m_name, model in models:
        m_results = {}
        for cond_name, layers in CONDITIONS:
            logits_A, norm_A = forward_with_ffn_ablation(model, tokens_A, layers)
            logits_B, norm_B = forward_with_ffn_ablation(model, tokens_B, layers)
            
            met_A = compute_metrics(logits_A, tok)
            met_B = compute_metrics(logits_B, tok)
            
            l2_diff = float(torch.norm(logits_A - logits_B).item())
            norm_A_val = float(torch.norm(logits_A).item())
            rel_diff = round((l2_diff / norm_A_val) * 100, 2) if norm_A_val > 0 else 0.0
            
            # Remove raw logits from saved json
            del met_A["logits_raw"]
            del met_B["logits_raw"]
            
            m_results[cond_name] = {
                "ablated_layers": layers,
                "prompt_A": met_A,
                "prompt_B": met_B,
                "logit_l2_diff": round(l2_diff, 4),
                "relative_divergence_pct": rel_diff,
                "final_norm_A": round(float(torch.norm(norm_A).item()), 4),
                "final_norm_B": round(float(torch.norm(norm_B).item()), 4)
            }
            print(f"[{m_name.upper()} | {cond_name}] Top1(A): {met_A['top1_str']!r} ({met_A['top1_logit']:.2f}) | Top1(B): {met_B['top1_str']!r} ({met_B['top1_logit']:.2f}) | RelDiff: {rel_diff}% | H(A): {met_A['entropy']:.2f}")
            
        all_results[m_name] = m_results
        
    out_file = REPO_ROOT / "experiments" / "cognition" / "cr_exp_03" / "experiment_b_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved Experiment B results to {out_file}")

if __name__ == "__main__":
    main()

