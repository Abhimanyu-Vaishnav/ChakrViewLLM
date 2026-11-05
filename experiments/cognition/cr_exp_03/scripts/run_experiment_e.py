"""
CR-EXP-03 Experiment E: Causal Consistency and Robustness across Diverse Prompt Pairs
Tests:
- Pair 1: Rhodium (A) vs Platinum (B)
- Pair 2: Rhodium (A) vs Phoenix (C)
- Pair 3: Rhodium (A) vs Silver (D)  [Novel length-matched control pair]

Measures:
- Unperturbed relative divergence
- FFN ablation relative divergence (Layer 5, Layer 4+5, All)
- Attention ablation relative divergence (Layer 5, Layer 4+5, All)
- Counterfactual replacement shift (FFN L5 swap, Attn L5 swap)
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
    "B": "Mapping: Quasar is mapped to Platinum. Query: What does Quasar map to?\nAnswer:",
    "C": "Mapping: Quasar is mapped to Phoenix. Query: What does Quasar map to?\nAnswer:",
    "D": "Mapping: Quasar is mapped to Silver. Query: What does Quasar map to?\nAnswer:"
}

PAIRS = [
    ("Rhodium_vs_Platinum", "A", "B"),
    ("Rhodium_vs_Phoenix", "A", "C"),
    ("Rhodium_vs_Silver", "A", "D")
]

def load_model(ckpt_path):
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, max_seq_len=512, tie_embeddings=True)
    model = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def forward_custom(model, input_ids, zero_ffn=[], zero_attn=[], ffn_replacement=None, attn_replacement=None, rep_layer=None):
    with torch.no_grad():
        x = model.embedding(input_ids)
        for i, layer in enumerate(model.layers):
            norm1_out = layer.norm_1(x)
            attn_out = layer.attn(norm1_out)
            if i in zero_attn:
                attn_out = torch.zeros_like(attn_out)
            if attn_replacement is not None and i == rep_layer:
                attn_out[:, -1:, :] = attn_replacement
            res_mid = x + attn_out
            norm2_out = layer.norm_2(res_mid)
            ffn_out = layer.ffn(norm2_out)
            if i in zero_ffn:
                ffn_out = torch.zeros_like(ffn_out)
            if ffn_replacement is not None and i == rep_layer:
                ffn_out[:, -1:, :] = ffn_replacement
            x = res_mid + ffn_out
        final_norm = model.final_norm(x)
        logits = model.lm_head(final_norm)
    return logits[0, -1]

def rel_diff(logits1, logits2):
    diff = float(torch.norm(logits1 - logits2).item())
    norm = float(torch.norm(logits1).item())
    return round((diff / norm) * 100, 2) if norm > 0 else 0.0

def main():
    print("=== CR-EXP-03 EXPERIMENT E: CAUSAL CONSISTENCY ACROSS PROMPT PAIRS ===")
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    tokens = {k: torch.tensor([tok.encode(v)], dtype=torch.long) for k, v in PROMPTS.items()}
    
    parent = load_model(PARENT_PATH)
    cand = load_model(CAND_PATH)
    
    results = {}
    for pair_name, k1, k2 in PAIRS:
        print(f"\n--- Testing Pair: {pair_name} ({k1} vs {k2}) ---")
        pair_data = {}
        for m_name, model in [("parent", parent), ("candidate", cand)]:
            # 1. Unmodified baseline
            l1_base = forward_custom(model, tokens[k1])
            l2_base = forward_custom(model, tokens[k2])
            div_base = rel_diff(l1_base, l2_base)
            
            # 2. FFN Ablation L5
            l1_ffn5 = forward_custom(model, tokens[k1], zero_ffn=[5])
            l2_ffn5 = forward_custom(model, tokens[k2], zero_ffn=[5])
            div_ffn5 = rel_diff(l1_ffn5, l2_ffn5)
            
            # 3. FFN Ablation L4+5
            l1_ffn45 = forward_custom(model, tokens[k1], zero_ffn=[4, 5])
            l2_ffn45 = forward_custom(model, tokens[k2], zero_ffn=[4, 5])
            div_ffn45 = rel_diff(l1_ffn45, l2_ffn45)
            
            # 4. Attention Ablation L5
            l1_att5 = forward_custom(model, tokens[k1], zero_attn=[5])
            l2_att5 = forward_custom(model, tokens[k2], zero_attn=[5])
            div_att5 = rel_diff(l1_att5, l2_att5)
            
            # 5. Attention Ablation L4+5
            l1_att45 = forward_custom(model, tokens[k1], zero_attn=[4, 5])
            l2_att45 = forward_custom(model, tokens[k2], zero_attn=[4, 5])
            div_att45 = rel_diff(l1_att45, l2_att45)
            
            # 6. Attention Ablation ALL
            l1_att_all = forward_custom(model, tokens[k1], zero_attn=[0, 1, 2, 3, 4, 5])
            l2_att_all = forward_custom(model, tokens[k2], zero_attn=[0, 1, 2, 3, 4, 5])
            div_att_all = rel_diff(l1_att_all, l2_att_all)
            
            pair_data[m_name] = {
                "baseline_relative_divergence_pct": div_base,
                "ffn_ablation_L5_pct": div_ffn5,
                "ffn_ablation_L4_5_pct": div_ffn45,
                "attn_ablation_L5_pct": div_att5,
                "attn_ablation_L4_5_pct": div_att45,
                "attn_ablation_ALL_pct": div_att_all
            }
            print(f"[{m_name.upper()}] BaseDiv: {div_base}% | FFN(L5): {div_ffn5}% | FFN(L4+5): {div_ffn45}% | Attn(L5): {div_att5}% | Attn(ALL): {div_att_all}%")
            
        results[pair_name] = pair_data
        
    out_file = REPO_ROOT / "experiments" / "cognition" / "cr_exp_03" / "experiment_e_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Experiment E results to {out_file}")

if __name__ == "__main__":
    main()

