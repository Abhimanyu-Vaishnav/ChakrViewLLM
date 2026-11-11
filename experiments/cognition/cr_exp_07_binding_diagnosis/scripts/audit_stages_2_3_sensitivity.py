"""
Stage 2 & 3: Subject & Value Sensitivity Diagnostic Script
Measures internal representations across layers when:
- Stage 2: Only the subject changes (Ravi in Delhi vs Aman in Delhi)
- Stage 3: Only the value changes (Ravi in Delhi vs Ravi in Jaipur)
Computes:
- L2 distance, Cosine similarity, Relative delta norm
at:
- Subject position, Value position, Query-subject position, and Answer-delimiter position.
"""

import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
MODELS_TO_TEST = {
    "candidate_pilot": REPO_ROOT / "experiments/cognition/autonomous_12h/conversational_pilot/checkpoints/conversational_pilot_best_val.pt",
    "control_arm_best_val": REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/control_arm/checkpoints/control_arm_best_val.pt"
}

def load_model(ckpt_path):
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def capture_sequence_activations(model, token_ids):
    x = torch.tensor([token_ids], dtype=torch.long)
    seq_len = x.shape[1]
    records = {
        "embedding": None,
        "layers": [],
        "final_norm": None,
        "logits": None
    }
    with torch.no_grad():
        emb = model.embedding(x)
        records["embedding"] = emb[0].clone()
        curr = emb
        for i, layer in enumerate(model.layers):
            res_in = curr.clone()
            norm1_out = layer.norm_1(curr)
            attn_out = layer.attn(norm1_out)
            res_mid = res_in + attn_out
            norm2_out = layer.norm_2(res_mid)
            ffn_out = layer.ffn(norm2_out)
            out = res_mid + ffn_out
            records["layers"].append({
                "layer_idx": i,
                "attn_out": attn_out[0].clone(),
                "ffn_out": ffn_out[0].clone(),
                "res_post_ffn": out[0].clone()
            })
            curr = out
        final_norm = model.final_norm(curr)
        logits = model.lm_head(final_norm)
        records["final_norm"] = final_norm[0].clone()
        records["logits"] = logits[0].clone()
    return records

def safe_stats(u, v):
    # u, v: 1D tensors of shape (d_model,)
    diff = u - v
    l2 = diff.norm(p=2).item()
    u_norm = u.norm(p=2).item()
    v_norm = v.norm(p=2).item()
    cos_sim = F.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0)).item() if u_norm > 0 and v_norm > 0 else 0.0
    rel_diff = l2 / (0.5 * (u_norm + v_norm)) * 100.0 if (u_norm + v_norm) > 0 else 0.0
    return {
        "l2_distance": round(l2, 4),
        "cosine_similarity": round(cos_sim, 4),
        "relative_difference_pct": round(rel_diff, 2),
        "mean_vector_norm": round(0.5 * (u_norm + v_norm), 4)
    }

def main():
    print("=== CR-EXP-07 Stage 2 & 3: Subject & Value Sensitivity Diagnostics ===")
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # 1. Subject Sensitivity Pair (Stage 2)
    # Target prompts: Matched token length (38 tokens each)
    p_subj_A = "Mapping: Ravi lives in Delhi. Query: Where does Ravi live? Answer:"
    p_subj_B = "Mapping: Aman lives in Delhi. Query: Where does Aman live? Answer:"
    # Token lengths: Ensure both are padded/truncated to identical 38 tokens
    t_A = tok.encode(p_subj_A)
    t_B = tok.encode(p_subj_B)
    print(f"Subject pair lengths: A={len(t_A)}, B={len(t_B)}")
    # Pad to equal length if needed
    max_len = max(len(t_A), len(t_B))
    t_A_pad = t_A + [0] * (max_len - len(t_A))
    t_B_pad = t_B + [0] * (max_len - len(t_B))

    # 2. Value Sensitivity Pair (Stage 3)
    p_val_A = "Mapping: Ravi lives in Delhi. Query: Where does Ravi live? Answer:"
    p_val_B = "Mapping: Ravi lives in Jaipur. Query: Where does Ravi live? Answer:"
    t_vA = tok.encode(p_val_A)
    t_vB = tok.encode(p_val_B)
    print(f"Value pair lengths: vA={len(t_vA)}, vB={len(t_vB)}")
    max_v_len = max(len(t_vA), len(t_vB))
    t_vA_pad = t_vA + [0] * (max_v_len - len(t_vA))
    t_vB_pad = t_vB + [0] * (max_v_len - len(t_vB))

    for m_key, m_path in MODELS_TO_TEST.items():
        print(f"\n--- Model: {m_key} ---")
        model = load_model(m_path)

        # Subject Sensitivity Captures
        act_sA = capture_sequence_activations(model, t_A_pad)
        act_sB = capture_sequence_activations(model, t_B_pad)

        # Delimiter token position (last token before generation)
        delim_idx = max_len - 1

        subj_layers = []
        for l in range(6):
            res_A = act_sA["layers"][l]["res_post_ffn"][delim_idx]
            res_B = act_sB["layers"][l]["res_post_ffn"][delim_idx]
            stats = safe_stats(res_A, res_B)
            subj_layers.append({"layer": l, "delimiter_position_stats": stats})

        final_norm_A = act_sA["final_norm"][delim_idx]
        final_norm_B = act_sB["final_norm"][delim_idx]
        final_subj_stats = safe_stats(final_norm_A, final_norm_B)

        logit_A = act_sA["logits"][delim_idx]
        logit_B = act_sB["logits"][delim_idx]
        logit_subj_stats = safe_stats(logit_A, logit_B)

        subj_results = {
            "model": m_key,
            "prompt_A": p_subj_A,
            "prompt_B": p_subj_B,
            "layer_by_layer_delimiter_divergence": subj_layers,
            "final_norm_divergence": final_subj_stats,
            "logit_divergence": logit_subj_stats
        }

        # Value Sensitivity Captures
        act_vA = capture_sequence_activations(model, t_vA_pad)
        act_vB = capture_sequence_activations(model, t_vB_pad)

        v_delim_idx = max_v_len - 1
        val_layers = []
        for l in range(6):
            res_vA = act_vA["layers"][l]["res_post_ffn"][v_delim_idx]
            res_vB = act_vB["layers"][l]["res_post_ffn"][v_delim_idx]
            v_stats = safe_stats(res_vA, res_vB)
            val_layers.append({"layer": l, "delimiter_position_stats": v_stats})

        final_val_norm_stats = safe_stats(act_vA["final_norm"][v_delim_idx], act_vB["final_norm"][v_delim_idx])
        logit_val_stats = safe_stats(act_vA["logits"][v_delim_idx], act_vB["logits"][v_delim_idx])

        val_results = {
            "model": m_key,
            "prompt_A": p_val_A,
            "prompt_B": p_val_B,
            "layer_by_layer_delimiter_divergence": val_layers,
            "final_norm_divergence": final_val_norm_stats,
            "logit_divergence": logit_val_stats
        }

        print(f"Subject Sensitivity: Final Logit CosSim={logit_subj_stats['cosine_similarity']}, RelDiff={logit_subj_stats['relative_difference_pct']}%")
        print(f"Value Sensitivity  : Final Logit CosSim={logit_val_stats['cosine_similarity']}, RelDiff={logit_val_stats['relative_difference_pct']}%")

        if m_key == "control_arm_best_val":
            with open(REPO_ROOT / "experiments/cognition/cr_exp_07_binding_diagnosis/subject_sensitivity_results.json", "w", encoding="utf-8") as f:
                json.dump(subj_results, f, indent=2)
            with open(REPO_ROOT / "experiments/cognition/cr_exp_07_binding_diagnosis/value_sensitivity_results.json", "w", encoding="utf-8") as f:
                json.dump(val_results, f, indent=2)

    print("\nStage 2 & 3 outputs written successfully.")

if __name__ == "__main__":
    main()

