"""
CR-EXP-03 Experiment A: Residual Vector Geometry
Calculates:
- ||Attn||, ||FFN||, ||Residual||
- Contextual deltas: dAttn, dFFN, dResidual
- Cosine similarities: cos(Attn, FFN), cos(dAttn, dFFN)
- Vector projections: proj(FFN onto dAttn), proj(FFN onto dResidual)
- Parallel and orthogonal decomposition
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

def load_model(ckpt_path):
    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, max_seq_len=512, tie_embeddings=True)
    model = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def capture_layer_tensors(model, input_ids):
    """
    Runs forward pass while capturing tensors at each layer:
    - input residual to layer
    - attention output
    - post-attention residual
    - FFN output
    - post-FFN residual
    """
    records = []
    with torch.no_grad():
        x = model.embedding(input_ids)
        for i, layer in enumerate(model.layers):
            res_in = x.clone()
            norm1_out = layer.norm_1(x)
            attn_out = layer.attn(norm1_out)
            res_post_attn = res_in + attn_out
            norm2_out = layer.norm_2(res_post_attn)
            ffn_out = layer.ffn(norm2_out)
            res_post_ffn = res_post_attn + ffn_out
            records.append({
                "layer": i,
                "res_in": res_in[0, -1].clone(),
                "attn_out": attn_out[0, -1].clone(),
                "res_post_attn": res_post_attn[0, -1].clone(),
                "ffn_out": ffn_out[0, -1].clone(),
                "res_post_ffn": res_post_ffn[0, -1].clone()
            })
            x = res_post_ffn
        final_norm = model.final_norm(x)
        logits = model.lm_head(final_norm)
    return records, final_norm[0, -1], logits[0, -1]

def safe_cos(u, v, eps=1e-8):
    nu = torch.norm(u)
    nv = torch.norm(v)
    if nu < eps or nv < eps:
        return 0.0
    return float(torch.dot(u, v) / (nu * nv))

def proj_b_onto_a(b, a, eps=1e-8):
    """Projection of vector b onto vector a: (b . a) / ||a||^2 * a"""
    na2 = torch.sum(a * a)
    if na2 < eps:
        return torch.zeros_like(b), 0.0
    scale = float(torch.dot(b, a) / na2)
    proj = scale * a
    return proj, scale

def analyze_geometry(rec_A, rec_B):
    results = []
    for l in range(len(rec_A)):
        rA = rec_A[l]
        rB = rec_B[l]
        
        attn_A = rA["attn_out"]
        attn_B = rB["attn_out"]
        ffn_A = rA["ffn_out"]
        ffn_B = rB["ffn_out"]
        res_A = rA["res_post_ffn"]
        res_B = rB["res_post_ffn"]
        
        d_attn = attn_A - attn_B
        d_ffn = ffn_A - ffn_B
        d_res = res_A - res_B
        
        # Norms
        norm_attn = float(torch.norm(attn_A))
        norm_ffn = float(torch.norm(ffn_A))
        norm_res = float(torch.norm(res_A))
        norm_d_attn = float(torch.norm(d_attn))
        norm_d_ffn = float(torch.norm(d_ffn))
        norm_d_res = float(torch.norm(d_res))
        
        # Cosine similarities
        cos_attn_ffn = safe_cos(attn_A, ffn_A)
        cos_dattn_dffn = safe_cos(d_attn, d_ffn)
        cos_dattn_dres = safe_cos(d_attn, d_res)
        cos_dffn_dres = safe_cos(d_ffn, d_res)
        
        # Projection of d_ffn onto d_attn
        proj_dffn_on_dattn, scale_dffn_on_dattn = proj_b_onto_a(d_ffn, d_attn)
        proj_dattn_on_dres, scale_dattn_on_dres = proj_b_onto_a(d_attn, d_res)
        proj_dffn_on_dres, scale_dffn_on_dres = proj_b_onto_a(d_ffn, d_res)
        
        # Orthogonal components
        orth_dffn_wrt_dattn = float(torch.norm(d_ffn - proj_dffn_on_dattn))
        
        # Residual delta build-up: d_res = d_res_in + d_attn + d_ffn
        d_res_in = rA["res_in"] - rB["res_in"]
        
        results.append({
            "layer": l,
            "norm_attn": norm_attn,
            "norm_ffn": norm_ffn,
            "norm_res": norm_res,
            "norm_d_attn": norm_d_attn,
            "norm_d_ffn": norm_d_ffn,
            "norm_d_res": norm_d_res,
            "cos_attn_ffn": cos_attn_ffn,
            "cos_dattn_dffn": cos_dattn_dffn,
            "cos_dattn_dres": cos_dattn_dres,
            "cos_dffn_dres": cos_dffn_dres,
            "scale_dffn_on_dattn": scale_dffn_on_dattn,
            "scale_dattn_on_dres": scale_dattn_on_dres,
            "scale_dffn_on_dres": scale_dffn_on_dres,
            "orth_dffn_wrt_dattn": orth_dffn_wrt_dattn,
            "norm_d_res_in": float(torch.norm(d_res_in))
        })
    return results

def main():
    print("=== CR-EXP-03 EXPERIMENT A: RESIDUAL VECTOR GEOMETRY ===")
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
    
    tokens = {k: torch.tensor([tok.encode(v)], dtype=torch.long) for k, v in PROMPTS.items()}
    
    parent = load_model(PARENT_PATH)
    cand = load_model(CAND_PATH)
    
    # Capture activations
    parent_A, _, parent_logits_A = capture_layer_tensors(parent, tokens["A"])
    parent_B, _, parent_logits_B = capture_layer_tensors(parent, tokens["B"])
    parent_C, _, _ = capture_layer_tensors(parent, tokens["C"])
    
    cand_A, _, cand_logits_A = capture_layer_tensors(cand, tokens["A"])
    cand_B, _, cand_logits_B = capture_layer_tensors(cand, tokens["B"])
    cand_C, _, _ = capture_layer_tensors(cand, tokens["C"])
    
    geom_parent_AB = analyze_geometry(parent_A, parent_B)
    geom_cand_AB = analyze_geometry(cand_A, cand_B)
    
    geom_parent_AC = analyze_geometry(parent_A, parent_C)
    geom_cand_AC = analyze_geometry(cand_A, cand_C)
    
    output = {
        "metadata": {
            "prompt_A": PROMPTS["A"],
            "prompt_B": PROMPTS["B"],
            "prompt_C": PROMPTS["C"],
            "delimiter_pos": 41
        },
        "parent_vs_candidate_AB": {
            "parent": geom_parent_AB,
            "candidate": geom_cand_AB
        },
        "parent_vs_candidate_AC": {
            "parent": geom_parent_AC,
            "candidate": geom_cand_AC
        }
    }
    
    out_file = REPO_ROOT / "experiments" / "cognition" / "cr_exp_03" / "experiment_a_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved Experiment A results to {out_file}")

if __name__ == "__main__":
    main()

