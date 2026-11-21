"""
UCIP-26 Phase 6: In-Memory Mechanistic Diagnostics.
Instruments per-layer hidden states, attention matrices, and residual stream norms
during generation-boundary forward pass to diagnose where context information is lost:
1. Attention weight distribution from Assistant boundary token back to context tokens
2. Layer-wise residual stream magnitude (L2 norm)
3. Attention output norm vs MLP output norm per layer
4. Attention Ablation: Replace attention output at layer 5/6 with zeros, observe logit impact.
"""

from __future__ import annotations
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def run_mechanistic_diagnostics():
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")
    ckpt_path = Path("experiments/ucip20/checkpoints/checkpoint_best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = ChakrMicro(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    prompt = "User: Specification dossier for NovaTrack:\n1. founded in: 1981\n2. headquartered in: Hyderabad\n3. flagship product: Product-Nova-001\nQuestion: What is the founded in of NovaTrack?\nAnswer concisely:\nAssistant: "
    input_ids = [0] + tok.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long)
    seq_len = x.shape[1]

    # Find position of '1981' and 'Hyderabad' in prompt
    tok_1981 = tok.encode("1981")[0]
    pos_1981 = [i for i, t in enumerate(input_ids) if t == tok_1981]

    # Forward pass hook to capture hidden states and attention
    hidden_states = []
    layer_norms = []

    # Run through embedding
    h = model.embedding(x)
    hidden_states.append(("emb", float(torch.norm(h[0, -1, :]).item())))

    # Model layers
    for l_idx, layer in enumerate(model.layers):
        # Attention
        norm_h1 = layer.norm_1(h)
        attn_out = layer.attn(norm_h1, layer_idx=l_idx)
        h = h + attn_out
        attn_norm = float(torch.norm(attn_out[0, -1, :]).item())

        # MLP
        norm_h2 = layer.norm_2(h)
        mlp_out = layer.ffn(norm_h2)
        h = h + mlp_out
        mlp_norm = float(torch.norm(mlp_out[0, -1, :]).item())

        total_norm = float(torch.norm(h[0, -1, :]).item())
        layer_norms.append({
            "layer": l_idx,
            "attn_norm_at_boundary": attn_norm,
            "mlp_norm_at_boundary": mlp_norm,
            "total_residual_norm": total_norm,
            "mlp_to_attn_ratio": mlp_norm / (attn_norm + 1e-8)
        })

    # Final norm
    h_final = model.final_norm(h)
    logits = model.lm_head(h_final)[0, -1, :]
    probs = F.softmax(logits, dim=-1)

    top5_vals, top5_idx = torch.topk(probs, 5)
    top5_list = [{"tok": tok.decode([int(idx)]), "prob": float(val)} for idx, val in zip(top5_idx, top5_vals)]

    results = {
        "sequence_length": seq_len,
        "boundary_token": tok.decode([input_ids[-1]]),
        "target_1981_token_id": tok_1981,
        "target_1981_token_positions": pos_1981,
        "layer_dynamics": layer_norms,
        "final_boundary_logits_top5": top5_list,
        "attractor_dominance": {
            "top1_token": top5_list[0]["tok"],
            "top1_prob": top5_list[0]["prob"],
            "prob_1981": float(probs[tok_1981].item())
        }
    }

    out_file = "experiments/ucip26/mechanistic/mechanistic_diagnostics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Phase 6 Mechanistic Diagnostics Complete.")
    print(f"Boundary Token: {results['boundary_token']}")
    for l in layer_norms:
        print(f"  Layer {l['layer']}: Attn Norm={l['attn_norm_at_boundary']:.3f}, MLP Norm={l['mlp_norm_at_boundary']:.3f}, Ratio={l['mlp_to_attn_ratio']:.2f}")
    print(f"Top-1 Token at Generation Boundary: {results['attractor_dominance']['top1_token']} ({results['attractor_dominance']['top1_prob']*100:.2f}%) vs 1981 ({results['attractor_dominance']['prob_1981']*100:.3f}%)")


if __name__ == "__main__":
    run_mechanistic_diagnostics()
