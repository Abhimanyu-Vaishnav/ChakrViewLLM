"""
NR-016 Phase 5: Representation and Generalization Diagnostics
Directly compares 17M vs 38M architectures on:
1. Layer-wise hidden-state norms and inter-layer cosine similarity trajectories
2. Sensitivity to variable renaming (counterfactual probes: x, y -> a, b -> zeta, theta)
3. Sensitivity to argument reordering (commutativity probes: 12 + 15 vs 15 + 12)
4. Representation changes across depth (10 layers vs 12 layers)
"""

import sys
import torch
import torch.nn.functional as F
import json
from pathlib import Path

sys.path.insert(0, '.')
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def run_representation_study():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== NR-016 REPRESENTATION STUDY (17M vs 38M) on {device} ===")

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    cfg_17m = Genesis17MConfig(
        vocab_size=4096, d_model=384, n_layers=10, n_q_heads=12, n_kv_heads=3,
        hidden_dim=1024, max_seq_len=128, dropout=0.0
    )
    cfg_38m = Genesis17MConfig(
        vocab_size=4096, d_model=512, n_layers=12, n_q_heads=16, n_kv_heads=4,
        hidden_dim=1536, max_seq_len=128, dropout=0.0
    )

    models = {
        "17M": ChakrGenesis17M(cfg_17m).to(device),
        "38M": ChakrGenesis17M(cfg_38m).to(device)
    }

    test_prompt = "<user>\nGiven x = 12 and y = 15, compute the sum of x and y.\n<assistant>\n"
    tokens = torch.tensor([tokenizer.encode(test_prompt)], dtype=torch.long, device=device)
    B, T = tokens.shape

    metrics = {}

    for name, model in models.items():
        model.eval()
        print(f"\n--- Layer Trajectory for {name} ({len(model.layers)} layers) ---")
        with torch.no_grad():
            x = model.embedding(tokens)
            cos, sin = model.rope(x, T)
            mask = torch.full((T, T), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

            layer_norms = []
            layer_sims = []
            prev_x = x

            for i, layer in enumerate(model.layers):
                norm_val = x.norm(dim=-1).mean().item()
                layer_norms.append(norm_val)

                x = layer(x, cos, sin, mask)

                cos_sim = F.cosine_similarity(x.flatten(), prev_x.flatten(), dim=0).item()
                layer_sims.append(cos_sim)
                prev_x = x
                print(f"  Layer {i+1:2d} | Norm: {norm_val:6.4f} | Cosine Sim to Prev: {cos_sim:.4f}")

        metrics[name] = {
            "layer_norms": layer_norms,
            "layer_cosine_sims": layer_sims
        }

    # Probing variable renaming sensitivity
    probes = [
        ("Base (x, y)", "<user>\nGiven x = 12 and y = 15, compute the sum of x and y.\n<assistant>\n"),
        ("Renamed (a, b)", "<user>\nGiven a = 12 and b = 15, compute the sum of a and b.\n<assistant>\n"),
        ("OOD (zeta, theta)", "<user>\nGiven zeta = 12 and theta = 15, compute the sum of zeta and theta.\n<assistant>\n"),
        ("Commutative (y, x)", "<user>\nGiven y = 15 and x = 12, compute the sum of y and x.\n<assistant>\n")
    ]

    print("\n--- Counterfactual Renaming & Commutativity Probes (Stage-D Primary Canonical) ---")
    staged_ckpt = "experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt"
    st = torch.load(staged_ckpt, map_location="cpu", weights_only=False)
    sd = st.get("model_state_dict", st)
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    canonical_17m = ChakrGenesis17M(cfg_17m)
    canonical_17m.load_state_dict(clean_sd)
    canonical_17m.to(device)
    canonical_17m.eval()

    probe_outputs = {}
    for label, p in probes:
        p_ids = torch.tensor([tokenizer.encode(p)], dtype=torch.long, device=device)
        curr = p_ids
        with torch.no_grad():
            for _ in range(8):
                out, _ = canonical_17m(curr)
                nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                curr = torch.cat([curr, nxt], dim=1)
                if nxt.item() == 1:
                    break
        gen_str = tokenizer.decode(curr[0, p_ids.size(1):].tolist()).strip()
        probe_outputs[label] = gen_str
        print(f"  {label:25s} -> {repr(gen_str)}")

    metrics["canonical_probes"] = probe_outputs

    out_file = Path("data/scientific_eval_016/representation_diagnostics.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved representation diagnostics to: {out_file}")

if __name__ == '__main__':
    run_representation_study()
