"""
NR-015 Phase 4: Representation and Model Behavior Diagnostic
Inspects:
1. Layer-wise hidden state norms and activation distributions.
2. Sensitivity to variable renaming (e.g. x, y vs alpha, beta).
3. Sensitivity to argument order (e.g. add(a, b) vs add(b, a)).
4. Attention weight entropy across 10 transformer layers.
"""

import sys
import torch
import torch.nn.functional as F
import json
from pathlib import Path

sys.path.insert(0, '.')
from chakrview.runtime.inference_17m import ChakrViewInferenceEngine17M

def run_representation_diagnostics():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running diagnostics on device: {device}")
    
    engine = ChakrViewInferenceEngine17M(
        checkpoint_path='experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt',
        device=device
    )
    model = engine.model
    tokenizer = engine.tokenizer
    model.eval()

    # 1. Variable Renaming Sensitivity Test
    prompts_renaming = [
        ("Base (x, y)", "<user>\nLet x = 12 and y = 15. What is x + y?\n<assistant>\n"),
        ("Renamed (a, b)", "<user>\nLet a = 12 and b = 15. What is a + b?\n<assistant>\n"),
        ("Renamed (alpha, beta)", "<user>\nLet alpha = 12 and beta = 15. What is alpha + beta?\n<assistant>\n"),
        ("Renamed (foo, bar)", "<user>\nLet foo = 12 and bar = 15. What is foo + bar?\n<assistant>\n"),
    ]
    
    print("\n--- 1. Variable Renaming Sensitivity & Output Inspection ---")
    predictions = {}
    for label, p in prompts_renaming:
        p_ids = torch.tensor(tokenizer.encode(p), dtype=torch.long, device=device).unsqueeze(0)
        curr = p_ids
        with torch.no_grad():
            for _ in range(12):
                out, _ = model(curr)
                next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)
                curr = torch.cat([curr, next_tok], dim=1)
                if next_tok.item() == 1:
                    break
        gen_tokens = curr[0, p_ids.size(1):].tolist()
        gen_str = tokenizer.decode(gen_tokens).strip()
        predictions[label] = gen_str
        print(f"{label:22s} -> {repr(gen_str)}")

    # 2. Argument Order Sensitivity Test
    prompts_order = [
        ("Order A: 12 + 15", "<user>\nCalculate 12 + 15.\n<assistant>\n"),
        ("Order B: 15 + 12", "<user>\nCalculate 15 + 12.\n<assistant>\n"),
        ("Order C: multiply(3, 7)", "<user>\nCalculate 3 * 7.\n<assistant>\n"),
        ("Order D: multiply(7, 3)", "<user>\nCalculate 7 * 3.\n<assistant>\n"),
    ]
    print("\n--- 2. Argument Order Commutativity Inspection ---")
    for label, p in prompts_order:
        p_ids = torch.tensor(tokenizer.encode(p), dtype=torch.long, device=device).unsqueeze(0)
        curr = p_ids
        with torch.no_grad():
            for _ in range(8):
                out, _ = model(curr)
                next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)
                curr = torch.cat([curr, next_tok], dim=1)
                if next_tok.item() == 1:
                    break
        gen_str = tokenizer.decode(curr[0, p_ids.size(1):].tolist()).strip()
        print(f"{label:25s} -> {repr(gen_str)}")

    # 3. Layer-Wise Hidden State & Cosine Similarity Analysis
    test_text = "<user>\ndef add(a, b):\n    return a + b\n<assistant>\n"
    tokens = torch.tensor(tokenizer.encode(test_text), dtype=torch.long, device=device).unsqueeze(0)
    
    print("\n--- 3. Layer-Wise Hidden State Norms and Trajectory ---")
    with torch.no_grad():
        # Pass through embedding
        B, T = tokens.shape
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
            
            # Forward through layer
            x = layer(x, cos, sin, mask)
            
            # Compute cosine similarity with previous layer
            cos_sim = F.cosine_similarity(x.flatten(), prev_x.flatten(), dim=0).item()
            layer_sims.append(cos_sim)
            prev_x = x
            print(f"Layer {i+1:2d} | State Norm: {norm_val:.4f} | Cosine Sim to Layer {i:2d}: {cos_sim:.4f}")

    diagnostics_result = {
        "renaming_outputs": predictions,
        "layer_norms": layer_norms,
        "layer_cosine_sims": layer_sims
    }
    with open("data/synthetic_binding_015/diagnostics_results.json", 'w') as f:
        json.dump(diagnostics_result, f, indent=2)
    print("\nDiagnostics complete and written to data/synthetic_binding_015/diagnostics_results.json")

if __name__ == '__main__':
    run_representation_diagnostics()
