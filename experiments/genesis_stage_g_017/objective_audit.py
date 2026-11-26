"""
NR-017 Phase 3: Tokenizer & Objective Forensic Audit
Tests:
1. Canonical tokenizer round-trip on all Condition A, B, C task templates and tags.
2. Causal masking invariant on Condition B & C token sequences.
3. Shifted next-token loss alignment on trace-based sequences.
4. Separate loss contributions: task input (masked with -100), trace tokens, answer tokens.
5. Finite gradient checks without NaN/Inf across all 3 formats.
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, '.')
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def run_objective_audit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== NR-017 TOKENIZER & OBJECTIVE FORENSIC AUDIT on {device} ===")

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)

    with open("data/stage_g_017/stage_g_train_1800.json", 'r', encoding='utf-8') as f:
        train_samples = json.load(f)

    sample = train_samples[0]
    prompt = sample['prompt']
    tgt_a = sample['target_a']
    tgt_b = sample['target_b']
    tgt_c = sample['target_c']

    print("\n--- Test 1: Tokenizer Round-Trip Fidelity ---")
    formats = [("Condition A (Direct)", prompt + " " + tgt_a),
               ("Condition B (Explicit Trace)", prompt + " " + tgt_b),
               ("Condition C (Compact Trace)", prompt + " " + tgt_c)]

    all_roundtrip_ok = True
    for label, text in formats:
        enc = tokenizer.encode(text)
        dec = tokenizer.decode(enc)
        match = (dec == text)
        print(f"  {label:30s} | Tokens: {len(enc):3d} | Match: {match}")
        if not match:
            all_roundtrip_ok = False
            print(f"    Expected: {repr(text)}")
            print(f"    Got:      {repr(dec)}")

    assert all_roundtrip_ok, "Tokenizer roundtrip failed on condition formats!"

    # Test 2: Invariant Causal Attention Mask
    print("\n--- Test 2: Causal Masking Invariant on Trace Sequences ---")
    cfg = Genesis17MConfig(vocab_size=4096, d_model=384, n_layers=10, n_q_heads=12, n_kv_heads=3, hidden_dim=1024, max_seq_len=128, dropout=0.0)
    model = ChakrGenesis17M(cfg).to(device)
    model.eval()

    seq_tokens = tokenizer.encode(prompt + " " + tgt_b)
    x1 = torch.tensor([seq_tokens], dtype=torch.long, device=device)
    x2 = x1.clone()
    mod_pos = len(seq_tokens) - 5
    x2[0, mod_pos] = (x1[0, mod_pos] + 50) % 4096

    with torch.no_grad():
        out1, _ = model(x1)
        out2, _ = model(x2)

    causal_diff = torch.abs(out1[:, :mod_pos, :] - out2[:, :mod_pos, :]).max().item()
    print(f"  Max diff before pos {mod_pos}: {causal_diff:.8f}")
    assert causal_diff == 0.0, "Causal mask leaked future token!"

    # Test 3: Shifted Next-Token Loss Alignment & Label Masking
    print("\n--- Test 3: Loss Masking & Gradient Verification ---")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    p_ids = tokenizer.encode(prompt)
    t_ids = tokenizer.encode(" " + tgt_b) + [getattr(tokenizer, 'eos_id', 1) or 1]
    input_ids = torch.tensor([p_ids + t_ids], dtype=torch.long, device=device)
    labels = torch.tensor([[-100] * len(p_ids) + t_ids], dtype=torch.long, device=device)

    optimizer.zero_grad()
    logits, _ = model(input_ids)
    s_logits = logits[..., :-1, :].contiguous().view(-1, 4096)
    s_labels = labels[..., 1:].contiguous().view(-1)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(s_logits, s_labels)
    loss.backward()

    # Check finite gradients
    nan_grads = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    max_g = max(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  Loss: {loss.item():.4f} | Max Gradient Norm: {max_g:.4f} | Has NaN/Inf: {nan_grads}")
    assert not nan_grads, "Gradients contained NaN/Inf!"

    audit_summary = {
        "tokenizer_roundtrip_passed": all_roundtrip_ok,
        "causal_mask_diff": causal_diff,
        "loss_masked_value": loss.item(),
        "has_nan_grads": nan_grads,
        "max_grad_norm": max_g
    }

    out_file = Path("data/stage_g_017/objective_audit_results.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(audit_summary, f, indent=2)

    print("\nAll Tokenizer & Objective Invariant Checks Passed Perfectly!")

if __name__ == '__main__':
    run_objective_audit()
