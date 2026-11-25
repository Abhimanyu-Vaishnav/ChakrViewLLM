import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

sys.path.insert(0, '.')
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.runtime.inference_17m import ChakrViewInferenceEngine17M

def test_causal_invariants_and_overfit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    # 1. Initialize fresh 17M model
    config = Genesis17MConfig(
        vocab_size=4096,
        d_model=384,
        n_layers=10,
        n_q_heads=12,
        n_kv_heads=3,
        hidden_dim=1024,
        max_seq_len=512,
        dropout=0.0
    )
    model = ChakrGenesis17M(config).to(device)
    model.train()

    # 2. Invariant 1: Causal masking (strictly zero future token leakage)
    seq_len = 16
    x1 = torch.randint(3, 4096, (1, seq_len), device=device)
    x2 = x1.clone()
    # Change token at position 10
    x2[0, 10] = (x1[0, 10] + 50) % 4096

    with torch.no_grad():
        out1, _ = model(x1)
        out2, _ = model(x2)

    diff = torch.abs(out1[:, :10, :] - out2[:, :10, :]).max().item()
    print(f"Invariant 1 (Causal Masking): max diff before altered token = {diff:.8f}")
    assert diff == 0.0, f"Causal mask leaked future token! Diff: {diff}"

    # 3. Invariant 2: Shifted Next-Token Loss alignment
    # logits[..., :-1, :] predicts labels[..., 1:]
    logits, _ = model(x1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = x1[..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(shift_logits.view(-1, 4096), shift_labels.view(-1))
    print(f"Invariant 2 (Next-Token Loss computed successfully): loss = {loss.item():.4f}")

    # 4. Invariant 3: Controlled Memorization Test (5 tiny sequences)
    # Can the model achieve near-zero loss and 100% exact reproduction on 5 sequences?
    print("\n--- Running Controlled Memorization Test (5 synthetic sequences) ---")
    tiny_sequences = [
        torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.long, device=device),
        torch.tensor([100, 105, 110, 115, 120, 125, 130, 135], dtype=torch.long, device=device),
        torch.tensor([500, 400, 300, 200, 100, 50, 25, 12], dtype=torch.long, device=device),
        torch.tensor([7, 14, 21, 28, 35, 42, 49, 56], dtype=torch.long, device=device),
        torch.tensor([99, 88, 77, 66, 55, 44, 33, 22], dtype=torch.long, device=device),
    ]
    batch = torch.stack(tiny_sequences) # (5, 8)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    for step in range(1, 101):
        optimizer.zero_grad()
        out, _ = model(batch)
        s_logits = out[:, :-1, :].contiguous().view(-1, 4096)
        s_labels = batch[:, 1:].contiguous().view(-1)
        step_loss = loss_fn(s_logits, s_labels)
        step_loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1:
            preds = out[:, :-1, :].argmax(dim=-1)
            acc = (preds == batch[:, 1:]).float().mean().item() * 100
            print(f"Step {step:3d} | Loss: {step_loss.item():.6f} | Next-Token Acc: {acc:.1f}%")

    # Evaluate exact autoregressive sequence reproduction from prompt [0]
    model.eval()
    print("\nVerifying Exact Autoregressive Generation from prefix:")
    all_matched = True
    for idx, seq in enumerate(tiny_sequences):
        prompt = seq[:1].unsqueeze(0) # (1, 1)
        curr = prompt
        with torch.no_grad():
            for _ in range(len(seq) - 1):
                logits, _ = model(curr)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                curr = torch.cat([curr, next_token], dim=1)
        generated = curr[0].tolist()
        expected = seq.tolist()
        match = (generated == expected)
        print(f"Seq {idx+1}: Expected {expected} -> Generated {generated} | Match: {match}")
        if not match:
            all_matched = False

    print(f"\nControlled Memorization Result: {'PASSED (100% exact reproduction)' if all_matched else 'FAILED'}")
    return all_matched

if __name__ == '__main__':
    res = test_causal_invariants_and_overfit()
    sys.exit(0 if res else 1)
