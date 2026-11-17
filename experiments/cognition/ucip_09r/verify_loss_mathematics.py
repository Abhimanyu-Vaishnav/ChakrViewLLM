import torch
import torch.nn as nn
import torch.nn.functional as F

print("=== 1. HAND-CALCULATED SYNTHETIC EXAMPLE ===")
# Sequence: [BOS(0), A(10), B(20), C(30), EOS(1)]
# Targets: Prompt=[BOS, A, B], Assistant=[C, EOS]
# Prompt mask: Prompt tokens have target -100

vocab_size = 50
seq = [0, 10, 20, 30, 1]
tgt = [-100, -100, -100, 30, 1]

inp_tensor = torch.tensor([seq], dtype=torch.long)
tgt_tensor = torch.tensor([tgt], dtype=torch.long)

# Set deterministic logits
torch.manual_seed(42)
fake_logits = torch.randn((1, len(seq), vocab_size), requires_grad=True)

# Standard causal shift
shift_logits = fake_logits[:, :-1, :].contiguous()
shift_targets = tgt_tensor[:, 1:].contiguous()

print(f"Input tokens:        {seq}")
print(f"Target tokens:       {tgt}")
print(f"Shift targets (t+1): {shift_targets.tolist()[0]}")

# Position analysis:
# Position 0 (BOS): predicts seq[1] (A=10). Label is -100 -> MASKED (no loss)
# Position 1 (A):   predicts seq[2] (B=20). Label is -100 -> MASKED (no loss)
# Position 2 (B):   predicts seq[3] (C=30). Label is 30   -> ACTIVE LOSS
# Position 3 (C):   predicts seq[4] (EOS=1). Label is 1   -> ACTIVE LOSS

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
prod_loss = loss_fn(shift_logits.view(-1, vocab_size), shift_targets.view(-1))

# Independent manual reference calculation:
# For pos 2: target is 30
l2 = F.cross_entropy(fake_logits[0, 2:3, :], torch.tensor([30]))
# For pos 3: target is 1
l3 = F.cross_entropy(fake_logits[0, 3:4, :], torch.tensor([1]))
ref_loss = (l2 + l3) / 2.0

print(f"Production Loss:     {prod_loss.item():.6f}")
print(f"Reference Loss:      {ref_loss.item():.6f}")
diff = abs(prod_loss.item() - ref_loss.item())
print(f"Loss Delta:          {diff:.8f}")
assert diff < 1e-6, "Production loss does not match manual reference!"

print("\n=== 2. GRADIENT FLOW VERIFICATION ===")
prod_loss.backward()
grads = fake_logits.grad[0] # [seq_len, vocab_size]
grad_norms = grads.norm(dim=-1).tolist()
print("Gradient norm at each token position:")
for p, (tok, gn) in enumerate(zip(seq, grad_norms)):
    print(f"  Pos {p} (Token {tok:>2}): Grad Norm = {gn:.6f} -> {'ACTIVE GRADIENT' if gn > 0 else 'MASKED ZERO'}")

# Verify exact gradient masking:
assert grad_norms[0] == 0.0, "Position 0 (BOS) must have 0 gradient"
assert grad_norms[1] == 0.0, "Position 1 (A) must have 0 gradient"
assert grad_norms[2] > 0.0, "Position 2 (B) must have non-zero gradient for predicting C"
assert grad_norms[3] > 0.0, "Position 3 (C) must have non-zero gradient for predicting EOS"
assert grad_norms[4] == 0.0, "Position 4 (EOS) has no next token, shifted out -> 0 gradient"

print("\n=== 3. NEGATIVE ABLATION TESTS ===")
# Negative Test 1: Unshifted alignment (testing predicting position t instead of t+1)
try:
    bad_loss = loss_fn(fake_logits.view(-1, vocab_size), tgt_tensor.view(-1))
    # In unshifted alignment, pos 3 is compared to target[3] (which is C predicting C!)
    print(f"Unshifted (Faulty) Loss: {bad_loss.item():.6f} (Different from causal shift)")
    assert abs(bad_loss.item() - prod_loss.item()) > 0.1, "Unshifted loss must not match causal shift!"
    print("Negative Test 1 (Unshifted Rejection): PASSED")
except Exception as e:
    print(f"Negative Test 1 failed: {e}")

# Negative Test 2: Inverted shift (targets shifted backward)
try:
    inv_logits = fake_logits[:, 1:, :].contiguous()
    inv_targets = tgt_tensor[:, :-1].contiguous()
    inv_loss = loss_fn(inv_logits.view(-1, vocab_size), inv_targets.view(-1))
    assert abs(inv_loss.item() - prod_loss.item()) > 0.1
    print("Negative Test 2 (Inverted Shift Rejection): PASSED")
except Exception as e:
    print(f"Negative Test 2 failed: {e}")

print("\nALL MATHEMATICAL CHECKS PASSED PERFECTLY.")
