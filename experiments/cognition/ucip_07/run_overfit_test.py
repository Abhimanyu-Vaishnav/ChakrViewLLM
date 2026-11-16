import os
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.cognition.ucip_07.evolved_model import get_architecture_preset
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer, _ = load_tokenizer_artifacts(os.path.join(REPO_ROOT, "data", "experiments", "vocab_4096"))

data = [
    ("User: What is 2 + 2?\nAssistant:", " 2 + 2 = 4."),
    ("User: Say exactly: The sky is blue.\nAssistant:", " The sky is blue."),
    ("User: What does a CPU do?\nAssistant:", " A CPU executes instructions from memory."),
    ("User: नमस्ते! आप क्या कर सकते हैं?\nAssistant:", " मैं आपकी सहायता कर सकता हूँ।"),
]

model = get_architecture_preset("core")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

print(f"Beginning 4-sample overfit verification on ChakrCore (27.8M parameters)...")
for step in range(40):
    total_loss = 0.0
    for p, t in data:
        p_ids = tokenizer.encode(p, add_bos=True, add_eos=False)
        t_ids = tokenizer.encode(t, add_bos=False, add_eos=True)
        inp = torch.tensor([p_ids + t_ids], dtype=torch.long)
        tgt = torch.tensor([[-100] * len(p_ids) + t_ids], dtype=torch.long)

        logits = model(inp)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = tgt[:, 1:].contiguous()
        loss = criterion(shift_logits.view(-1, 4096), shift_targets.view(-1))

        loss.backward()
        total_loss += loss.item()

    optimizer.step()
    optimizer.zero_grad()
    if (step + 1) % 10 == 0:
        print(f"Step {step+1:2d} | Overfit Loss: {total_loss / len(data):.4f}")

print("\n--- Autoregressive Verification ---")
model.eval()
all_passed = True
for p, t in data:
    curr = tokenizer.encode(p, add_bos=True, add_eos=False)
    gen = []
    with torch.no_grad():
        for _ in range(20):
            t_in = torch.tensor([curr], dtype=torch.long)
            nxt = torch.argmax(model(t_in)[0, -1, :]).item()
            if nxt == 1:
                break
            gen.append(nxt)
            curr.append(nxt)
    decoded = tokenizer.decode(gen)
    match = (decoded.strip() == t.strip())
    if not match:
        all_passed = False
    print(f"Prompt: {p.splitlines()[0]}")
    print(f"  Target:  {t.strip()!r}")
    print(f"  Decoded: {decoded.strip()!r} (Exact: {match})\n")

print(f"OVERFIT VERIFICATION SUCCESS: {all_passed}")
