import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from pathlib import Path

REPO_ROOT = Path(r"D:\Project\ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from scripts.run_m3_evaluations import DirectNeuralRunner

tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")

# 1. Exact sequence copying & small mapping tasks
TRAIN_TASKS = [
    ("User: Copy: Alpha Beta Gamma\nAssistant:", " Alpha Beta Gamma"),
    ("User: Copy: The ocean is vast and deep.\nAssistant:", " The ocean is vast and deep."),
    ("User: What is 2 + 2?\nAssistant:", " 4"),
    ("User: What is 5 - 2?\nAssistant:", " 3"),
    ("User: Capital of France?\nAssistant:", " Paris")
]

HELD_OUT_TASKS = [
    ("User: Copy: Delta Epsilon Zeta\nAssistant:", " Delta Epsilon Zeta"),
    ("User: What is 3 + 3?\nAssistant:", " 6"),
    ("User: What is 6 - 2?\nAssistant:", " 4")
]

def collate_task(p, t):
    p_ids = tokenizer.encode(p, add_bos=True, add_eos=False)
    t_ids = tokenizer.encode(t, add_bos=False, add_eos=True)
    inp = torch.tensor([p_ids + t_ids], dtype=torch.long)
    tgt = torch.tensor([[-100] * len(p_ids) + t_ids], dtype=torch.long)
    return inp, tgt

def run_proof():
    print("=== PHASE G: MINIMAL SYNTHETIC LEARNING PROOF ===")
    parent_path = REPO_ROOT / "releases" / "ucip_09_candidates" / "chakrview_ucip09_medium_model.pt"
    runner = DirectNeuralRunner(parent_path)
    model = runner.model
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    print("Initial Evaluation on Train Tasks:")
    for p, expected in TRAIN_TASKS:
        res = runner.generate(p, max_new_tokens=15, temperature=0.0)
        print(f"  {p.replace(chr(10), ' ')} -> Got: {repr(res['text'].strip())} (Exp: {expected.strip()})")

    # Train for 20 steps
    print("\nTraining on 5 synthetic examples...")
    for step in range(1, 21):
        total_loss = 0.0
        optimizer.zero_grad()
        for p, t in TRAIN_TASKS:
            inp, tgt = collate_task(p, t)
            logits = model(inp)
            s_logits = logits[:, :-1, :].contiguous()
            s_tgt = tgt[:, 1:].contiguous()
            loss = criterion(s_logits.view(-1, s_logits.size(-1)), s_tgt.view(-1))
            loss.backward()
            total_loss += loss.item()
        optimizer.step()
        if step % 5 == 0:
            print(f"  Step {step:>2}/20 | Loss: {total_loss / len(TRAIN_TASKS):.4f}")

    print("\nPost-Training Evaluation on Train Tasks:")
    train_correct = 0
    for p, expected in TRAIN_TASKS:
        res = runner.generate(p, max_new_tokens=15, temperature=0.0)
        got = res['text'].strip()
        is_match = (got == expected.strip())
        if is_match:
            train_correct += 1
        print(f"  {p.replace(chr(10), ' ')} -> Got: {repr(got)} (Match: {is_match})")

    print("\nPost-Training Evaluation on Held-Out Tasks:")
    for p, expected in HELD_OUT_TASKS:
        res = runner.generate(p, max_new_tokens=15, temperature=0.0)
        print(f"  {p.replace(chr(10), ' ')} -> Got: {repr(res['text'].strip())} (Exp: {expected.strip()})")

    print(f"\nProof Result: {train_correct}/{len(TRAIN_TASKS)} training tasks perfectly memorized.")
    assert train_correct >= 4, "Neural learning failed to acquire minimal synthetic tasks!"

if __name__ == "__main__":
    run_proof()
