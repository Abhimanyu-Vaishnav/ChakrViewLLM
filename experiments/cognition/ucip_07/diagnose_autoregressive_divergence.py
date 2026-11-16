import os
import sys
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer, _ = load_tokenizer_artifacts(os.path.join(REPO_ROOT, "data", "experiments", "vocab_4096"))

ckpt_path = os.path.join(REPO_ROOT, "releases", "ucip_06_candidate", "chakrview_ucip06_medium_model.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model = ChakrMicro(ModelConfig(**ckpt["config"]))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

prompt = "User: What does a CPU do?\nAssistant:"
target = " A CPU retrieves instructions from memory, decodes them, and executes mathematical and logical operations."

prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
target_ids = tokenizer.encode(target, add_bos=False, add_eos=True)
full_ids = prompt_ids + target_ids

inp = torch.tensor([full_ids], dtype=torch.long)
with torch.no_grad():
    logits = model(inp)[0]  # (seq_len, vocab_size)

print("=== Teacher-Forced Step-by-Step Analysis ===")
first_err = None
for pos in range(len(prompt_ids) - 1, min(len(prompt_ids) + 12, len(full_ids) - 1)):
    step = pos - (len(prompt_ids) - 1)
    tgt_tok = full_ids[pos + 1]
    step_logits = logits[pos]
    step_probs = torch.softmax(step_logits, dim=-1)
    pred_tok = torch.argmax(step_logits).item()
    tgt_prob = step_probs[tgt_tok].item()
    nll = -torch.log(step_probs[tgt_tok] + 1e-12).item()

    top3 = torch.topk(step_probs, 3)
    top3_str = ", ".join([f"{tokenizer.decode([idx.item()])!r}: {val.item():.3f}" for idx, val in zip(top3.indices, top3.values)])

    match = (pred_tok == tgt_tok)
    if not match and first_err is None:
        first_err = (step, tokenizer.decode([tgt_tok]), tokenizer.decode([pred_tok]))

    print(f"Step {step:2d} | Tgt: {tokenizer.decode([tgt_tok])!r:<12} (prob={tgt_prob:.4f}, nll={nll:.2f}) | Pred: {tokenizer.decode([pred_tok])!r:<12} | Top3: [{top3_str}]")

if first_err:
    print(f"\nFirst Teacher-Forced Discrepancy at Step {first_err[0]}: Expected {first_err[1]!r}, Predicted {first_err[2]!r}")

# Autoregressive generation rollout
curr = list(prompt_ids)
gen = []
for _ in range(25):
    t_in = torch.tensor([curr], dtype=torch.long)
    with torch.no_grad():
        nxt = torch.argmax(model(t_in)[0, -1, :]).item()
    if nxt == 1:
        break
    gen.append(nxt)
    curr.append(nxt)

print(f"\nGreedy Autoregressive Decoded:\n\"{tokenizer.decode(gen)}\"")
