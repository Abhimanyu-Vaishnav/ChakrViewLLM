import json
import torch
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
model = ChakrMicro(ModelConfig(max_seq_len=512))
st = torch.load(REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp05_slot_weighted_sft" / "checkpoints_treatment" / "treatment_best_checkpoint.pt", map_location="cpu", weights_only=False)
model.load_state_dict(st["model_state_dict"])
model.eval()

with open(REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "data" / "train.jsonl") as f:
    rec = json.loads(f.readline())

p = rec["prompt"]
# Check generation steps
curr = [0] + tok.encode(p)

print("Step-by-step greedy token emission for Treatment Arm:")
for step in range(8):
    with torch.no_grad():
        logits = model(torch.tensor([curr]))[0, -1, :]
    top5_v, top5_i = torch.topk(logits, 5)
    nxt = int(top5_i[0].item())
    tok_str = tok.decode([nxt])
    print(f"Step {step}: Emitted {nxt} ({repr(tok_str)}) | Top-3:")
    for v, i in zip(top5_v[:3].tolist(), top5_i[:3].tolist()):
        print(f"    {i:4d} ({repr(tok.decode([i])):10s}): {v:.2f}")
    curr.append(nxt)
