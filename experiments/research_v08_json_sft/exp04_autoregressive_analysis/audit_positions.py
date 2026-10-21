import json
import torch
import torch.nn as nn
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro

tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
model = ChakrMicro(ModelConfig(max_seq_len=512))
st = torch.load(REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "checkpoints" / "exp02_best_checkpoint.pt", map_location="cpu", weights_only=False)
model.load_state_dict(st["model_state_dict"])
model.eval()

with open(REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "data" / "train.jsonl") as f:
    rec = json.loads(f.readline())

p = rec["prompt"]
r = rec["response"]
expected_status = rec["expected_status"]
p_toks = tok.encode(p)
r_toks = tok.encode(r)

resp_start_pos = len(p_toks)
tf_input = [0] + p_toks + r_toks
tf_target = ([-100] * len(p_toks)) + r_toks + [1]

logits = model(torch.tensor([tf_input]))[0]

print(f"Status value: {expected_status}")
print(f"p_toks len: {len(p_toks)}, r_toks len: {len(r_toks)}, total tf_input len: {len(tf_input)}")
print("\nDetailed causal trace of response positions:")
for i in range(resp_start_pos, len(tf_input)):
    in_t = tf_input[i]
    tg_t = tf_target[i]
    pred_t = int(torch.argmax(logits[i, :]).item())
    in_s = tok.decode([in_t])
    tg_s = tok.decode([tg_t]) if tg_t != 1 else "<EOS>"
    pred_s = tok.decode([pred_t]) if pred_t != 1 else "<EOS>"
    match = (pred_t == tg_t)
    print(f"Pos {i:3d} | Input: {in_t:5d} ({repr(in_s):12s}) -> Target: {tg_t:5d} ({repr(tg_s):12s}) | Pred: {pred_t:5d} ({repr(pred_s):12s}) | Match: {match}")
