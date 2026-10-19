import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")

val_path = Path("experiments/single_field_copy_pilot/authorized_run_001/data/val.jsonl")
val_records = [json.loads(line) for line in open(val_path, encoding="utf-8")]

device = torch.device("cpu")
cfg = ModelConfig()
cand_model = ChakrMicro(cfg).to(device)
cand_ckpt = torch.load("experiments/single_field_copy_pilot/authorized_run_001/checkpoint_step_200.pt", map_location=device, weights_only=False)
cand_model.load_state_dict(cand_ckpt["model_state_dict"], strict=True)
cand_model.eval()

rec = val_records[0]
prompt = rec["prompt"]
p_ids = tokenizer.encode(prompt)
p_ids_plus_space = p_ids + [35]

with torch.no_grad():
    logits = cand_model(torch.tensor([p_ids_plus_space], dtype=torch.long))
    probs = F.softmax(logits[0, -1, :], dim=-1)
    top10 = torch.topk(probs, 10)
    print("Top 10 tokens after 'Assistant: ':")
    for tok, prob in zip(top10.indices, top10.values):
        print(f"  Token {tok.item()}: {repr(tokenizer.decode([tok.item()]))} ({prob.item():.4f})")

    print("\nExpected first token of 'standby' is 304 ('st'):", probs[304].item())
    run_toks = tokenizer.encode("running")
    print("Tokens for 'running':", run_toks, [tokenizer.decode([t]) for t in run_toks])
    for rt in run_toks:
        print(f"Probability of token {rt} ({repr(tokenizer.decode([rt]))}):", probs[rt].item())
