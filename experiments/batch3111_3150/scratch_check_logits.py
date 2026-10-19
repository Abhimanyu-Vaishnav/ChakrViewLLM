import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

TOK_DIR = Path(__file__).resolve().parents[2] / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
CKPT_PATH = Path(__file__).resolve().parents[2] / "experiments" / "batch3111_3150" / "json_sft_v2_pilot_run" / "sft_best_checkpoint.pt"

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tok, _ = load_tokenizer_artifacts(str(TOK_DIR))
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model = ChakrMicro(ModelConfig(max_seq_len=512))
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()

prompt = "Instruction: Return a JSON object with fields 'status' and 'code' indicating server health status and return code.\nResponse:"

# 1. Prefix: '{"status"'
t1 = [0] + tok.encode(prompt + '{"status"')
# 2. Prefix: '{"status'
t2 = [0] + tok.encode(prompt + '{"status')

with torch.no_grad():
    l1 = model(torch.tensor([t1]))[0, -1, :]
    l2 = model(torch.tensor([t2]))[0, -1, :]

print("=== Logits after '{\"status\"' ===")
for tid in torch.topk(l1, 10).indices.tolist():
    print(f"ID {tid:4d} | {repr(tok.decode([tid])):15s} | score: {l1[tid].item():.2f}")

print("\n=== Logits after '{\"status' ===")
for tid in torch.topk(l2, 10).indices.tolist():
    print(f"ID {tid:4d} | {repr(tok.decode([tid])):15s} | score: {l2[tid].item():.2f}")
