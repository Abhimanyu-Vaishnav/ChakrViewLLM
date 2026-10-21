import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
model = ChakrMicro(ModelConfig(max_seq_len=512))
st = torch.load(REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "checkpoints" / "exp02_best_checkpoint.pt", map_location="cpu", weights_only=False)
model.load_state_dict(st["model_state_dict"])
model.eval()

# Check nominal prompt
prompt = "Instruction: Extract the system status as a JSON object with key \"status\".\nContext: Telemetry monitor report: Component subsystem_alpha_01 reported status 'nominal' during periodic scan.\nResponse: {\"status\": \""
toks = [0] + tok.encode(prompt)
with torch.no_grad():
    logits = model(torch.tensor([toks]))[0, -1, :]

top10_v, top10_i = torch.topk(logits, 10)
print("Nominal prompt tokens after '{\"status\": \"':")
for v, i in zip(top10_v, top10_i):
    ind = int(i.item())
    print(f"  {ind:5d} ({repr(tok.decode([ind])):10s}): logit = {v.item():.2f}")

# Check token for 'no'
no_tok = tok.encode("no")[0]
print(f"Token for 'no' ({no_tok}): logit = {logits[no_tok].item():.2f}")
