import json
import re
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096")
ckpt = torch.load(REPO_ROOT / "experiments" / "batch541_580" / "checkpoints" / "candidate_step250_intent_understanding.pt", map_location="cpu", weights_only=False)
model = ChakrMicro(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

prompt = "User: Turn 'SAFE GOVERNED RUNTIME' into all small letters\nIntent:"
p_toks = [0] + tokenizer.encode(prompt)
gen = list(p_toks)
with torch.no_grad():
    for _ in range(80):
        x = torch.tensor([gen[-256:]], dtype=torch.long)
        nxt = int(torch.argmax(model(x)[0, -1, :]).item())
        if nxt == 1:
            break
        gen.append(nxt)

raw_gen = tokenizer.decode(gen[len(p_toks):], errors="replace").strip()
print("RAW_GEN:", repr(raw_gen))

try:
    m = re.search(r"({.*})", raw_gen)
    print("MATCH OBJECT:", m)
    if m:
        val = m.group(1)
        print("GROUP 1:", repr(val))
        pred_obj = json.loads(val)
        print("PRED OBJ:", pred_obj)
except Exception as e:
    print("CAUGHT EXCEPTION:", type(e), e)
