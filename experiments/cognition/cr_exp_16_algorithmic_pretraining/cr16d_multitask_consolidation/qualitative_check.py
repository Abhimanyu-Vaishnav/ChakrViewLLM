import torch, json
from pathlib import Path
import sys

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

BENCH_COMP_PATH = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/data/cr16d_compositional_benchmark_1000.json"
with open(BENCH_COMP_PATH) as f:
    bench = json.load(f)

ckpt_path = REPO_ROOT / "experiments/cognition/cr_exp_16_algorithmic_pretraining/cr16d_multitask_consolidation/checkpoints/arm_c_replay50_seed101/arm_c_replay50_best.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
config = ckpt["model_config"] if "model_config" in ckpt else ckpt["config"]
model = ChakrMicro(config).to(device)
model.load_state_dict(state)
model.eval()

def generate(model, prompt_tokens, max_new_tokens=15):
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids)
            logits = out[0] if isinstance(out, tuple) else out
            next_token = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            token_id = next_token.item()
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if token_id == 1: break
            dec = tokenizer.decode([token_id], errors="replace")
            if "\n" in dec: break
    out_str = tokenizer.decode(input_ids[0, len(prompt_tokens):].tolist(), errors="replace")
    return out_str.strip().split("\n")[0].strip()

print("=== G2 (Unseen Values) Sample Predictions ===")
g2_items = [x for x in bench if x["group_code"] == "G2"][:5]
for item in g2_items:
    pred = generate(model, tokenizer.encode(item["prompt"]))
    print("Prompt:", repr(item["prompt"]))
    print("Target:", repr(item["expected_target"]), "| Pred:", repr(pred))

print("\n=== G1 (Seen Combos) Sample Predictions ===")
g1_items = [x for x in bench if x["group_code"] == "G1"][:5]
for item in g1_items:
    pred = generate(model, tokenizer.encode(item["prompt"]))
    print("Prompt:", repr(item["prompt"]))
    print("Target:", repr(item["expected_target"]), "| Pred:", repr(pred))

print("\n=== G3 (Unseen Keys) Sample Predictions ===")
g3_items = [x for x in bench if x["group_code"] == "G3"][:5]
for item in g3_items:
    pred = generate(model, tokenizer.encode(item["prompt"]))
    print("Prompt:", repr(item["prompt"]))
    print("Target:", repr(item["expected_target"]), "| Pred:", repr(pred))

print("\n=== G4 (Both Unseen) Sample Predictions ===")
g4_items = [x for x in bench if x["group_code"] == "G4"][:5]
for item in g4_items:
    pred = generate(model, tokenizer.encode(item["prompt"]))
    print("Prompt:", repr(item["prompt"]))
    print("Target:", repr(item["expected_target"]), "| Pred:", repr(pred))
