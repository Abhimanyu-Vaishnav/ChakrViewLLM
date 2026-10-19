import json
from pathlib import Path
import torch
import torch.nn.functional as F

import sys
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")

val_path = Path("experiments/single_field_copy_pilot/authorized_run_001/data/val.jsonl")
val_records = [json.loads(line) for line in open(val_path, encoding="utf-8")]

# Load candidate model
device = torch.device("cpu")
cfg = ModelConfig()
cand_model = ChakrMicro(cfg).to(device)
cand_ckpt = torch.load("experiments/single_field_copy_pilot/authorized_run_001/checkpoint_step_200.pt", map_location=device, weights_only=False)
cand_model.load_state_dict(cand_ckpt["model_state_dict"], strict=True)
cand_model.eval()

# Load base model
base_model = ChakrMicro(cfg).to(device)
base_ckpt = torch.load("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt", map_location=device, weights_only=False)
base_model.load_state_dict(base_ckpt["model_state_dict"], strict=True)
base_model.eval()

print("Loaded models successfully.")

# Check first 5 records in detail
for idx in range(5):
    rec = val_records[idx]
    prompt = rec["prompt"]
    exp_val = rec["expected_value"]
    target_str = rec["target"]  # e.g. " standby"
    
    # 1. Without BOS (how generate_greedy ran)
    p_ids_no_bos = tokenizer.encode(prompt)
    inp_tensor_no_bos = torch.tensor([p_ids_no_bos], dtype=torch.long)
    with torch.no_grad():
        logits_no_bos = cand_model(inp_tensor_no_bos)
        next_tok_no_bos = int(torch.argmax(logits_no_bos[0, -1, :]).item())
        
    # 2. With BOS (how training input was formed: [0] + prompt + target)
    p_ids_with_bos = [0] + p_ids_no_bos
    inp_tensor_with_bos = torch.tensor([p_ids_with_bos], dtype=torch.long)
    with torch.no_grad():
        logits_with_bos = cand_model(inp_tensor_with_bos)
        next_tok_with_bos = int(torch.argmax(logits_with_bos[0, -1, :]).item())
        
    # Check top 5 next tokens in both cases
    probs_no_bos = F.softmax(logits_no_bos[0, -1, :], dim=-1)
    top5_no_bos = torch.topk(probs_no_bos, 5)
    
    probs_with_bos = F.softmax(logits_with_bos[0, -1, :], dim=-1)
    top5_with_bos = torch.topk(probs_with_bos, 5)
    
    # Expected target tokens
    tgt_tokens = tokenizer.encode(target_str)
    
    print(f"\n--- Val Record {idx} ({rec['id']}) ---")
    print(f"  Field: {rec['target_field']} | Expected: '{exp_val}' | Target string: '{target_str}'")
    print(f"  Expected target tokens: {tgt_tokens} -> {[tokenizer.decode([t]) for t in tgt_tokens]}")
    print(f"  WITHOUT BOS top-5: {[(int(tok.item()), tokenizer.decode([int(tok.item())]), f'{prob.item():.4f}') for tok, prob in zip(top5_no_bos.indices, top5_no_bos.values)]}")
    print(f"  WITH BOS top-5:    {[(int(tok.item()), tokenizer.decode([int(tok.item())]), f'{prob.item():.4f}') for tok, prob in zip(top5_with_bos.indices, top5_with_bos.values)]}")

    # Check context tokens to see if the expected value tokens appear in context
    ctx_ids = tokenizer.encode(rec["context"])
    print(f"  Context has target tokens? {all(t in ctx_ids for t in tgt_tokens)}")
