"""
Controlled training and evaluation runner for Batches 123–126.
Trains two controlled arms on the new prompt-diverse two-digit dataset:
- Arm A: Canonical ChakrMicro v0.1 Checkpoint initialization.
- Arm B: Copy-Task Pre-Trained Checkpoint initialization (`copy_task_sft_step150.pt`).

Both arms are trained under identical conditions:
- Dataset: experiments/batch123_126_prompt_diverse_multidigit/data/train.jsonl
- Steps: 200
- Optimizer: AdamW (lr=3e-4, weight_decay=0.01, grad_clip=1.0)
- Batch size: 4
- Seed: 42

Evaluates cold-reloaded checkpoints on test.jsonl using free-running greedy decoding.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from experiments.batch103_106_slot_weighted_sft.slot_weighted_training import (
    BASE_CKPT, TOKENIZER_DIR, ChakrMicro, ModelConfig, load_tokenizer_artifacts, compute_file_sha256
)
from experiments.batch115_118_tool_gating_curriculum.metrics import (
    evaluate_single_prediction,
    compute_aggregate_metrics,
)

EXPERIMENT_DIR = Path("experiments/batch123_126_prompt_diverse_multidigit")
DATA_DIR = EXPERIMENT_DIR / "data"
CKPT_DIR = EXPERIMENT_DIR / "checkpoints"
REPORTS_DIR = EXPERIMENT_DIR / "reports"

COPY_CKPT = Path("experiments/batch107_110_operand_copy_benchmark/checkpoints/copy_task_sft_step150.pt")

def load_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def build_training_sequence(record: dict, tokenizer):
    prompt_tokens = tokenizer.encode(record["prompt"])
    full_tokens = tokenizer.encode(record["full_conversation"])
    input_ids = [0] + full_tokens + [1]
    target_ids = [-100] * (len(prompt_tokens) + 1) + full_tokens[len(prompt_tokens):] + [1]
    return input_ids, target_ids

def train_arm(
    train_records: list,
    tokenizer,
    condition_name: str,
    base_state_dict: dict,
    steps: int = 200,
    batch_size: int = 4,
    lr: float = 3e-4,
    seed: int = 42,
):
    print(f"\n{'='*70}")
    print(f"Training Arm: {condition_name} ({steps} steps)")
    print(f"{'='*70}")
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    
    prepared = [build_training_sequence(r, tokenizer) for r in train_records]
    
    cfg = ModelConfig(max_seq_len=512)
    model = ChakrMicro(cfg)
    model.load_state_dict(base_state_dict)
    
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    history = []
    model.train()
    for step in range(steps):
        batch_indices = rng.randint(0, len(prepared), size=batch_size)
        batch = [prepared[i] for i in batch_indices]
        
        max_len = max(len(inp) for inp, _ in batch)
        x_pad = torch.full((batch_size, max_len), 0, dtype=torch.long)
        y_pad = torch.full((batch_size, max_len), -100, dtype=torch.long)
        for b, (inp, tgt) in enumerate(batch):
            x_pad[b, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            y_pad[b, :len(tgt)] = torch.tensor(tgt, dtype=torch.long)
            
        x_in = x_pad[:, :-1]
        y_tgt = y_pad[:, 1:]
        
        opt.zero_grad()
        logits = model(x_in)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if (step + 1) % 25 == 0 or step == 0:
            print(f"Step {step+1:3d}/{steps} | Loss: {loss.item():.4f}")
            history.append({"step": step + 1, "loss": round(loss.item(), 4)})
            
    ckpt_path = CKPT_DIR / f"{condition_name}_step{steps}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "history": history,
        "condition_name": condition_name,
        "cfg": cfg.__dict__,
    }, ckpt_path)
    sha = compute_file_sha256(str(ckpt_path))
    print(f"Saved: {ckpt_path} (SHA-256: {sha})")
    return ckpt_path, sha, history

def evaluate_arm(ckpt_path: Path, condition_name: str, test_records: list, tokenizer):
    print(f"\n{'='*70}")
    print(f"Evaluating {condition_name} on {len(test_records)} test records")
    print(f"{'='*70}")
    
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ModelConfig(max_seq_len=512)
    model = ChakrMicro(cfg)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()
    
    eval_list = []
    for r in test_records:
        prompt_ids = [0] + tokenizer.encode(r["prompt"])
        gen_tokens = list(prompt_ids)
        with torch.no_grad():
            for _ in range(50):
                logits = model(torch.tensor([gen_tokens]))[0, -1]
                nxt = torch.argmax(logits).item()
                gen_tokens.append(nxt)
                if nxt == 1:
                    break
                    
        raw_gen = tokenizer.decode(gen_tokens[len(prompt_ids):])
        clean_gen = raw_gen.split("<|im_end|>")[0].strip()
        
        res = evaluate_single_prediction(clean_gen, r)
        res["id"] = r["id"]
        res["category"] = r.get("category", "")
        res["prompt_snippet"] = r["prompt"].replace("\n", " ")[:60]
        res["full_prediction"] = clean_gen
        eval_list.append(res)
        
    overall_agg = compute_aggregate_metrics(eval_list)
    overall_agg["condition_name"] = condition_name
    overall_agg["checkpoint_path"] = str(ckpt_path)
    overall_agg["checkpoint_sha256"] = compute_file_sha256(str(ckpt_path))
    
    # Subcategory breakdown
    categories = sorted(list({r.get("category", "") for r in test_records}))
    cat_breakdown = {}
    for cat in categories:
        cat_recs = [r for r in eval_list if r["category"] == cat]
        cat_breakdown[cat] = compute_aggregate_metrics(cat_recs)
        
    overall_agg["subcategory_breakdown"] = cat_breakdown
    overall_agg["per_example_details"] = eval_list
    return overall_agg

def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    train_records = load_jsonl(DATA_DIR / "train.jsonl")
    test_records = load_jsonl(DATA_DIR / "test.jsonl")
    
    # Load Initializations
    print("Loading canonical base state dict...")
    canon_data = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    canon_state = canon_data["model_state_dict"]
    
    print("Loading copy-pre-trained state dict...")
    copy_data = torch.load(COPY_CKPT, map_location="cpu", weights_only=True)
    copy_state = copy_data["model_state_dict"]
    
    # 1. Train Arm A: Canonical Base Initialization
    ckpt_a, sha_a, hist_a = train_arm(
        train_records, tokenizer, "multidigit_armA_canonical_base", canon_state, steps=200
    )
    
    # 2. Train Arm B: Copy-Task Pre-Trained Initialization
    ckpt_b, sha_b, hist_b = train_arm(
        train_records, tokenizer, "multidigit_armB_copy_init", copy_state, steps=200
    )
    
    # Evaluate Arm A
    res_a = evaluate_arm(ckpt_a, "multidigit_armA_canonical_base", test_records, tokenizer)
    
    # Evaluate Arm B
    res_b = evaluate_arm(ckpt_b, "multidigit_armB_copy_init", test_records, tokenizer)
    
    combined_report = {
        "benchmark": "batch123_126_prompt_diverse_multidigit",
        "steps": 200,
        "batch_size": 4,
        "lr": 3e-4,
        "seed": 42,
        "armA_canonical_base": res_a,
        "armB_copy_init": res_b,
    }
    
    out_file = REPORTS_DIR / "multidigit_experiment_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, indent=2)
        
    print(f"\nExperiment complete. Saved report to {out_file}")

if __name__ == "__main__":
    main()
