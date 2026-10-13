"""
Batch 131–134: Training & Evaluation Harness for Relational Rule Induction (RRI-1).
Compares:
- Arm A: Canonical Base Initialization (Step 10,000 checkpoint)
- Arm B: Copy-Task Pre-Trained Initialization (Step 150 checkpoint)
Both models trained for 150 steps under AdamW (lr=3e-4) on 100 RRI records.
Evaluated via free-running greedy decoding on 45 strictly isolated test records.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from experiments.batch103_106_slot_weighted_sft.slot_weighted_training import (
    TOKENIZER_DIR, ChakrMicro, ModelConfig, load_tokenizer_artifacts, compute_file_sha256
)

EXP_DIR = Path("experiments/batch131_134_mission_alignment_audit")
DATA_DIR = EXP_DIR / "data"
CKPT_DIR = EXP_DIR / "checkpoints"
REPORTS_DIR = EXP_DIR / "reports"

CANONICAL_BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
COPY_PRETRAIN_CKPT = Path("experiments/batch107_110_operand_copy_benchmark/checkpoints/copy_task_sft_step150.pt")

def load_jsonl(p: Path) -> list:
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def build_seq(record: dict, tokenizer):
    prompt_tokens = tokenizer.encode(record["prompt"])
    full_tokens = tokenizer.encode(record["full_text"])
    input_ids = [0] + full_tokens + [1]
    target_ids = [-100] * (len(prompt_tokens) + 1) + full_tokens[len(prompt_tokens):] + [1]
    return input_ids, target_ids

def train_arm(train_records: list, tokenizer, name: str, base_state: dict, steps: int = 150, batch_size: int = 4, lr: float = 3e-4, seed: int = 42):
    print(f"\nTraining Arm: {name} ({steps} steps, batch {batch_size}, lr {lr})")
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    
    prepared = [build_seq(r, tokenizer) for r in train_records]
    
    cfg = ModelConfig(max_seq_len=512)
    model = ChakrMicro(cfg)
    model.load_state_dict(base_state)
    
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
            print(f"[{name}] Step {step+1:3d}/{steps} | Loss: {loss.item():.4f}")
            history.append({"step": step + 1, "loss": round(loss.item(), 4)})
            
    ckpt_path = CKPT_DIR / f"{name}_step{steps}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "history": history,
        "name": name,
        "cfg": cfg.__dict__,
    }, ckpt_path)
    sha = compute_file_sha256(str(ckpt_path))
    print(f"Saved: {ckpt_path} (SHA-256: {sha})")
    return ckpt_path, sha, history

def evaluate_arm(ckpt_path: Path, name: str, test_records: list, tokenizer):
    print(f"\nEvaluating: {name} on {len(test_records)} test examples")
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ModelConfig(max_seq_len=512)
    model = ChakrMicro(cfg)
    model.load_state_dict(ckpt_data["model_state_dict"])
    model.eval()
    
    per_example = []
    for r in test_records:
        prompt_ids = [0] + tokenizer.encode(r["prompt"])
        gen_tokens = list(prompt_ids)
        with torch.no_grad():
            for _ in range(10): # targets are short (1-2 tokens: single letter or UNKNOWN)
                logits = model(torch.tensor([gen_tokens]))[0, -1]
                nxt = torch.argmax(logits).item()
                gen_tokens.append(nxt)
                if nxt == 1:
                    break
                    
        raw_gen = tokenizer.decode(gen_tokens[len(prompt_ids):])
        clean_gen = raw_gen.split("<|im_end|>")[0].strip().split("\n")[0].strip()
        
        expected = r["target"].strip()
        exact = (clean_gen == expected)
        
        per_example.append({
            "id": r["id"],
            "category": r["category"],
            "is_routable": r["is_routable"],
            "prompt_snippet": r["prompt"].replace("\n", " ")[:60],
            "expected": expected,
            "prediction": clean_gen,
            "exact_match": exact,
        })
        
    # Aggregate statistics
    categories = sorted(list({p["category"] for p in per_example}))
    cat_summary = {}
    for cat in categories:
        recs = [p for p in per_example if p["category"] == cat]
        corr = sum(1 for p in recs if p["exact_match"])
        cat_summary[cat] = {
            "count": len(recs),
            "correct": corr,
            "accuracy": round(corr / len(recs), 4) if recs else 0.0,
        }
        
    total_eval = len(per_example)
    total_corr = sum(1 for p in per_example if p["exact_match"])
    
    summary = {
        "name": name,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": compute_file_sha256(str(ckpt_path)),
        "total_count": total_eval,
        "total_correct": total_corr,
        "overall_accuracy": round(total_corr / total_eval, 4),
        "category_summary": cat_summary,
        "per_example": per_example,
    }
    return summary

def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    train_recs = load_jsonl(DATA_DIR / "train.jsonl")
    test_recs = load_jsonl(DATA_DIR / "test.jsonl")
    
    # Arm A: Canonical Base Checkpoint
    print(f"Loading Canonical Base from {CANONICAL_BASE_CKPT}...")
    base_data = torch.load(CANONICAL_BASE_CKPT, map_location="cpu", weights_only=False)
    base_state = base_data.get("model_state_dict", base_data)
    
    ckpt_a, sha_a, hist_a = train_arm(train_recs, tokenizer, "rri_armA_canonical_base", base_state, steps=150, seed=42)
    
    # Arm B: Copy-Pre-Trained Checkpoint
    print(f"Loading Copy Pre-trained Base from {COPY_PRETRAIN_CKPT}...")
    copy_data = torch.load(COPY_PRETRAIN_CKPT, map_location="cpu", weights_only=True)
    copy_state = copy_data["model_state_dict"]
    
    ckpt_b, sha_b, hist_b = train_arm(train_recs, tokenizer, "rri_armB_copy_init", copy_state, steps=150, seed=42)
    
    # Evaluations
    res_a = evaluate_arm(ckpt_a, "rri_armA_canonical_base", test_recs, tokenizer)
    res_b = evaluate_arm(ckpt_b, "rri_armB_copy_init", test_recs, tokenizer)
    
    results = {
        "benchmark": "Batch 131-134 Relational Rule Induction (RRI-1)",
        "train_records": len(train_recs),
        "test_records": len(test_recs),
        "steps": 150,
        "batch_size": 4,
        "lr": 3e-4,
        "seed": 42,
        "armA_canonical_base": res_a,
        "armB_copy_init": res_b,
    }
    
    out_p = REPORTS_DIR / "rri_experiment_results.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRRI Experiment complete. Results written to {out_p}")

if __name__ == "__main__":
    main()
