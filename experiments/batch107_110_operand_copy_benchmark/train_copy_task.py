import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple

from experiments.batch103_106_slot_weighted_sft.slot_weighted_training import (
    BASE_CKPT, TOKENIZER_DIR, ChakrMicro, ModelConfig, load_tokenizer_artifacts, compute_file_sha256
)
from experiments.batch107_110_operand_copy_benchmark.eval_canonical_baseline import (
    DATA_DIR, load_records, evaluate_copy_model
)

OUTPUT_DIR = Path("experiments/batch107_110_operand_copy_benchmark")

def build_training_sequence(record: Dict, tokenizer) -> Tuple[List[int], List[int]]:
    prompt_tokens = tokenizer.encode(record["prompt"])
    full_tokens = tokenizer.encode(record["full_conversation"])
    input_ids = [0] + full_tokens + [1]
    # Mask prompt positions with -100
    target_ids = [-100] * (len(prompt_tokens) + 1) + full_tokens[len(prompt_tokens):] + [1]
    return input_ids, target_ids

def train_copy_condition(
    records: List[Dict],
    tokenizer,
    condition_name: str,
    steps: int = 150,
    batch_size: int = 4,
    lr: float = 3e-4,
    seed: int = 42,
) -> Tuple[ChakrMicro, List[Dict], Path, str]:
    print(f"\n============================================================")
    print(f"Training Copy Model: {condition_name} ({steps} steps)")
    print(f"============================================================")
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    
    # Pre-encode all training records
    prepared = [build_training_sequence(r, tokenizer) for r in records]
    
    cfg = ModelConfig(max_seq_len=512)
    model = ChakrMicro(cfg)
    base_state = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(base_state["model_state_dict"] if "model_state_dict" in base_state else base_state)
    
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    history = []
    model.train()
    for step in range(steps):
        batch_indices = rng.randint(0, len(prepared), size=batch_size)
        batch = [prepared[i] for i in batch_indices]
        
        # Pad to max length in batch
        max_len = max(len(inp) for inp, _ in batch)
        x_pad = torch.full((batch_size, max_len), 0, dtype=torch.long)
        y_pad = torch.full((batch_size, max_len), -100, dtype=torch.long)
        for b, (inp, tgt) in enumerate(batch):
            x_pad[b, :len(inp)] = torch.tensor(inp, dtype=torch.long)
            y_pad[b, :len(tgt)] = torch.tensor(tgt, dtype=torch.long)
            
        # Target alignment: input x[:, :-1], target y[:, 1:]
        x_in = x_pad[:, :-1]
        y_tgt = y_pad[:, 1:]
        
        opt.zero_grad()
        logits = model(x_in) # (B, T, V)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if (step + 1) % 25 == 0 or step == 0:
            print(f"Step {step+1:3d}/{steps} | Loss: {loss.item():.4f}")
            history.append({"step": step + 1, "loss": round(loss.item(), 4)})
            
    ckpt_path = OUTPUT_DIR / "checkpoints" / f"{condition_name}_step{steps}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "history": history,
        "condition_name": condition_name,
    }, ckpt_path)
    sha = compute_file_sha256(str(ckpt_path))
    print(f"Saved checkpoint: {ckpt_path} (SHA-256: {sha})")
    return model, history, ckpt_path, sha

def main():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    splits = {
        "train": load_records(DATA_DIR / "train.jsonl"),
        "val": load_records(DATA_DIR / "val.jsonl"),
        "test": load_records(DATA_DIR / "test.jsonl"),
    }
    
    # 1. Train copy task model
    model, hist, ckpt_path, sha = train_copy_condition(
        splits["train"], tok, "copy_task_sft", steps=150, batch_size=4, lr=3e-4, seed=42
    )
    
    # 2. Reload model from checkpoint for cold reload evaluation
    print("\n--- Cold Reload Verification for Copy Model ---")
    reload_cfg = ModelConfig(max_seq_len=512)
    reload_model = ChakrMicro(reload_cfg)
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    reload_model.load_state_dict(st["model_state_dict"])
    reload_model.eval()
    
    results = {
        "checkpoint": str(ckpt_path),
        "sha256": sha,
        "history": hist,
        "eval_splits": {},
    }
    for s_name, recs in splits.items():
        res = evaluate_copy_model(reload_model, tok, recs)
        results["eval_splits"][s_name] = res
        print(f"[{s_name:5s}] A_acc={res['operand_a_accuracy']*100:.1f}%, B_acc={res['operand_b_accuracy']*100:.1f}%, Pair_acc={res['exact_pair_accuracy']*100:.1f}%, Collapse={res['collapse_rate']*100:.1f}%")
        print(f"        Mean P(A)={res['mean_target_prob_a']*100:.2f}% (Rank {res['mean_rank_a']}), Mean P(B)={res['mean_target_prob_b']*100:.2f}% (Rank {res['mean_rank_b']})")
        print(f"        Sample out: {repr(res['samples'][0])}")
        
    out_rep = OUTPUT_DIR / "reports" / "copy_learning_results.json"
    with open(out_rep, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Copy learning results saved to {out_rep}")

if __name__ == "__main__":
    main()
