"""
train_exp07.py
ChakrView Experiment 07 — Small-Set Copying Overfit Training Script

Trains ChakrMicro strictly on tiny_train_16.jsonl for up to 500 optimizer steps.
Evaluates every 25 steps:
- Train loss
- Teacher-forced first-token accuracy on train set (excluding EOS)
- Free-generation exact match on train set (greedy argmax decoding)

Saves checkpoints to:
experiments/research_v08_json_sft/exp07_copy_overfit/checkpoints/
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
CHECKPOINT_DIR = EXP_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DATA_PATH = DATA_DIR / "tiny_train_16.jsonl"
CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_on_train(model: ChakrMicro, tokenizer, train_records, device="cpu"):
    """
    Evaluates:
    1. Teacher-forced first-token accuracy (target value token only, NOT EOS)
    2. Free-generation exact match (generating up to 10 tokens, argmax)
    """
    model.eval()
    first_token_hits = 0
    gen_exact_matches = 0
    total = len(train_records)

    with torch.no_grad():
        for rec in train_records:
            prompt = rec["prompt"]
            target_val = rec["response"]
            
            p_toks = tokenizer.encode(prompt)
            r_toks = tokenizer.encode(target_val)
            first_target_id = r_toks[0]
            
            # Form full causal sequence with BOS=0 and EOS=1
            full_ids = [0] + p_toks + r_toks + [1]
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
            
            # Forward pass for teacher forcing
            logits = model(input_tensor)
            # Logit predicting the first token of response is at index len(p_toks) (since 0 is at index 0)
            # input_tensor[0, 0] = 0
            # input_tensor[0, 1..len(p_toks)] = prompt tokens
            # Position len(p_toks) contains the last prompt token (":")
            # Its output logit predicts the first response token!
            first_resp_logit = logits[0, len(p_toks)]
            pred_first_token_id = int(torch.argmax(first_resp_logit).item())
            
            if pred_first_token_id == first_target_id:
                first_token_hits += 1
                
            # Free generation: feed [0] + p_toks and autoregressively sample with argmax
            curr_input = [0] + list(p_toks)
            gen_tokens = []
            for _ in range(10):
                inp = torch.tensor([curr_input], dtype=torch.long, device=device)
                gen_logits = model(inp)
                nxt = int(torch.argmax(gen_logits[0, -1, :]).item())
                if nxt == 1:  # EOS token id is 1
                    break
                gen_tokens.append(nxt)
                curr_input.append(nxt)
                
            try:
                gen_text = tokenizer.decode(gen_tokens, errors="replace")
            except Exception:
                gen_text = tokenizer.decode(gen_tokens)
                
            if gen_text.strip() == target_val.strip():
                gen_exact_matches += 1

    model.train()
    return {
        "first_token_acc": (first_token_hits / total) * 100.0,
        "first_token_hits": first_token_hits,
        "free_gen_em": (gen_exact_matches / total) * 100.0,
        "free_gen_hits": gen_exact_matches,
        "total": total
    }


def main():
    print("=" * 70)
    print("CHAKRVIEW EXPERIMENT 07: SMALL-SET COPYING OVERFIT TRAINING")
    print("=" * 70)

    # 1. Verify parent checkpoint integrity before loading
    parent_hash_before = compute_sha256(CANONICAL_CKPT)
    print(f"Parent Checkpoint Hash Before: {parent_hash_before}")
    assert parent_hash_before == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

    # 2. Load tokenizer
    tokenizer, tok_meta = load_tokenizer_artifacts(TOKENIZER_DIR)
    print(f"Loaded tokenizer with vocab size {len(tokenizer.vocab)}.")

    # 3. Load dataset
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(train_records)} training records from {TRAIN_DATA_PATH}")

    # 4. Prepare pre-tokenized tensors for batching
    prepared_items = []
    for rec in train_records:
        p_toks = tokenizer.encode(rec["prompt"])
        r_toks = tokenizer.encode(rec["response"])
        # full sequence: [0] + prompt + response + [1]
        seq = [0] + p_toks + r_toks + [1]
        # response-only target masking: -100 for BOS and prompt tokens
        labels = [-100] * (len(p_toks) + 1) + r_toks + [1]
        prepared_items.append((
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        ))

    # 5. Initialize model from canonical parent checkpoint in isolated memory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")
    model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    parent_state = torch.load(CANONICAL_CKPT, map_location=device, weights_only=False)
    if "model_state_dict" in parent_state:
        model.load_state_dict(parent_state["model_state_dict"])
    else:
        model.load_state_dict(parent_state)
    model.train()
    print("Successfully loaded isolated ChakrMicro instance from Step-10000.")

    # 6. Training Hyperparameters
    MAX_STEPS = 500
    EVAL_INTERVAL = 25
    BATCH_SIZE = 4  # 4 sequences per step; 16 examples = 4 steps per full pass
    LR = 5e-4
    WEIGHT_DECAY = 0.01

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Pre-training baseline evaluation (Step 0)
    baseline_eval = evaluate_on_train(model, tokenizer, train_records, device=device)
    print(f"\n[Baseline Step 0] First-Token Acc: {baseline_eval['first_token_acc']:.2f}% ({baseline_eval['first_token_hits']}/16) | "
          f"Free Gen EM: {baseline_eval['free_gen_em']:.2f}% ({baseline_eval['free_gen_hits']}/16)")

    history = [{
        "step": 0,
        "train_loss": None,
        "first_token_acc": baseline_eval["first_token_acc"],
        "free_gen_em": baseline_eval["free_gen_em"]
    }]

    best_free_em = baseline_eval["free_gen_em"]
    best_step = 0
    best_state_dict = copy.deepcopy(model.state_dict())

    # 7. Training loop
    item_idx = 0
    step_loss_accum = 0.0

    print(f"\nStarting optimization loop (budget: {MAX_STEPS} steps, batch_size: {BATCH_SIZE}, lr: {LR})...")
    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad()
        
        # Accumulate gradients across batch
        batch_loss = 0.0
        for b in range(BATCH_SIZE):
            input_ids, targets = prepared_items[item_idx]
            item_idx = (item_idx + 1) % len(prepared_items)
            
            input_ids = input_ids.unsqueeze(0).to(device)
            targets = targets.unsqueeze(0).to(device)
            
            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) / BATCH_SIZE
            loss.backward()
            batch_loss += loss.item() * BATCH_SIZE

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step_loss_accum += batch_loss

        # Evaluation interval
        if step % EVAL_INTERVAL == 0:
            avg_loss = step_loss_accum / EVAL_INTERVAL
            step_loss_accum = 0.0
            
            metrics = evaluate_on_train(model, tokenizer, train_records, device=device)
            history.append({
                "step": step,
                "train_loss": round(avg_loss, 4),
                "first_token_acc": metrics["first_token_acc"],
                "free_gen_em": metrics["free_gen_em"]
            })
            
            print(f"Step {step:3d}/{MAX_STEPS} | Loss: {avg_loss:.4f} | "
                  f"First-Token Acc: {metrics['first_token_acc']:5.1f}% ({metrics['first_token_hits']}/16) | "
                  f"Free Gen EM: {metrics['free_gen_em']:5.1f}% ({metrics['free_gen_hits']}/16)")
            
            # Track best checkpoint based on free generation exact match
            if metrics["free_gen_em"] > best_free_em or (metrics["free_gen_em"] == best_free_em and avg_loss < (history[-2].get("train_loss") or 999)):
                best_free_em = metrics["free_gen_em"]
                best_step = step
                best_state_dict = copy.deepcopy(model.state_dict())
                
            # Early stopping check if 100% exact match achieved and maintained with low loss
            if metrics["free_gen_em"] == 100.0 and metrics["first_token_acc"] == 100.0 and avg_loss < 0.05:
                print(f"\n[EARLY STOP] Perfect copying and near-zero loss ({avg_loss:.4f}) achieved at Step {step}!")
                break

    # 8. Save best and final checkpoints
    best_ckpt_path = CHECKPOINT_DIR / "exp07_best_checkpoint.pt"
    final_ckpt_path = CHECKPOINT_DIR / "exp07_final_checkpoint.pt"

    torch.save({
        "step": best_step,
        "model_state_dict": best_state_dict,
        "best_free_em": best_free_em,
        "parent_hash": parent_hash_before
    }, best_ckpt_path)
    print(f"\nSaved best checkpoint (Step {best_step}, EM: {best_free_em}%) to {best_ckpt_path}")

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "final_loss": avg_loss,
        "parent_hash": parent_hash_before
    }, final_ckpt_path)
    print(f"Saved final checkpoint (Step {step}) to {final_ckpt_path}")

    # 9. Save training log
    log_path = EXP_DIR / "training_history.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "best_step": best_step,
            "best_free_em": best_free_em,
            "best_checkpoint_sha256": compute_sha256(best_ckpt_path),
            "final_checkpoint_sha256": compute_sha256(final_ckpt_path)
        }, f, indent=2)
    print(f"Training history saved to {log_path}")

    # 10. Post-training integrity check
    parent_hash_after = compute_sha256(CANONICAL_CKPT)
    assert parent_hash_after == parent_hash_before, "CRITICAL ERROR: Parent checkpoint was modified!"
    print(f"[PASS] Post-training integrity verified. Canonical parent hash unchanged: {parent_hash_after}")


if __name__ == "__main__":
    main()
