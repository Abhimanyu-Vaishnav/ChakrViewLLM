"""Diagnostic probe script for Experiment 01 SFT Checkpoint and Data.

Investigates:
1. Training vs validation token-level loss.
2. Logit distribution at position 107 (the integer code token).
3. Why `{"code": 3}` was emitted invariant across test prompts.
4. Alignment verification between input sequences and targets.
"""

from __future__ import annotations

import json
from pathlib import Path
import torch
import torch.nn as nn

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.sft_engine import SFTDataset

def run_diagnostics():
    device = torch.device("cpu")
    tok_dir = Path("data/experiments/vocab_4096")
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    exp01_ckpt_path = Path("experiments/research_v08_json_sft/checkpoints/sft_best_checkpoint.pt")
    parent_ckpt_path = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")

    exp01_state = torch.load(exp01_ckpt_path, map_location=device, weights_only=False)
    parent_state = torch.load(parent_ckpt_path, map_location=device, weights_only=False)

    model_exp01 = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    model_exp01.load_state_dict(exp01_state["model_state_dict"])
    model_exp01.eval()

    model_parent = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    model_parent.load_state_dict(parent_state["model_state_dict"])
    model_parent.eval()

    # Inspect the training dataset
    train_path = Path("experiments/research_v08_json_sft/data/train.jsonl")
    with open(train_path, "r", encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f]

    two_key_train = [r for r in train_records if r.get("category") == "two_key_json_extraction"]
    replay_train = [r for r in train_records if r.get("category") == "base_continuation_replay"]

    print(f"Total train records: {len(train_records)}")
    print(f"Two-key JSON train records: {len(two_key_train)}")
    print(f"Replay train records: {len(replay_train)}")

    # Check whether all two-key train records have both keys
    has_status_and_code = all(
        '"status"' in r["response"] and '"code"' in r["response"] for r in two_key_train
    )
    print(f"All two-key train records contain 'status' and 'code': {has_status_and_code}")

    # Inspect token supervision for record 0
    ds = SFTDataset(train_path, tokenizer, max_seq_len=256, strip_prompt_delimiters=False)
    r0 = ds.records[0]
    inp, tgt = ds.encode_example(r0["prompt"], r0["response"], append_eos=True)

    # Check prompt vs response tokens
    prompt_tokens_masked = sum(1 for t in tgt if t == -100)
    supervised_tokens = sum(1 for t in tgt if t != -100)
    print(f"Record 0 total tokens: {len(inp)}")
    print(f"Record 0 masked prompt tokens: {prompt_tokens_masked}")
    print(f"Record 0 supervised response tokens: {supervised_tokens}")

    # Forward pass on record 0
    with torch.no_grad():
        x = torch.tensor([inp])
        logits = model_exp01(x)[0]
    
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    losses = []
    for i in range(prompt_tokens_masked, len(inp)):
        target_token = tgt[i]
        pos_loss = loss_fn(logits[i:i+1, :], torch.tensor([target_token])).item()
        pred_token = int(torch.argmax(logits[i, :]).item())
        losses.append((i, inp[i], target_token, pred_token, pos_loss))

    print("\nPer-token supervision and loss for Record 0 (nominal, 214):")
    for pos, in_t, tg_t, pr_t, l_val in losses:
        in_s = repr(tokenizer.decode([in_t]))
        tg_s = repr(tokenizer.decode([tg_t])) if tg_t != 1 else "<EOS>"
        pr_s = repr(tokenizer.decode([pr_t])) if pr_t != 1 else "<EOS>"
        print(f"  Pos {pos:3d} (given {in_s:12s}): target={tg_s:12s} pred={pr_s:12s} loss={l_val:.4f}")

    # Test what the model predicts if forced to start with `{"status": "`
    forced_prefix = r0["prompt"] + '{"status": "'
    forced_toks = [0] + tokenizer.encode(forced_prefix)
    with torch.no_grad():
        f_logits = model_exp01(torch.tensor([forced_toks]))[0, -1, :]
    top5_vals, top5_inds = torch.topk(f_logits, 5)
    print("\nWhen forced to predict after '{\"status\": \"':")
    for val, ind in zip(top5_vals.tolist(), top5_inds.tolist()):
        tok_str = tokenizer.decode([ind])
        print(f"  {ind:5d} ({repr(tok_str):12s}): logit={val:.2f}")

if __name__ == "__main__":
    run_diagnostics()
