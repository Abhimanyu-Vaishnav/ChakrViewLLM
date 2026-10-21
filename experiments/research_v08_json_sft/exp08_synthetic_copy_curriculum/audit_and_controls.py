"""
audit_and_controls.py
ChakrView Experiment 08 — Pre-Training Leakage, Controls, and Baseline Audit

Verifies:
1. Canonical parent checkpoint & tokenizer hash integrity.
2. Leakage audit:
   - Zero prompt-response pair overlap between Train and Val/TestA/TestB.
   - Test A targets are strict subset of Train targets.
   - Test B targets are strictly disjoint from Train targets.
3. Constant-output baseline evaluation (majority token / generic token prediction).
4. Causal alignment and response-only supervision assertions.
5. Finite forward loss and finite backward gradients on ChakrMicro.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"

EXPECTED_PARENT_SHA = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
EXPECTED_VOCAB_SHA = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def run_audit():
    print("=" * 70)
    print("EXP 08: LEAKAGE, SHORTCUTS, CONTROLS & CAUSAL ALIGNMENT AUDIT")
    print("=" * 70)

    # 1. Canonical Artifact Hashes
    parent_sha = compute_sha256(CANONICAL_CKPT)
    assert parent_sha == EXPECTED_PARENT_SHA, f"Parent hash mismatch: {parent_sha}"
    print(f"  [PASS] Canonical parent hash verified: {parent_sha[:16]}...")

    vocab_sha = compute_sha256(TOKENIZER_DIR / "vocab.json")
    assert vocab_sha == EXPECTED_VOCAB_SHA, f"Vocab hash mismatch: {vocab_sha}"
    print(f"  [PASS] Tokenizer vocab hash verified: {vocab_sha[:16]}...")

    # 2. Dataset Leakage & Disjointness Audit
    with open(DATA_DIR / "train.jsonl", "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f if line.strip()]
    with open(DATA_DIR / "val.jsonl", "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f if line.strip()]
    with open(DATA_DIR / "test_a_familiar_tokens.jsonl", "r", encoding="utf-8") as f:
        test_a_data = [json.loads(line) for line in f if line.strip()]
    with open(DATA_DIR / "test_b_held_out_tokens.jsonl", "r", encoding="utf-8") as f:
        test_b_data = [json.loads(line) for line in f if line.strip()]

    train_prompts = set(d["prompt"] for d in train_data)
    val_prompts = set(d["prompt"] for d in val_data)
    test_a_prompts = set(d["prompt"] for d in test_a_data)
    test_b_prompts = set(d["prompt"] for d in test_b_data)

    assert train_prompts.isdisjoint(val_prompts), "Train and Val prompts overlap!"
    assert train_prompts.isdisjoint(test_a_prompts), "Train and Test A prompts overlap!"
    assert train_prompts.isdisjoint(test_b_prompts), "Train and Test B prompts overlap!"
    print(f"  [PASS] Zero exact prompt overlap between Train and all eval splits.")

    train_target_ids = set(d["target_token_id"] for d in train_data)
    test_a_target_ids = set(d["target_token_id"] for d in test_a_data)
    test_b_target_ids = set(d["target_token_id"] for d in test_b_data)

    assert test_a_target_ids.issubset(train_target_ids), "Test A targets not in train!"
    assert train_target_ids.isdisjoint(test_b_target_ids), "Test B targets overlap with train!"
    print(f"  [PASS] Token partition verified: Test A targets are strict subset of Train, Test B targets are strictly disjoint.")

    # 3. Constant-Output Baseline
    # Find most common target token in Train
    train_counts = collections.Counter(d["target_token_id"] for d in train_data)
    top_tok_id, top_count = train_counts.most_common(1)[0]
    baseline_acc_train = (top_count / len(train_data)) * 100.0
    
    test_a_hits = sum(1 for d in test_a_data if d["target_token_id"] == top_tok_id)
    baseline_acc_test_a = (test_a_hits / len(test_a_data)) * 100.0
    
    test_b_hits = sum(1 for d in test_b_data if d["target_token_id"] == top_tok_id)
    baseline_acc_test_b = (test_b_hits / len(test_b_data)) * 100.0

    print(f"\n[Constant-Output Baseline (Predicting Most Frequent Token ID {top_tok_id})]:")
    print(f"  Train Baseline Exact Match:  {baseline_acc_train:.2f}%")
    print(f"  Test A Baseline Exact Match: {baseline_acc_test_a:.2f}%")
    print(f"  Test B Baseline Exact Match: {baseline_acc_test_b:.2f}%")

    # 4. Causal Alignment and Loss Test
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    sample = train_data[0]
    p_toks = tokenizer.encode(sample["prompt"])
    r_toks = tokenizer.encode(sample["target_value"])
    full_seq = [0] + p_toks + r_toks + [1]
    labels = [-100] * (len(p_toks) + 1) + r_toks + [1]

    assert len(full_seq) == len(labels)
    # Target token should be at index len(p_toks) + 1
    target_pos = len(p_toks) + 1
    assert labels[target_pos] == sample["target_token_id"]
    # The logit that predicts labels[target_pos] is from position len(p_toks)
    print(f"  [PASS] Causal alignment: input position {len(p_toks)} strictly predicts target token {sample['target_token_id']} at position {target_pos}.")

    # 5. Model Forward Loss and Backward Gradients
    model = ChakrMicro(ModelConfig(max_seq_len=512))
    state = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.train()

    inp = torch.tensor([full_seq], dtype=torch.long)
    tgt = torch.tensor([labels], dtype=torch.long)
    logits = model(inp)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = tgt[..., 1:].contiguous()
    loss = nn.CrossEntropyLoss(ignore_index=-100)(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss.backward()

    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN/inf!"
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Grad None for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN grad for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf grad for {name}"
    print(f"  [PASS] Finite forward loss ({loss.item():.4f}) and finite backward gradients on ChakrMicro.")

    # Save audit report
    audit_report = {
        "parent_sha256": parent_sha,
        "vocab_sha256": vocab_sha,
        "leakage_audit": {
            "prompt_overlap_train_val": len(train_prompts.intersection(val_prompts)),
            "prompt_overlap_train_test_a": len(train_prompts.intersection(test_a_prompts)),
            "prompt_overlap_train_test_b": len(train_prompts.intersection(test_b_prompts)),
            "test_a_targets_subset_of_train": test_a_target_ids.issubset(train_target_ids),
            "test_b_targets_disjoint_from_train": train_target_ids.isdisjoint(test_b_target_ids)
        },
        "constant_output_baseline": {
            "top_token_id": top_tok_id,
            "train_acc": baseline_acc_train,
            "test_a_acc": baseline_acc_test_a,
            "test_b_acc": baseline_acc_test_b
        }
    }
    with open(EXP_DIR / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)
    print(f"\nAudit complete and saved to {EXP_DIR / 'audit_report.json'}.")

if __name__ == "__main__":
    run_audit()
