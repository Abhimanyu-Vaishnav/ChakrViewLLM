"""
Experiment 31 Runner: Capability Improvement & Generalization Validation
Audits 3 arms:
- Arm 0: Canonical ChakrMicro Zero-Shot (Frozen Baseline)
- Arm 1: Canonical ChakrMicro SFT (300 steps, cross-entropy on prompt-masked targets)
- Arm 2: ChakrPointerMicro (300 steps, pointer-generator copy attention)

Evaluates on 6 distinct leakage-resistant frozen test groups:
- Group A: Familiar tokens (in-distribution)
- Group B: Unseen token identities (out-of-distribution value generalization)
- Group C: Unseen slot permutations (positional invariance)
- Group D: Distractor complexity (retrieval selectivity)
- Group E: Counterfactual pairs (responsiveness to dynamic context)
- Group F: Negative control / Abstention (abstention when key missing)
"""

import os
import sys
import json
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.pointer_generator import ChakrPointerMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import build_causal_sequence

EXP_DIR = REPO_ROOT / "experiments" / "exp31_capability_generalization"
DATA_DIR = EXP_DIR / "data"
CKPT_DIR = EXP_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"

# Resource and training budget bounds
TOTAL_STEPS = 300
BATCH_SIZE = 4
LR_MAX = 5e-5
LR_MIN = 5e-6
SEED = 42

torch.set_num_threads(4)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_data(filename):
    path = DATA_DIR / filename
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def prepare_tensors(records, tok):
    data = []
    for r in records:
        p_toks = tok.encode(r["prompt"])
        ans_toks = tok.encode(r["answer"])
        inp, tgt = build_causal_sequence(p_toks, ans_toks, bos_id=0, eos_id=1, ignore_index=-100)
        data.append({
            "record": r,
            "inp": torch.tensor(inp, dtype=torch.long),
            "tgt": torch.tensor(tgt, dtype=torch.long),
            "prompt_len": len(p_toks) + 1  # includes BOS
        })
    return data

def collate_batch(batch_items, pad_id=2, ignore_index=-100):
    max_len = max(len(it["inp"]) for it in batch_items)
    B = len(batch_items)
    padded_x = torch.full((B, max_len), pad_id, dtype=torch.long)
    padded_y = torch.full((B, max_len), ignore_index, dtype=torch.long)
    prompt_lens = [it["prompt_len"] for it in batch_items]
    for i, it in enumerate(batch_items):
        l = len(it["inp"])
        padded_x[i, :l] = it["inp"]
        padded_y[i, :l] = it["tgt"]
    return padded_x, padded_y, prompt_lens

def evaluate_loss(model, tensor_data, is_pointer=False):
    model.eval()
    losses = []
    with torch.no_grad():
        for it in tensor_data:
            x = it["inp"].unsqueeze(0)
            y = it["tgt"].unsqueeze(0)
            p_len = it["prompt_len"]
            if is_pointer:
                _, _, _, probs = model(x, prompt_len=p_len)
                log_p = torch.log(probs + 1e-9)
                loss = F.nll_loss(log_p.view(-1, 4096), y.view(-1), ignore_index=-100)
            else:
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(logits.view(-1, 4096), y.view(-1), ignore_index=-100)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else 0.0

def train_variant(name, model, train_data, val_data, is_pointer=False):
    print(f"\n--- Training {name} ---")
    set_seed(SEED)
    rng = np.random.RandomState(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.01)

    init_val = evaluate_loss(model, val_data, is_pointer=is_pointer)
    print(f"Step 0: Initial Val Loss = {init_val:.4f}")

    t0 = time.perf_counter()
    train_logs = []

    for step in range(1, TOTAL_STEPS + 1):
        decay = (step - 1) / max(1, TOTAL_STEPS)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
        lr = LR_MIN + coeff * (LR_MAX - LR_MIN)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch_idxs = rng.choice(len(train_data), size=BATCH_SIZE, replace=False)
        batch_items = [train_data[i] for i in batch_idxs]
        x, y, p_lens = collate_batch(batch_items)

        optimizer.zero_grad()
        if is_pointer:
            max_p_len = max(p_lens)
            _, _, _, probs = model(x, prompt_len=max_p_len)
            log_p = torch.log(probs + 1e-9)
            loss = F.nll_loss(log_p.view(-1, 4096), y.view(-1), ignore_index=-100)
        else:
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits.view(-1, 4096), y.view(-1), ignore_index=-100)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == TOTAL_STEPS:
            val_loss = evaluate_loss(model, val_data, is_pointer=is_pointer)
            elapsed = time.perf_counter() - t0
            print(f"Step {step:3d}/{TOTAL_STEPS}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}, Elapsed = {elapsed:.1f}s")
            train_logs.append({
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss),
                "elapsed_s": float(elapsed)
            })

    # Save checkpoint
    ckpt_path = CKPT_DIR / f"{name.lower().replace(' ', '_')}_step300.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved {name} checkpoint to {ckpt_path}")
    return train_logs, ckpt_path

def evaluate_test_group(model, records, tok, is_pointer=False):
    model.eval()
    correct = 0
    total = len(records)
    predictions = []

    with torch.no_grad():
        for idx, r in enumerate(records):
            p_toks = tok.encode(r["prompt"])
            target = r["answer"].strip()
            x = torch.tensor([[0] + p_toks], dtype=torch.long)
            p_len = len(p_toks) + 1

            if is_pointer:
                _, _, p_gen, probs = model(x, prompt_len=p_len)
                pred_id = torch.argmax(probs[0, -1, :]).item()
                p_gen_val = float(p_gen[0, -1, 0].item())
            else:
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                pred_id = torch.argmax(logits[0, -1, :]).item()
                p_gen_val = 1.0

            pred_text = tok.decode([pred_id]).strip()
            is_match = (pred_text == target)
            if is_match:
                correct += 1

            predictions.append({
                "id": idx,
                "target": target,
                "pred_id": pred_id,
                "pred_text": pred_text,
                "correct": is_match,
                "p_gen": p_gen_val
            })

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0.0,
        "predictions": predictions
    }

def main():
    print("=" * 60)
    print("EXPERIMENT 31: CAPABILITY GENERALIZATION & REPRODUCIBILITY AUDIT")
    print("=" * 60)

    tok, _ = load_tokenizer_artifacts(VOCAB_DIR)

    train_records = load_data("train.jsonl")
    val_records = load_data("val.jsonl")
    train_tensors = prepare_tensors(train_records, tok)
    val_tensors = prepare_tensors(val_records, tok)

    test_splits = {
        "Group_A_Familiar": load_data("test_group_A_familiar.jsonl"),
        "Group_B_Unseen_Tokens": load_data("test_group_B_unseen_tokens.jsonl"),
        "Group_C_Permutations": load_data("test_group_C_permutations.jsonl"),
        "Group_D_Distractors": load_data("test_group_D_distractors.jsonl"),
        "Group_E_Counterfactuals": load_data("test_group_E_counterfactuals.jsonl"),
        "Group_F_Negative_Abstain": load_data("test_group_F_negative_abstain.jsonl"),
    }

    results = {}

    # ----------------------------------------------------
    # 1. Arm 0: Canonical Zero-Shot Baseline
    # ----------------------------------------------------
    print("\n--- Initializing Arm 0: Canonical Zero-Shot Baseline ---")
    cfg = ModelConfig()
    base_model = ChakrMicro(cfg)
    base_state = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
    if "model_state_dict" in base_state:
        base_state = base_state["model_state_dict"]
    base_model.load_state_dict(base_state)

    arm0_results = {}
    for group_name, records in test_splits.items():
        res = evaluate_test_group(base_model, records, tok, is_pointer=False)
        arm0_results[group_name] = {
            "correct": res["correct"],
            "total": res["total"],
            "accuracy": res["accuracy"],
            "sample_errors": [p for p in res["predictions"] if not p["correct"]][:3]
        }
        print(f"Arm 0 [{group_name}]: {res['correct']}/{res['total']} ({res['accuracy']*100:.1f}%)")
    results["Arm_0_Canonical_ZeroShot"] = arm0_results

    # ----------------------------------------------------
    # 2. Arm 1: Canonical SFT
    # ----------------------------------------------------
    print("\n--- Initializing Arm 1: Canonical SFT ---")
    sft_model = ChakrMicro(cfg)
    sft_model.load_state_dict(base_state)
    arm1_train_logs, arm1_ckpt = train_variant("Arm_1_Canonical_SFT", sft_model, train_tensors, val_tensors, is_pointer=False)

    arm1_results = {"train_logs": arm1_train_logs}
    for group_name, records in test_splits.items():
        res = evaluate_test_group(sft_model, records, tok, is_pointer=False)
        arm1_results[group_name] = {
            "correct": res["correct"],
            "total": res["total"],
            "accuracy": res["accuracy"],
            "sample_errors": [p for p in res["predictions"] if not p["correct"]][:3]
        }
        print(f"Arm 1 [{group_name}]: {res['correct']}/{res['total']} ({res['accuracy']*100:.1f}%)")
    results["Arm_1_Canonical_SFT"] = arm1_results

    # ----------------------------------------------------
    # 3. Arm 2: ChakrPointerMicro
    # ----------------------------------------------------
    print("\n--- Initializing Arm 2: ChakrPointerMicro ---")
    ptr_model = ChakrPointerMicro(cfg)
    # Load canonical base weights into ptr_model.core
    ptr_model.core.load_state_dict(base_state)
    arm2_train_logs, arm2_ckpt = train_variant("Arm_2_ChakrPointerMicro", ptr_model, train_tensors, val_tensors, is_pointer=True)

    arm2_results = {"train_logs": arm2_train_logs}
    for group_name, records in test_splits.items():
        res = evaluate_test_group(ptr_model, records, tok, is_pointer=True)
        arm2_results[group_name] = {
            "correct": res["correct"],
            "total": res["total"],
            "accuracy": res["accuracy"],
            "sample_errors": [p for p in res["predictions"] if not p["correct"]][:3]
        }
        print(f"Arm 2 [{group_name}]: {res['correct']}/{res['total']} ({res['accuracy']*100:.1f}%)")
    results["Arm_2_ChakrPointerMicro"] = arm2_results

    # Save full machine-readable results
    out_results_path = EXP_DIR / "RESULTS.json"
    with open(out_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_results_path}")

if __name__ == "__main__":
    main()
