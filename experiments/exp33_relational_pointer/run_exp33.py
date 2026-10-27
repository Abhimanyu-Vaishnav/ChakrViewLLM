"""
Experiment 33 Runner: Relational Pointer Circuit, Key-to-Value Binding & Abstention.
Compares:
- Arm 0: Canonical Zero-Shot Baseline (Frozen Step 10,000 checkpoint)
- Arm 1: Standard Pointer Baseline (ChakrPointerMicro)
- Arm 2: Relational Pointer Circuit (ChakrRelationalPointerMicro)

Evaluates on 350 frozen test instances across Groups A through F.
Records:
- EM accuracy per group
- Abstention precision, recall, and F1
- False abstention rate on retrieval instances
- Distractor selection rate vs hallucination rate
- Detailed key-binding error breakdown
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
from experiments.exp33_relational_pointer.relational_pointer_model import ChakrRelationalPointerMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import build_causal_sequence

EXP_DIR = REPO_ROOT / "experiments" / "exp33_relational_pointer"
DATA_DIR = EXP_DIR / "data"
CKPT_DIR = EXP_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"

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
            "prompt_len": len(p_toks) + 1
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

def evaluate_loss(model, tensor_data, is_pointer=True):
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

def train_model(name, model, train_tensors, val_tensors):
    print(f"\n--- Training {name} ---")
    set_seed(SEED)
    rng = np.random.RandomState(SEED)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=0.01)

    init_val = evaluate_loss(model, val_tensors, is_pointer=True)
    print(f"Step 0: Initial Val Loss = {init_val:.4f}")

    t0 = time.perf_counter()
    train_logs = []

    for step in range(1, TOTAL_STEPS + 1):
        decay = (step - 1) / max(1, TOTAL_STEPS)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay))
        lr = LR_MIN + coeff * (LR_MAX - LR_MIN)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch_idxs = rng.choice(len(train_tensors), size=BATCH_SIZE, replace=False)
        batch_items = [train_tensors[i] for i in batch_idxs]
        x, y, p_lens = collate_batch(batch_items)

        optimizer.zero_grad()
        max_p_len = max(p_lens)
        _, _, _, probs = model(x, prompt_len=max_p_len)
        log_p = torch.log(probs + 1e-9)
        loss = F.nll_loss(log_p.view(-1, 4096), y.view(-1), ignore_index=-100)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == TOTAL_STEPS:
            val_loss = evaluate_loss(model, val_tensors, is_pointer=True)
            elapsed = time.perf_counter() - t0
            print(f"Step {step:3d}/{TOTAL_STEPS}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}, Elapsed = {elapsed:.1f}s")
            train_logs.append({
                "step": step,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss),
                "elapsed_s": float(elapsed)
            })

    ckpt_path = CKPT_DIR / f"{name.lower().replace(' ', '_')}_step300.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved {name} checkpoint to {ckpt_path}")
    return train_logs, ckpt_path

def evaluate_test_group(model, records, tok, is_pointer=True):
    model.eval()
    total = len(records)
    correct = 0
    abstain_count = 0
    distractor_errors = 0
    hallucination_errors = 0
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
            is_abstain = (pred_text == "?")

            if is_match:
                correct += 1
            if is_abstain:
                abstain_count += 1

            error_cat = "CORRECT"
            context_vals = [v for k, v in r["bindings"]]
            if not is_match:
                if target == "?":
                    error_cat = "MISSED_ABSTENTION"
                elif is_abstain:
                    error_cat = "FALSE_ABSTENTION"
                elif pred_text in context_vals:
                    error_cat = "DISTRACTOR_SELECTION"
                    distractor_errors += 1
                else:
                    error_cat = "HALLUCINATION_OR_BIAS"
                    hallucination_errors += 1

            predictions.append({
                "id": idx,
                "target": target,
                "pred_id": pred_id,
                "pred_text": pred_text,
                "correct": is_match,
                "p_gen": p_gen_val,
                "error_category": error_cat
            })

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total > 0 else 0.0,
        "abstain_count": abstain_count,
        "distractor_errors": distractor_errors,
        "hallucination_errors": hallucination_errors,
        "predictions": predictions
    }

def main():
    print("=" * 65)
    print("EXPERIMENT 33: RELATIONAL POINTER CIRCUIT & DISTRACTOR RESOLUTION")
    print("=" * 65)

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
    cfg = ModelConfig()
    base_state = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
    if "model_state_dict" in base_state:
        base_state = base_state["model_state_dict"]

    # ----------------------------------------------------
    # 1. Arm 0: Canonical Zero-Shot Baseline
    # ----------------------------------------------------
    print("\n--- Arm 0: Canonical Zero-Shot Baseline ---")
    base_model = ChakrMicro(cfg)
    base_model.load_state_dict(base_state)
    arm0_results = {}
    for g_name, records in test_splits.items():
        res = evaluate_test_group(base_model, records, tok, is_pointer=False)
        arm0_results[g_name] = {
            "correct": res["correct"],
            "total": res["total"],
            "accuracy": res["accuracy"],
            "abstain_count": res["abstain_count"],
            "distractor_errors": res["distractor_errors"],
            "hallucination_errors": res["hallucination_errors"],
            "sample_errors": [p for p in res["predictions"] if not p["correct"]][:3]
        }
        print(f"Arm 0 [{g_name}]: {res['correct']}/{res['total']} ({res['accuracy']*100:.1f}%) | Abstains: {res['abstain_count']}")
    results["Arm_0_Canonical_ZeroShot"] = arm0_results

    # ----------------------------------------------------
    # 2. Arm 1: Standard Pointer Baseline (ChakrPointerMicro)
    # ----------------------------------------------------
    print("\n--- Arm 1: Standard Pointer Baseline ---")
    arm1_model = ChakrPointerMicro(cfg)
    arm1_model.core.load_state_dict(base_state)
    arm1_train_logs, arm1_ckpt = train_model("Arm_1_Standard_Pointer", arm1_model, train_tensors, val_tensors)

    arm1_results = {"train_logs": arm1_train_logs}
    for g_name, records in test_splits.items():
        res = evaluate_test_group(arm1_model, records, tok, is_pointer=True)
        arm1_results[g_name] = {
            "correct": res["correct"],
            "total": res["total"],
            "accuracy": res["accuracy"],
            "abstain_count": res["abstain_count"],
            "distractor_errors": res["distractor_errors"],
            "hallucination_errors": res["hallucination_errors"],
            "sample_errors": [p for p in res["predictions"] if not p["correct"]][:3]
        }
        print(f"Arm 1 [{g_name}]: {res['correct']}/{res['total']} ({res['accuracy']*100:.1f}%) | Distractors: {res['distractor_errors']} | Abstains: {res['abstain_count']}")
    results["Arm_1_Standard_Pointer"] = arm1_results

    # ----------------------------------------------------
    # 3. Arm 2: Relational Pointer Circuit (ChakrRelationalPointerMicro)
    # ----------------------------------------------------
    print("\n--- Arm 2: Relational Pointer Circuit ---")
    arm2_model = ChakrRelationalPointerMicro(cfg)
    arm2_model.core.load_state_dict(base_state)
    arm2_train_logs, arm2_ckpt = train_model("Arm_2_Relational_Pointer", arm2_model, train_tensors, val_tensors)

    arm2_results = {"train_logs": arm2_train_logs}
    for g_name, records in test_splits.items():
        res = evaluate_test_group(arm2_model, records, tok, is_pointer=True)
        arm2_results[g_name] = {
            "correct": res["correct"],
            "total": res["total"],
            "accuracy": res["accuracy"],
            "abstain_count": res["abstain_count"],
            "distractor_errors": res["distractor_errors"],
            "hallucination_errors": res["hallucination_errors"],
            "sample_errors": [p for p in res["predictions"] if not p["correct"]][:3]
        }
        print(f"Arm 2 [{g_name}]: {res['correct']}/{res['total']} ({res['accuracy']*100:.1f}%) | Distractors: {res['distractor_errors']} | Abstains: {res['abstain_count']}")
    results["Arm_2_Relational_Pointer"] = arm2_results

    # Summary calculations for Arm 2 Abstention
    f_total = 50
    f_arm2_tp = arm2_results["Group_F_Negative_Abstain"]["correct"]
    pos_eval_total = 300
    pos_abstains_arm2 = sum(arm2_results[g]["abstain_count"] for g in ["Group_A_Familiar", "Group_B_Unseen_Tokens", "Group_C_Permutations", "Group_D_Distractors", "Group_E_Counterfactuals"])

    precision = f_arm2_tp / (f_arm2_tp + pos_abstains_arm2) if (f_arm2_tp + pos_abstains_arm2) > 0 else 0.0
    recall = f_arm2_tp / f_total if f_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    abstention_summary = {
        "true_positives": f_arm2_tp,
        "false_negatives": f_total - f_arm2_tp,
        "false_positives": pos_abstains_arm2,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_abstention_rate": pos_abstains_arm2 / pos_eval_total
    }
    results["Arm_2_Abstention_Summary"] = abstention_summary
    print("\n--- Arm 2 Relational Abstention Summary ---")
    print(f"Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1: {f1:.4f}, False Abstention Rate: {pos_abstains_arm2}/{pos_eval_total} ({pos_abstains_arm2/pos_eval_total*100:.2f}%)")

    out_results_path = EXP_DIR / "RESULTS.json"
    with open(out_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_results_path}")

if __name__ == "__main__":
    main()
