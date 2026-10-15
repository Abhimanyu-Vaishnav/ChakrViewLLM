"""
Batches 604-609: Training Pipeline for Neural Span Pointer Candidate.
Location: experiments/batch581_620/train_span_candidate.py

Features:
1. Multi-Task Objective:
   L_total = L_intent + 1.0 * L_start + 1.0 * L_end + 0.5 * L_gate
2. Uses balanced dataset (train_span_balanced.jsonl, N=175) and validates against val_span_sft.jsonl.
3. Supports --pilot mode for sanity verification.
4. Evaluates on frozen held-out test suite (test_span_heldout.jsonl, N=59) measuring:
   - Intent accuracy
   - Exact token span accuracy
   - Exact character substring recovery accuracy
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.brain.span_pointer_model import ChakrSpanPointerModel, INTENT_TO_ID, ID_TO_INTENT
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

STARTING_CANDIDATE = REPO_ROOT / "experiments" / "batch541_580" / "checkpoints" / "candidate_step250_intent_understanding.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"

TRAIN_DATA = REPO_ROOT / "experiments" / "batch581_620" / "data" / "train_span_balanced.jsonl"
VAL_DATA = REPO_ROOT / "experiments" / "batch581_620" / "data" / "val_span_sft.jsonl"
TEST_DATA = REPO_ROOT / "experiments" / "batch581_620" / "data" / "test_span_heldout.jsonl"

OUT_CKPT_PATH = REPO_ROOT / "experiments" / "batch581_620" / "checkpoints" / "candidate_step250_span_pointer.pt"
REPORT_PATH = REPO_ROOT / "experiments" / "batch581_620" / "reports" / "TRAINING_AND_EVALUATION_RESULTS.json"


def encode_item(tokenizer: Any, item: Dict[str, Any]) -> Dict[str, Any]:
    prompt = item["prompt"]
    input_ids = [0] + tokenizer.encode(prompt)
    intent_id = INTENT_TO_ID.get(item["intent"], INTENT_TO_ID["unsupported"])

    has_span = item["span_text"] is not None
    st_tok = item["start_tok"] if has_span else 0
    end_tok = item["end_tok"] if has_span else 0

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "intent_id": torch.tensor(intent_id, dtype=torch.long),
        "has_span": torch.tensor(1.0 if has_span else 0.0, dtype=torch.float),
        "start_tok": torch.tensor(st_tok, dtype=torch.long),
        "end_tok": torch.tensor(end_tok, dtype=torch.long),
        "item": item,
    }


def evaluate_span_model(
    model: ChakrSpanPointerModel,
    tokenizer: Any,
    test_items: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total = len(test_items)
    intent_correct = 0
    exact_span_correct = 0
    exact_substring_correct = 0

    per_intent: Dict[str, Dict[str, int]] = {}
    details = []

    with torch.no_grad():
        for it in test_items:
            cat = it["intent"]
            if cat not in per_intent:
                per_intent[cat] = {"total": 0, "intent_correct": 0, "span_correct": 0, "exact_value": 0}
            per_intent[cat]["total"] += 1

            prompt = it["prompt"]
            input_ids = torch.tensor([[0] + tokenizer.encode(prompt)], dtype=torch.long, device=device)

            pred_intent, pred_span, conf = model.predict_span(input_ids)
            i_match = (pred_intent == it["intent"])
            if i_match:
                intent_correct += 1
                per_intent[cat]["intent_correct"] += 1

            # Evaluate span match
            s_match = False
            v_match = False
            extracted_text = None

            if it["span_text"] is None:
                # Query should have NO span
                if pred_span is None:
                    s_match = True
                    v_match = True
            else:
                if pred_span is not None:
                    pst, pend = pred_span
                    exp_st, exp_end = it["start_tok"], it["end_tok"]
                    s_match = (pst == exp_st and pend == exp_end)

                    # Extract substring from decoded token slice
                    tok_slice = input_ids[0, pst:pend].tolist()
                    extracted_text = tokenizer.decode(tok_slice, errors="replace").strip()
                    # Normalization for exact value comparison
                    exp_val = it["span_text"].strip()
                    v_match = (extracted_text == exp_val or extracted_text.replace(" ", "") == exp_val.replace(" ", ""))

            if s_match:
                exact_span_correct += 1
                per_intent[cat]["span_correct"] += 1
            if v_match:
                exact_substring_correct += 1
                per_intent[cat]["exact_value"] += 1

            details.append({
                "id": it["id"],
                "prompt": prompt,
                "expected_intent": it["intent"],
                "predicted_intent": pred_intent,
                "intent_correct": i_match,
                "expected_span": (it["start_tok"], it["end_tok"]) if it["span_text"] else None,
                "predicted_span": pred_span,
                "expected_value": it["span_text"],
                "extracted_value": extracted_text,
                "exact_value_match": v_match,
                "confidence": round(conf, 4),
            })

    return {
        "total": total,
        "intent_accuracy": intent_correct / max(total, 1),
        "span_accuracy": exact_span_correct / max(total, 1),
        "exact_value_accuracy": exact_substring_correct / max(total, 1),
        "per_intent": per_intent,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run small 30-step pilot")
    parser.add_argument("--steps", type=int, default=250, help="Total training steps")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate")
    args = parser.parse_args()

    total_steps = 30 if args.pilot else args.steps
    print(f"=== BATCHES 604-609: SPAN POINTER MODEL TRAINING (Mode: {'PILOT' if args.pilot else 'FULL'}) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # 1. Load starting base model
    ck_start = torch.load(STARTING_CANDIDATE, map_location="cpu", weights_only=False)
    base_model = ChakrMicro(ck_start["config"])
    base_model.load_state_dict(ck_start["model_state_dict"])

    # 2. Initialize ChakrSpanPointerModel
    span_model = ChakrSpanPointerModel(base_model=base_model).to(device)

    def load_jsonl(p: Path) -> List[Dict[str, Any]]:
        with open(p, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    train_raw = load_jsonl(TRAIN_DATA)
    val_raw = load_jsonl(VAL_DATA)
    test_raw = load_jsonl(TEST_DATA)

    train_encoded = [encode_item(tokenizer, it) for it in train_raw]
    val_encoded = [encode_item(tokenizer, it) for it in val_raw]

    # Pre-training baseline evaluation
    print("\n--- Evaluating Pre-Training Model Baseline ---")
    pre_eval = evaluate_span_model(span_model, tokenizer, test_raw, device=device)
    print(f"Pre-Train Intent Accuracy: {pre_eval['intent_accuracy']*100:.1f}% ({int(pre_eval['intent_accuracy']*pre_eval['total'])}/{pre_eval['total']})")
    print(f"Pre-Train Value Accuracy:  {pre_eval['exact_value_accuracy']*100:.1f}% ({int(pre_eval['exact_value_accuracy']*pre_eval['total'])}/{pre_eval['total']})")

    # Optimizer
    optimizer = optim.AdamW(span_model.parameters(), lr=args.lr, weight_decay=0.01)
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    torch.manual_seed(42)
    random.seed(42)

    print(f"\n--- Training Span Pointer Model ({total_steps} steps) ---")
    t0 = time.perf_counter()
    best_val_loss = float("inf")
    best_state = None

    for step in range(total_steps):
        span_model.train()
        optimizer.zero_grad()

        sample = train_encoded[step % len(train_encoded)]
        x = sample["input_ids"].unsqueeze(0).to(device)
        tgt_intent = sample["intent_id"].unsqueeze(0).to(device)
        tgt_has_span = sample["has_span"].unsqueeze(0).unsqueeze(1).to(device)

        out = span_model(x)

        # Loss 1: Intent loss
        l_intent = ce_loss(out["intent_logits"], tgt_intent)
        # Loss 2: Has-span gate loss
        l_gate = bce_loss(out["has_span_logits"], tgt_has_span)

        # Loss 3 & 4: Start and End span loss (if span present)
        if sample["has_span"].item() > 0.5:
            st_target = sample["start_tok"].unsqueeze(0).to(device)
            end_target = sample["end_tok"].unsqueeze(0).to(device)
            l_start = ce_loss(out["start_logits"], st_target)
            l_end = ce_loss(out["end_logits"], end_target)
            loss = l_intent + 0.5 * l_gate + 1.0 * l_start + 1.0 * l_end
        else:
            loss = l_intent + 0.5 * l_gate

        loss.backward()
        torch.nn.utils.clip_grad_norm_(span_model.parameters(), max_norm=1.0)
        optimizer.step()

        val_freq = 10 if args.pilot else 25
        if (step + 1) % val_freq == 0:
            span_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v_samp in val_encoded:
                    vx = v_samp["input_ids"].unsqueeze(0).to(device)
                    vi = v_samp["intent_id"].unsqueeze(0).to(device)
                    vout = span_model(vx)
                    val_loss += ce_loss(vout["intent_logits"], vi).item()
            val_loss /= max(len(val_encoded), 1)
            print(f"Step {step+1:3d}/{total_steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(span_model.state_dict())

    elapsed = time.perf_counter() - t0
    print(f"\nTraining completed in {elapsed:.2f}s. Best Val Loss: {best_val_loss:.4f}")

    if best_state is not None:
        span_model.load_state_dict(best_state)

    if args.pilot:
        print("\nPilot run complete. Exiting without overwriting checkpoint.")
        return

    # Save Checkpoint
    OUT_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "name": "candidate_step250_span_pointer",
        "steps": total_steps,
        "base_config": base_model.config,
        "model_state_dict": span_model.state_dict(),
        "elapsed_seconds": round(elapsed, 2),
        "best_val_loss": round(best_val_loss, 4),
    }, OUT_CKPT_PATH)

    with open(OUT_CKPT_PATH, "rb") as f:
        cand_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Saved Checkpoint: {OUT_CKPT_PATH} (SHA-256: {cand_hash})")

    # Post-training evaluation
    print("\n--- Post-Training Evaluation on Held-Out Test Set (N=59) ---")
    post_eval = evaluate_span_model(span_model, tokenizer, test_raw, device=device)
    print(f"Trained Intent Accuracy:      {post_eval['intent_accuracy']*100:.1f}% ({int(post_eval['intent_accuracy']*post_eval['total'])}/{post_eval['total']})")
    print(f"Trained Token Span Accuracy:  {post_eval['span_accuracy']*100:.1f}% ({int(post_eval['span_accuracy']*post_eval['total'])}/{post_eval['total']})")
    print(f"Trained Exact Value Accuracy: {post_eval['exact_value_accuracy']*100:.1f}% ({int(post_eval['exact_value_accuracy']*post_eval['total'])}/{post_eval['total']})")

    print("\n--- Per-Intent Breakdown ---")
    for cat, s in post_eval["per_intent"].items():
        tot = s["total"]
        print(f"  {cat:22s} | N={tot:2d} | Intent Acc: {s['intent_correct']/tot*100:5.1f}% | Exact Value: {s['exact_value']/tot*100:5.1f}%")

    report = {
        "pre_train": pre_eval,
        "post_train": post_eval,
        "checkpoint": str(OUT_CKPT_PATH),
        "sha256": cand_hash,
        "steps": total_steps,
        "elapsed_seconds": round(elapsed, 2),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
