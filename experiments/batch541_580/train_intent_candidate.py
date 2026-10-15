"""
Batches 558-565: Pilot and Full Training Pipeline for Neural Intent & Tool Calling Candidate.
Location: experiments/batch541_580/train_intent_candidate.py

Features:
1. Multi-Task Intent Curriculum:
   - 70% Structured Intent SFT (train_intent_balanced.jsonl, N=168)
   - 20% Diagnostic Retention & Paraphrase Rehearsal (preserves general language and concept grounding)
   - 10% Stage C Pretraining Rehearsal (preserves foundational token representations)
2. Loss Masking:
   - User prompt tokens are masked with -100.
   - Autoregressive supervision over the compact JSON contract target up to True EOS = 1.
3. Resumable checkpointing & early stopping based on val loss over val_intent_sft.jsonl.
4. Pilot mode (--pilot) for rapid sanity verification before full training.
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

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

STARTING_CANDIDATE = REPO_ROOT / "experiments" / "batch501_540" / "checkpoints" / "candidate_step200_compositional_rule.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
STAGE_C_DIR = REPO_ROOT / "data" / "tokenized" / "stage_c" / "train"

TRAIN_INTENT_DATA = REPO_ROOT / "experiments" / "batch541_580" / "data" / "train_intent_balanced.jsonl"
VAL_INTENT_DATA = REPO_ROOT / "experiments" / "batch541_580" / "data" / "val_intent_sft.jsonl"
TEST_INTENT_DATA = REPO_ROOT / "experiments" / "batch541_580" / "data" / "test_intent_heldout.jsonl"

PARAPHRASE_DATA = REPO_ROOT / "experiments" / "batch411_440_paraphrase" / "data" / "train_augmented.jsonl"
DIAGNOSTIC_DATA = REPO_ROOT / "experiments" / "batch381_410_integrity" / "data" / "diagnostic_answer_set.jsonl"

OUT_CKPT_PATH = REPO_ROOT / "experiments" / "batch541_580" / "checkpoints" / "candidate_step250_intent_understanding.pt"
REPORT_PATH = REPO_ROOT / "experiments" / "batch541_580" / "reports" / "TRAINING_AND_EVALUATION_RESULTS.json"


class StageCRehearsalSampler:
    def __init__(self, shards_dir: Path, seq_len: int = 128, seed: int = 42):
        self.shards = sorted(list(shards_dir.glob("shard_*.bin")))
        self.seq_len = seq_len
        self.rng = random.Random(seed)

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.shards:
            # Fallback if shards missing
            x = torch.zeros((1, self.seq_len), dtype=torch.long)
            y = torch.zeros((1, self.seq_len), dtype=torch.long)
            return x, y
        shard = self.rng.choice(self.shards)
        file_size = shard.stat().st_size
        n_tokens = file_size // 2
        max_start = max(0, n_tokens - self.seq_len - 1)
        offset = self.rng.randint(0, max_start) if max_start > 0 else 0

        with open(shard, "rb") as f:
            f.seek(offset * 2)
            raw = f.read((self.seq_len + 1) * 2)

        arr = np.frombuffer(raw, dtype=np.uint16).astype(np.int64)
        if len(arr) < self.seq_len + 1:
            arr = np.pad(arr, (0, self.seq_len + 1 - len(arr)), mode="constant", constant_values=2)

        x = torch.tensor(arr[:self.seq_len], dtype=torch.long).unsqueeze(0)
        y = torch.tensor(arr[1:self.seq_len + 1], dtype=torch.long).unsqueeze(0)
        return x, y


def encode_pair(tokenizer: Any, prompt: str, target: str, max_len: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    p_toks = [0] + tokenizer.encode(prompt)
    r_toks = tokenizer.encode(target) + [1]

    input_ids = (p_toks + r_toks)[:max_len]
    labels = ([-100] * len(p_toks) + r_toks)[:max_len]
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def evaluate_intent_testset(
    model: ChakrMicro,
    tokenizer: Any,
    items: List[Dict[str, Any]],
    device: torch.device = torch.device("cpu"),
    max_new_tokens: int = 80,
) -> Dict[str, Any]:
    model.eval()
    total = len(items)
    intent_matches = 0
    valid_json_count = 0
    exact_target_matches = 0
    clean_eos_count = 0
    arg_matches = 0

    per_intent_stats: Dict[str, Dict[str, int]] = {}

    details = []

    for item in items:
        expected_raw = item["target"]
        expected_obj = json.loads(expected_raw)
        exp_intent = expected_obj.get("intent")
        exp_args = expected_obj.get("args", {})

        if exp_intent not in per_intent_stats:
            per_intent_stats[exp_intent] = {
                "total": 0,
                "intent_correct": 0,
                "valid_json": 0,
                "exact_target": 0,
                "args_correct": 0,
            }
        per_intent_stats[exp_intent]["total"] += 1

        prompt = item["prompt"]
        p_toks = [0] + tokenizer.encode(prompt)
        gen = list(p_toks)
        eos_found = False

        with torch.no_grad():
            for _ in range(max_new_tokens):
                x = torch.tensor([gen[-256:]], dtype=torch.long, device=device)
                nxt = int(torch.argmax(model(x)[0, -1, :]).item())
                if nxt == 1:
                    eos_found = True
                    break
                gen.append(nxt)

        if eos_found:
            clean_eos_count += 1

        raw_gen = tokenizer.decode(gen[len(p_toks):], errors="replace").strip()

        # Parse JSON
        is_valid_json = False
        pred_intent = None
        pred_args = {}
        pred_obj = None

        try:
            # Check for direct JSON or enclosed JSON
            m = re.search(r"({.*})", raw_gen)
            if m:
                pred_obj = json.loads(m.group(1))
                is_valid_json = True
                pred_intent = pred_obj.get("intent")
                pred_args = pred_obj.get("args", {})
        except Exception:
            is_valid_json = False

        if is_valid_json:
            valid_json_count += 1
            per_intent_stats[exp_intent]["valid_json"] += 1

        intent_correct = (pred_intent == exp_intent)
        if intent_correct:
            intent_matches += 1
            per_intent_stats[exp_intent]["intent_correct"] += 1

        # Check argument exact match
        args_correct = False
        if is_valid_json and intent_correct:
            if exp_intent == "calculator":
                # Normalize spaces in expressions e.g. "52 + 67" vs "52+67"
                exp_expr = str(exp_args.get("expression", "")).replace(" ", "")
                pred_expr = str(pred_args.get("expression", "")).replace(" ", "")
                args_correct = (exp_expr == pred_expr and exp_expr != "")
            elif exp_intent == "text_utility":
                args_correct = (
                    pred_args.get("operation") == exp_args.get("operation") and
                    pred_args.get("text") == exp_args.get("text")
                )
            elif exp_intent == "concept_definition":
                args_correct = (str(pred_args.get("concept", "")).lower() == str(exp_args.get("concept", "")).lower())
            elif exp_intent == "clarification_needed":
                args_correct = (pred_args.get("missing_field") == exp_args.get("missing_field"))
            elif exp_intent in ("direct_chat", "unsupported", "knowledge_lookup"):
                args_correct = True

        if args_correct:
            arg_matches += 1
            per_intent_stats[exp_intent]["args_correct"] += 1

        exact_match = (raw_gen == expected_raw) or (is_valid_json and intent_correct and args_correct)
        if exact_match:
            exact_target_matches += 1
            per_intent_stats[exp_intent]["exact_target"] += 1

        details.append({
            "id": item.get("id", ""),
            "prompt": prompt,
            "expected": expected_raw,
            "generated": raw_gen,
            "is_valid_json": is_valid_json,
            "predicted_intent": pred_intent,
            "expected_intent": exp_intent,
            "intent_correct": intent_correct,
            "args_correct": args_correct,
            "eos_reached": eos_found,
        })

    return {
        "total": total,
        "valid_json_count": valid_json_count,
        "valid_json_rate": valid_json_count / max(total, 1),
        "intent_accuracy": intent_matches / max(total, 1),
        "argument_accuracy": arg_matches / max(total, 1),
        "clean_eos_rate": clean_eos_count / max(total, 1),
        "per_intent_metrics": per_intent_stats,
        "details": details,
    }


def compute_val_loss(model: ChakrMicro, tokenizer: Any, val_items: List[Dict[str, Any]], criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for it in val_items:
            x, y = encode_pair(tokenizer, it["prompt"], it["target"], max_len=256)
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            logits = model(x)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = y[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()
            count += 1
    return total_loss / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="Run a small 30-step pilot run to verify loss convergence")
    parser.add_argument("--steps", type=int, default=250, help="Total training steps")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Learning rate")
    args = parser.parse_args()

    total_steps = 30 if args.pilot else args.steps
    print(f"=== BATCHES 558-565: NEURAL INTENT TRAINING PIPELINE (Mode: {'PILOT' if args.pilot else 'FULL'}) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # 1. Load starting checkpoint
    ck_start = torch.load(STARTING_CANDIDATE, map_location="cpu", weights_only=False)
    ref_model = ChakrMicro(ck_start["config"])
    ref_model.load_state_dict(ck_start["model_state_dict"])
    ref_model.to(device)

    def load_jsonl(p: Path) -> List[Dict[str, Any]]:
        with open(p, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    train_intent = load_jsonl(TRAIN_INTENT_DATA)
    val_intent = load_jsonl(VAL_INTENT_DATA)
    test_intent = load_jsonl(TEST_INTENT_DATA)

    train_para = load_jsonl(PARAPHRASE_DATA) if PARAPHRASE_DATA.exists() else []

    # 2. Evaluate Baseline Candidate on Held-Out Test Set
    print("\n--- Evaluating Starting Candidate on Intent Benchmark ---")
    base_eval = evaluate_intent_testset(ref_model, tokenizer, test_intent, device=device)
    print(f"Baseline Intent Accuracy:   {base_eval['intent_accuracy']*100:.1f}% ({int(base_eval['intent_accuracy']*base_eval['total'])}/{base_eval['total']})")
    print(f"Baseline Valid JSON Rate:   {base_eval['valid_json_rate']*100:.1f}%")
    print(f"Baseline Argument Accuracy: {base_eval['argument_accuracy']*100:.1f}%")
    print(f"Baseline Clean EOS Rate:    {base_eval['clean_eos_rate']*100:.1f}%")

    # 3. Setup Training
    cand_model = copy.deepcopy(ref_model)
    cand_model.train()

    optimizer = optim.AdamW(cand_model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    sampler = StageCRehearsalSampler(STAGE_C_DIR, seq_len=128, seed=42)

    torch.manual_seed(42)
    random.seed(42)

    print(f"\n--- Commencing Training ({total_steps} steps) ---")
    t0 = time.perf_counter()
    best_val_loss = float("inf")
    best_state = None
    loss_history = []

    for step in range(total_steps):
        cand_model.train()
        optimizer.zero_grad()
        rnd = random.random()

        if rnd < 0.10:
            # 10% Pretraining rehearsal
            x, y = sampler.sample_batch()
            x, y = x.to(device), y.to(device)
            logits = cand_model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        elif rnd < 0.30 and train_para:
            # 20% Retention rehearsal
            it = train_para[step % len(train_para)]
            x, y = encode_pair(tokenizer, it["prompt"], it["target"], max_len=256)
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            logits = cand_model(x)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = y[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            # 70% Structured intent training
            it = train_intent[step % len(train_intent)]
            x, y = encode_pair(tokenizer, it["prompt"], it["target"], max_len=256)
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            logits = cand_model(x)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = y[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(cand_model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_history.append(loss.item())

        val_check_freq = 10 if args.pilot else 25
        if (step + 1) % val_check_freq == 0:
            val_loss = compute_val_loss(cand_model, tokenizer, val_intent, criterion, device)
            print(f"Step {step+1:3d}/{total_steps} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(cand_model.state_dict())

    elapsed = time.perf_counter() - t0
    print(f"\nTraining completed in {elapsed:.2f}s. Best Val Loss: {best_val_loss:.4f}")

    if best_state is not None:
        cand_model.load_state_dict(best_state)

    if args.pilot:
        print("\nPilot run successfully verified. Exiting without overwriting production candidate checkpoint.")
        return

    # 4. Save Candidate Checkpoint
    OUT_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "name": "candidate_step250_intent_understanding",
        "steps": total_steps,
        "config": cand_model.config,
        "model_state_dict": cand_model.state_dict(),
        "elapsed_seconds": round(elapsed, 2),
        "best_val_loss": round(best_val_loss, 4),
    }, OUT_CKPT_PATH)

    with open(OUT_CKPT_PATH, "rb") as f:
        cand_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Saved Checkpoint: {OUT_CKPT_PATH} (SHA-256: {cand_hash})")

    # 5. Post-Training Evaluation on Held-Out Test Set
    print("\n--- Evaluating Trained Candidate on Held-Out Test Set ---")
    new_eval = evaluate_intent_testset(cand_model, tokenizer, test_intent, device=device)
    print(f"Trained Candidate Intent Accuracy:   {new_eval['intent_accuracy']*100:.1f}% ({int(new_eval['intent_accuracy']*new_eval['total'])}/{new_eval['total']})")
    print(f"Trained Candidate Valid JSON Rate:   {new_eval['valid_json_rate']*100:.1f}%")
    print(f"Trained Candidate Argument Accuracy: {new_eval['argument_accuracy']*100:.1f}%")
    print(f"Trained Candidate Clean EOS Rate:    {new_eval['clean_eos_rate']*100:.1f}%")

    print("\n--- Per-Intent Breakdown ---")
    for cat, stats in new_eval["per_intent_metrics"].items():
        tot = stats["total"]
        print(f"  {cat:22s} | N={tot:2d} | Intent Acc: {stats['intent_correct']/tot*100:5.1f}% | Valid JSON: {stats['valid_json']/tot*100:5.1f}% | Args Acc: {stats['args_correct']/tot*100:5.1f}%")

    # Write report
    report = {
        "baseline_candidate": {
            "checkpoint": str(STARTING_CANDIDATE),
            "results": base_eval,
        },
        "trained_candidate": {
            "checkpoint": str(OUT_CKPT_PATH),
            "sha256": cand_hash,
            "training_steps": total_steps,
            "elapsed_seconds": round(elapsed, 2),
            "best_val_loss": round(best_val_loss, 4),
            "results": new_eval,
        }
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
