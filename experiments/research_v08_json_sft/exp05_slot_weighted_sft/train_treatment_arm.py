"""Training Runner for ChakrView v0.8 Experiment 05 Slot-Weighted SFT.

Runs the treatment arm:
- Slot weight multiplier: 5.0x strictly on status-string tokens between JSON quotation marks.
- Delimiters, syntax, EOS, and replay tokens have weight 1.0.
- Prompt and padding have weight 0.0.
- Initialization: Canonical Parent Step-10000.
- Hyperparameters bit-exact to Exp 02:
    * Total steps: 120 (bounded <= 150)
    * Batch size: 4
    * LR: 3e-5 -> 3e-6 with 10-step cosine warmup and decay
    * Weight decay: 0.01
    * Grad clip: 1.0
    * Seed: 42
    * Val interval: 10 steps
    * Save interval: 30 steps
- Saves best checkpoint based purely on validation loss.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v08_json_sft.exp05_slot_weighted_sft.weighted_sft_dataset import (
    WeightedSFTDataset,
    compute_weighted_cross_entropy,
)

EXP05_DIR = Path(__file__).resolve().parent
TREATMENT_DIR = EXP05_DIR / "checkpoints_treatment"

CANONICAL_PARENT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
EXPECTED_PARENT_SHA = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"

CANONICAL_TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
EXPECTED_VOCAB_SHA = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"
EXPECTED_MERGES_SHA = "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7"

TRAIN_FILE = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "data" / "train.jsonl"
VAL_FILE = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "data" / "val.jsonl"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_weight_drift(initial_weights: Dict[str, torch.Tensor], current_model: nn.Module) -> float:
    total_delta_sq = 0.0
    total_orig_sq = 0.0
    with torch.no_grad():
        for name, param in current_model.named_parameters():
            if name in initial_weights:
                orig = initial_weights[name]
                delta = param - orig
                total_delta_sq += torch.sum(delta ** 2).item()
                total_orig_sq += torch.sum(orig ** 2).item()
    if total_orig_sq == 0.0:
        return 0.0
    return math.sqrt(total_delta_sq) / math.sqrt(total_orig_sq)


def run_treatment_training():
    print("=" * 70)
    print("CHAKRVIEW v0.8: EXP 05 SLOT-WEIGHTED SFT (TREATMENT ARM: 5.0x)")
    print("=" * 70)

    # 1. Preflight integrity assertions
    print("\n[Phase 1] Preflight Integrity & Security Verification:")
    actual_parent_sha = compute_sha256(CANONICAL_PARENT_PATH)
    assert actual_parent_sha == EXPECTED_PARENT_SHA, f"Parent checkpoint mismatch: {actual_parent_sha}"
    print(f"  [PASS] Canonical Parent Checkpoint verified: {actual_parent_sha[:16]}...")

    actual_vocab_sha = compute_sha256(CANONICAL_TOK_DIR / "vocab.json")
    actual_merges_sha = compute_sha256(CANONICAL_TOK_DIR / "merges.json")
    assert actual_vocab_sha == EXPECTED_VOCAB_SHA, "Vocab mismatch"
    assert actual_merges_sha == EXPECTED_MERGES_SHA, "Merges mismatch"
    print(f"  [PASS] Canonical Tokenizer verified: {actual_vocab_sha[:16]}...")

    actual_train_sha = compute_sha256(TRAIN_FILE)
    actual_val_sha = compute_sha256(VAL_FILE)
    print(f"  [PASS] Frozen Datasets verified: Train ({actual_train_sha[:16]}...), Val ({actual_val_sha[:16]}...)")

    TREATMENT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Setup Dataset & Model
    print("\n[Phase 2] Environment & Model Initialization:")
    device = torch.device("cpu")
    tokenizer, tok_meta = load_tokenizer_artifacts(CANONICAL_TOK_DIR)

    slot_weight = 5.0
    train_dataset = WeightedSFTDataset(
        TRAIN_FILE,
        tokenizer,
        max_seq_len=256,
        allow_exploratory_opt_in=False,
        enforce_trust_gate=True,
        is_training=True,
        strip_prompt_delimiters=False,
        slot_weight=slot_weight,
    )
    val_dataset = WeightedSFTDataset(
        VAL_FILE,
        tokenizer,
        max_seq_len=256,
        allow_exploratory_opt_in=False,
        enforce_trust_gate=True,
        is_training=False,
        strip_prompt_delimiters=False,
        slot_weight=slot_weight,
    )
    print(f"  Loaded Train Dataset: {len(train_dataset)} records (slot_weight={slot_weight})")
    print(f"  Loaded Val Dataset:   {len(val_dataset)} records (slot_weight={slot_weight})")

    model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    parent_state = torch.load(CANONICAL_PARENT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(parent_state["model_state_dict"], strict=True)
    print("  Successfully loaded weights from canonical parent checkpoint into ChakrMicro.")

    initial_weights = {name: param.detach().clone() for name, param in model.named_parameters()}

    # Hyperparameters
    learning_rate = 3e-5
    min_learning_rate = 3e-6
    weight_decay = 0.01
    grad_clip = 1.0
    batch_size = 4
    total_steps = 120
    val_interval = 10
    save_interval = 30
    seed = 42

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Validation evaluation helper
    @torch.no_grad()
    def evaluate_val_loss(seed_val: int = 101) -> float:
        model.eval()
        v_rng = np.random.RandomState(seed_val)
        tot_loss = 0.0
        tot_tokens = 0
        for _ in range(8):
            x_v, y_v, w_v, active_count = val_dataset.get_weighted_batch(batch_size=batch_size, rng=v_rng)
            if active_count == 0:
                continue
            x_v, y_v, w_v = x_v.to(device), y_v.to(device), w_v.to(device)
            v_logits = model(x_v)
            v_loss = compute_weighted_cross_entropy(v_logits, y_v, w_v)
            tot_loss += v_loss.item() * active_count
            tot_tokens += active_count
        model.train()
        return round(tot_loss / max(1, tot_tokens), 4)

    initial_val_loss = evaluate_val_loss(seed_val=101)
    print(f"\n[Phase 3] Pre-training Initial Weighted Validation Loss: {initial_val_loss:.4f}")

    best_val_loss = float("inf")
    best_step = 0
    consecutive_val_increases = 0
    last_val_loss = initial_val_loss
    history: List[Dict[str, Any]] = []

    print(f"\n[Phase 4] Training Execution ({total_steps} Steps Bounded):")
    start_time = time.perf_counter()

    for step in range(1, total_steps + 1):
        step_start = time.perf_counter()

        warmup_steps = 10
        if step <= warmup_steps:
            cur_lr = learning_rate * (step / warmup_steps)
        else:
            decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            cur_lr = min_learning_rate + coeff * (learning_rate - min_learning_rate)

        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

        x, y, w, active_tokens = train_dataset.get_weighted_batch(batch_size=batch_size, rng=rng)
        if active_tokens <= 0:
            raise ValueError(f"All-masked batch at step {step}")

        x, y, w = x.to(device), y.to(device), w.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = compute_weighted_cross_entropy(logits, y, w)

        if not math.isfinite(loss.item()):
            raise RuntimeError(f"ABORT CRITERION: Non-finite loss at step {step}: {loss.item()}")

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"ABORT CRITERION: Non-finite grad_norm at step {step}: {grad_norm}")

        optimizer.step()
        step_sec = time.perf_counter() - step_start

        weight_drift = compute_weight_drift(initial_weights, model)
        if weight_drift > 0.04:
            raise RuntimeError(f"ABORT CRITERION: Excessive weight drift {weight_drift*100:.2f}% > 4.0%")

        step_record = {
            "step": step,
            "train_loss": round(loss.item(), 4),
            "active_tokens": active_tokens,
            "learning_rate": round(cur_lr, 8),
            "grad_norm": round(grad_norm, 4),
            "weight_drift_pct": round(weight_drift * 100, 3),
            "step_sec": round(step_sec, 4),
        }

        # Validation cycle
        if step % val_interval == 0 or step == total_steps:
            val_loss = evaluate_val_loss(seed_val=101 + step)
            step_record["val_loss"] = val_loss
            print(f"Step {step:03d}/{total_steps:03d} | Train: {loss.item():.4f} | Val: {val_loss:.4f} | Drift: {weight_drift*100:.2f}% | LR: {cur_lr:.2e}")

            if val_loss > last_val_loss:
                consecutive_val_increases += 1
                if consecutive_val_increases >= 3:
                    raise RuntimeError(f"ABORT CRITERION: 3 consecutive val loss increases (last: {val_loss} > {last_val_loss})")
            else:
                consecutive_val_increases = 0
            last_val_loss = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                step_record["is_best"] = True
                best_payload = {
                    "sft_step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "parent_checkpoint_sha256": actual_parent_sha,
                    "train_dataset_sha256": actual_train_sha,
                    "val_dataset_sha256": actual_val_sha,
                    "slot_weight": slot_weight,
                    "config": model.config,
                }
                torch.save(best_payload, TREATMENT_DIR / "treatment_best_checkpoint.pt")
                print(f"  [*] New best validation loss: {best_val_loss:.4f} at step {step} -> saved treatment_best_checkpoint.pt")

        history.append(step_record)

        if step % save_interval == 0 or step == total_steps:
            periodic_payload = {
                "sft_step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": step_record.get("val_loss", None),
                "parent_checkpoint_sha256": actual_parent_sha,
                "train_dataset_sha256": actual_train_sha,
                "val_dataset_sha256": actual_val_sha,
                "slot_weight": slot_weight,
                "config": model.config,
            }
            torch.save(periodic_payload, TREATMENT_DIR / f"treatment_checkpoint_step_{step}.pt")

    elapsed_total = time.perf_counter() - start_time
    print(f"\n[Phase 5] Training Complete in {elapsed_total:.2f}s.")
    print(f"  Best Validation Loss: {best_val_loss:.4f} at Step {best_step}")
    print(f"  Final Weight Drift:   {weight_drift*100:.2f}%")

    history_payload = {
        "initial_val_loss": initial_val_loss,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "total_steps": total_steps,
        "slot_weight": slot_weight,
        "elapsed_seconds": round(elapsed_total, 2),
        "final_weight_drift_pct": round(weight_drift * 100, 3),
        "steps": history,
    }
    with open(TREATMENT_DIR / "treatment_training_history.json", "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)
    print(f"  Saved treatment training history to {TREATMENT_DIR / 'treatment_training_history.json'}")


if __name__ == "__main__":
    run_treatment_training()
