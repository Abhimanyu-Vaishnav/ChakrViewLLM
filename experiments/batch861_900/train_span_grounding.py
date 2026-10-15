"""
WP 878–883: Train and compare span-grounding candidates.
Trains ChakrSpanGroundingModel on annotated span data with joint loss:
L_total = L_vocab + alpha * (L_start + L_end) + beta * L_type + gamma * L_conf
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.span_grounding import ChakrSpanGroundingModel, SPAN_TYPE_TO_ID
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
STARTING_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TRAIN_DATA = REPO_ROOT / "experiments" / "batch861_900" / "data" / "train_span_grounding.jsonl"
VAL_DATA = REPO_ROOT / "experiments" / "batch861_900" / "data" / "val_span_grounding.jsonl"
OUT_CKPT_DIR = REPO_ROOT / "experiments" / "batch861_900" / "checkpoints"
REPORT_DIR = REPO_ROOT / "experiments" / "batch861_900" / "reports"


def train_span_model(
    seed: int = 42,
    steps: int = 250,
    ablation: Optional[str] = None,  # e.g., 'no_type', 'oracle_upper_bound'
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, _ = load_tokenizer_artifacts(str(TOKENIZER_DIR))

    cfg = ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, max_seq_len=512)
    model = ChakrSpanGroundingModel(cfg, max_slots=4)

    # Load canonical step 10,000 weights
    raw_ckpt = torch.load(STARTING_CKPT, map_location=device, weights_only=False)
    state = raw_ckpt.get("model", raw_ckpt.get("model_state", raw_ckpt))
    clean_state = {k.replace("module.", ""): v for k, v in state.items()}
    model.core.load_state_dict(clean_state, strict=False)
    model.to(device)

    with open(TRAIN_DATA, "r", encoding="utf-8") as f:
        train_items = [json.loads(line) for line in f]
    with open(VAL_DATA, "r", encoding="utf-8") as f:
        val_items = [json.loads(line) for line in f]

    optimizer = optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.01)

    model.train()
    t0 = time.perf_counter()
    loss_history = []

    for step in range(1, steps + 1):
        it = random.choice(train_items)
        prompt = it["prompt"]
        target_dsl = it["target_dsl"]
        slots = it.get("slots", [])

        p_tokens = [0] + tokenizer.encode(prompt)
        t_tokens = tokenizer.encode(target_dsl)

        full_input = p_tokens + t_tokens
        full_target = [-100] * len(p_tokens) + t_tokens

        input_ids = torch.tensor(full_input[:-1], dtype=torch.long).unsqueeze(0).to(device)
        labels = torch.tensor(full_target[1:], dtype=torch.long).unsqueeze(0).to(device)
        prompt_len = len(p_tokens)

        optimizer.zero_grad()

        # Pick one slot to supervise this step
        if slots:
            slot_info = random.choice(slots)
            slot_idx = slot_info["slot_idx"]
            target_s = slot_info["start"]
            target_e = slot_info["end"]
            type_id = SPAN_TYPE_TO_ID.get(slot_info["type"], 0)
        else:
            slot_idx = 0
            target_s = -1
            target_e = -1
            type_id = 0

        vocab_logits, start_logits, end_logits, type_logits, conf = model(
            input_ids, prompt_len=prompt_len, slot_idx=slot_idx
        )

        # 1. LM Loss
        loss_lm = F.cross_entropy(vocab_logits.view(-1, 4096), labels.view(-1), ignore_index=-100)

        # 2. Span Boundary Loss
        if target_s >= 0 and target_e >= 0 and target_s < prompt_len and target_e < prompt_len:
            target_s_t = torch.tensor([target_s], device=device)
            target_e_t = torch.tensor([target_e], device=device)
            loss_span = F.cross_entropy(start_logits, target_s_t) + F.cross_entropy(end_logits, target_e_t)
            target_conf = torch.tensor([[1.0]], device=device)
        else:
            # Ungrounded / missing argument slot: uniformly suppress or target 0
            loss_span = torch.tensor(0.0, device=device)
            target_conf = torch.tensor([[0.0]], device=device)

        # 3. Type Classification Loss
        if ablation == "no_type":
            loss_type = torch.tensor(0.0, device=device)
        else:
            loss_type = F.cross_entropy(type_logits, torch.tensor([type_id], device=device))

        # 4. Confidence BCE Loss
        loss_conf = F.binary_cross_entropy(conf, target_conf)

        total_loss = loss_lm + 0.5 * loss_span + 0.2 * loss_type + 0.1 * loss_conf
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == steps:
            loss_history.append({
                "step": step,
                "loss_total": round(total_loss.item(), 4),
                "loss_lm": round(loss_lm.item(), 4),
                "loss_span": round(loss_span.item(), 4),
                "conf": round(conf[0, 0].item(), 4),
            })

    duration = time.perf_counter() - t0

    # Save checkpoint
    OUT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"seed{seed}" if ablation is None else f"seed{seed}_{ablation}"
    ckpt_path = OUT_CKPT_DIR / f"candidate_step{steps}_{suffix}_span_grounding.pt"
    torch.save(model.state_dict(), ckpt_path)

    ckpt_hash = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

    return {
        "seed": seed,
        "steps": steps,
        "ablation": ablation,
        "train_duration_sec": round(duration, 2),
        "final_loss": round(total_loss.item(), 4),
        "loss_trajectory": loss_history,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_hash,
    }


def main():
    print("--- Training Seed 42 Span Grounding Model ---")
    res_42 = train_span_model(seed=42, steps=250)
    print(f"Seed 42 completed in {res_42['train_duration_sec']}s, final loss: {res_42['final_loss']}")

    print("--- Training Seed 101 Span Grounding Model ---")
    res_101 = train_span_model(seed=101, steps=250)
    print(f"Seed 101 completed in {res_101['train_duration_sec']}s, final loss: {res_101['final_loss']}")

    print("--- Training Ablation: No Type Classification ---")
    res_no_type = train_span_model(seed=42, steps=250, ablation="no_type")
    print(f"No Type completed, final loss: {res_no_type['final_loss']}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    training_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed_42": res_42,
        "seed_101": res_101,
        "ablation_no_type": res_no_type,
    }
    with open(REPORT_DIR / "SPAN_TRAINING_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump(training_report, f, indent=2)

    with open(REPORT_DIR / "SPAN_ABLATION_RESULTS.json", "w", encoding="utf-8") as f:
        json.dump({"ablation_no_type": res_no_type}, f, indent=2)

    with open(REPORT_DIR / "SEED_COMPARISON.json", "w", encoding="utf-8") as f:
        json.dump({"seed_42": res_42, "seed_101": res_101}, f, indent=2)

    print("Span training and ablation reports generated successfully.")


if __name__ == "__main__":
    main()
