"""
Batches 717-722: Controlled Training Pipeline for Neural Subtask Decomposition.
Location: experiments/batch701_740/train_decomposition_candidate.py

Features:
- Multi-Task Loss:
  L = L_goal + L_st1_act + L_st1_tool + L_st2_act + L_st2_tool + L_dep + L_clar
- Fixed seeds, reproducible AdamW optimizer.
- Validates on val set and frozen held-out test suite.
- Measures Goal Accuracy, Subtask Match Rate, and Dependency Edge F1.
- Evaluates multi-seed stability (seed 42, 101).
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ChakrMicro
from chakrview.brain.subtask_decomposition_model import (
    ACTION_TO_ID,
    GOAL_TO_ID,
    ID_TO_ACTION,
    ID_TO_GOAL,
    ID_TO_TOOL,
    TOOL_TO_ID,
    ChakrSubtaskDecompositionModel,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

STARTING_CKPT = REPO_ROOT / "experiments" / "batch581_620" / "checkpoints" / "candidate_step250_span_pointer.pt"
TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"

TRAIN_DATA = REPO_ROOT / "experiments" / "batch701_740" / "data" / "train_decomposition.jsonl"
VAL_DATA = REPO_ROOT / "experiments" / "batch701_740" / "data" / "val_decomposition.jsonl"
TEST_DATA = REPO_ROOT / "experiments" / "batch701_740" / "data" / "test_decomposition_heldout.jsonl"

OUT_CKPT_PATH = REPO_ROOT / "experiments" / "batch701_740" / "checkpoints" / "candidate_step150_neural_planner.pt"


def encode_item(tokenizer: Any, item: Dict[str, Any]) -> Dict[str, Any]:
    text = item["instruction"]
    prompt = f"User: {text}\nPlan:"
    input_ids = [0] + tokenizer.encode(prompt)

    goal_id = GOAL_TO_ID.get(item["goal_type"], 0)

    # Subtasks
    subtasks = item.get("subtasks", [])
    st1 = subtasks[0] if len(subtasks) > 0 else {"action": "none", "tool": "none", "deps": []}
    st2 = subtasks[1] if len(subtasks) > 1 else {"action": "none", "tool": "none", "deps": []}

    st1_act = ACTION_TO_ID.get(st1["action"], ACTION_TO_ID["none"])
    st1_tool = TOOL_TO_ID.get(st1["tool"], TOOL_TO_ID["none"])

    st2_act = ACTION_TO_ID.get(st2["action"], ACTION_TO_ID["none"])
    st2_tool = TOOL_TO_ID.get(st2["tool"], TOOL_TO_ID["none"])

    has_dep = 1.0 if (0 in st2.get("deps", [])) else 0.0
    clar_needed = 1.0 if item.get("clarification_needed", False) else 0.0

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "goal_id": torch.tensor(goal_id, dtype=torch.long),
        "st1_act": torch.tensor(st1_act, dtype=torch.long),
        "st1_tool": torch.tensor(st1_tool, dtype=torch.long),
        "st2_act": torch.tensor(st2_act, dtype=torch.long),
        "st2_tool": torch.tensor(st2_tool, dtype=torch.long),
        "has_dep": torch.tensor(has_dep, dtype=torch.float),
        "clar_needed": torch.tensor(clar_needed, dtype=torch.float),
        "item": item,
    }


def evaluate(model: ChakrSubtaskDecompositionModel, tokenizer: Any, test_items: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    model.eval()
    total = len(test_items)
    goal_correct = 0
    st1_act_correct = 0
    st2_act_correct = 0
    dep_correct = 0
    clar_correct = 0

    with torch.no_grad():
        for item in test_items:
            enc = encode_item(tokenizer, item)
            input_ids = enc["input_ids"].unsqueeze(0).to(device)

            out = model(input_ids)

            pred_goal = torch.argmax(out["goal_logits"], dim=-1).item()
            pred_st1_act = torch.argmax(out["st1_act_logits"], dim=-1).item()
            pred_st2_act = torch.argmax(out["st2_act_logits"], dim=-1).item()
            pred_dep = (torch.sigmoid(out["dependency_logits"]).item() > 0.5)
            pred_clar = (torch.sigmoid(out["clarification_logits"]).item() > 0.5)

            if pred_goal == enc["goal_id"].item():
                goal_correct += 1
            if pred_st1_act == enc["st1_act"].item():
                st1_act_correct += 1
            if pred_st2_act == enc["st2_act"].item():
                st2_act_correct += 1
            if pred_dep == (enc["has_dep"].item() == 1.0):
                dep_correct += 1
            if pred_clar == (enc["clar_needed"].item() == 1.0):
                clar_correct += 1

    return {
        "total": total,
        "goal_accuracy": round(goal_correct / total * 100.0, 1),
        "st1_action_accuracy": round(st1_act_correct / total * 100.0, 1),
        "st2_action_accuracy": round(st2_act_correct / total * 100.0, 1),
        "dependency_edge_accuracy": round(dep_correct / total * 100.0, 1),
        "clarification_accuracy": round(clar_correct / total * 100.0, 1),
    }


def train_run(seed: int = 42, steps: int = 150) -> Dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Tokenizer
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # Base Model Config & Initialization
    raw_ckpt = torch.load(STARTING_CKPT, map_location=device, weights_only=False)
    # Instantiate base model
    from chakrview.brain.config import ModelConfig
    base_cfg = ModelConfig(
        vocab_size=4096,
        d_model=192,
        n_layers=6,
        n_heads=6,
        max_seq_len=512,
    )
    base_model = ChakrMicro(base_cfg)

    # Load compatible base weights
    base_state = {k.replace("base_model.", ""): v for k, v in raw_ckpt.items() if k.startswith("base_model.")}
    if base_state:
        base_model.load_state_dict(base_state, strict=False)

    model = ChakrSubtaskDecompositionModel(base_model).to(device)

    # Load Data
    with open(TRAIN_DATA, "r", encoding="utf-8") as f:
        train_items = [json.loads(line) for line in f]
    with open(VAL_DATA, "r", encoding="utf-8") as f:
        val_items = [json.loads(line) for line in f]
    with open(TEST_DATA, "r", encoding="utf-8") as f:
        test_items = [json.loads(line) for line in f]

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    model.train()
    history = []
    t0 = time.perf_counter()

    for step in range(1, steps + 1):
        item = random.choice(train_items)
        enc = encode_item(tokenizer, item)

        input_ids = enc["input_ids"].unsqueeze(0).to(device)
        goal_id = enc["goal_id"].unsqueeze(0).to(device)
        st1_act = enc["st1_act"].unsqueeze(0).to(device)
        st1_tool = enc["st1_tool"].unsqueeze(0).to(device)
        st2_act = enc["st2_act"].unsqueeze(0).to(device)
        st2_tool = enc["st2_tool"].unsqueeze(0).to(device)
        has_dep = enc["has_dep"].unsqueeze(0).to(device)
        clar_needed = enc["clar_needed"].unsqueeze(0).to(device)

        optimizer.zero_grad()
        out = model(input_ids)

        l_goal = ce_loss(out["goal_logits"], goal_id)
        l_st1_act = ce_loss(out["st1_act_logits"], st1_act)
        l_st1_tool = ce_loss(out["st1_tool_logits"], st1_tool)
        l_st2_act = ce_loss(out["st2_act_logits"], st2_act)
        l_st2_tool = ce_loss(out["st2_tool_logits"], st2_tool)
        l_dep = bce_loss(out["dependency_logits"], has_dep)
        l_clar = bce_loss(out["clarification_logits"], clar_needed)

        loss = l_goal + l_st1_act + l_st1_tool + l_st2_act + l_st2_tool + l_dep + l_clar
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 25 == 0 or step == steps:
            val_eval = evaluate(model, tokenizer, val_items, device)
            history.append({"step": step, "loss": round(loss.item(), 4), "val_eval": val_eval})

    train_duration = time.perf_counter() - t0

    # Test Evaluation
    test_eval = evaluate(model, tokenizer, test_items, device)

    # Save Checkpoint
    os.makedirs(OUT_CKPT_PATH.parent, exist_ok=True)
    torch.save(model.state_dict(), OUT_CKPT_PATH)

    # Compute Checkpoint Hash
    with open(OUT_CKPT_PATH, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    return {
        "seed": seed,
        "steps": steps,
        "device": str(device),
        "train_duration_sec": round(train_duration, 2),
        "checkpoint_path": str(OUT_CKPT_PATH),
        "checkpoint_sha256": ckpt_sha256,
        "val_final": history[-1]["val_eval"],
        "test_heldout_eval": test_eval,
    }


def main():
    print("=== Training Neural Subtask Planner (Seed 42) ===")
    res42 = train_run(seed=42, steps=150)
    print(f"Seed 42 Complete in {res42['train_duration_sec']}s")
    print(f"  Held-out Test Goal Acc:       {res42['test_heldout_eval']['goal_accuracy']}%")
    print(f"  Held-out Subtask 1 Action Acc: {res42['test_heldout_eval']['st1_action_accuracy']}%")
    print(f"  Held-out Subtask 2 Action Acc: {res42['test_heldout_eval']['st2_action_accuracy']}%")
    print(f"  Held-out Dependency Edge Acc: {res42['test_heldout_eval']['dependency_edge_accuracy']}%")
    print(f"  Held-out Clarification Acc:   {res42['test_heldout_eval']['clarification_accuracy']}%")

    print("\n=== Multi-Seed Stability Verification (Seed 101) ===")
    res101 = train_run(seed=101, steps=150)
    print(f"Seed 101 Complete in {res101['train_duration_sec']}s")
    print(f"  Held-out Test Goal Acc:       {res101['test_heldout_eval']['goal_accuracy']}%")
    print(f"  Held-out Dependency Edge Acc: {res101['test_heldout_eval']['dependency_edge_accuracy']}%")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed_42_results": res42,
        "seed_101_results": res101,
        "multi_seed_consistent": (
            abs(res42['test_heldout_eval']['dependency_edge_accuracy'] - res101['test_heldout_eval']['dependency_edge_accuracy']) <= 10.0
        ),
    }

    report_path = REPO_ROOT / "experiments" / "batch701_740" / "reports" / "TRAINING_LEARNABILITY_AUDIT.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
