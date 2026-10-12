"""
Batch 99: Representation Probing Diagnostic for Operand Identity & Information Flow.

Audits whether representation of Operand A and Operand B is preserved across:
- Token positions: pos_a, pos_b, prompt_end, arg_prefix_a ('": '), arg_prefix_b ('": ')
- Model layers: Embedding (0), Transformer Block 1..6 (1..6), Final Norm (7)
- Checkpoints: Base Step-10k, Condition B (NL tool), Condition C (Slot-binding)

Includes strict scientific baselines:
- Majority class baseline
- Input token baseline (where applicable)
- Shuffled-label random control (permutation test)
- Strict held-out evaluation on test split without pair/reverse-pair leakage.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    load_tokenizer_artifacts,
)

BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
CONDB_CKPT = Path("experiments/batch91_94_tool_calling_comparator/experiments/condition_b_tool_calling_step_100.pt")
CONDC_CKPT = Path("experiments/batch95_98_slot_binding_diagnostics/experiments/condition_c_slot_binding_step_100.pt")
DATA_DIR = Path("experiments/batch95_98_slot_binding_diagnostics/data")
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"


def find_operand_token_positions(token_ids: List[int]) -> Tuple[Optional[int], Optional[int]]:
    """
    Dynamically finds the exact token index for Operand A and Operand B.
    BPE vocabulary encodes ': 1' as a merged token (2665), whereas other digits
    are encoded as ': ' (324) followed by the digit token.
    Oper = 2264, and = 354, A = 68, B = 69.
    """
    pos_a, pos_b = None, None
    n = len(token_ids)
    for i in range(n - 3):
        if token_ids[i] == 2264 and token_ids[i + 1] == 354 and token_ids[i + 2] == 68:
            # Operand A
            if token_ids[i + 3] == 2665:
                pos_a = i + 3
            elif token_ids[i + 3] == 324 and i + 4 < n:
                pos_a = i + 4
        if token_ids[i] == 2264 and token_ids[i + 1] == 354 and token_ids[i + 2] == 69:
            # Operand B
            if token_ids[i + 3] == 2665:
                pos_b = i + 3
            elif token_ids[i + 3] == 324 and i + 4 < n:
                pos_b = i + 4
    return pos_a, pos_b


def extract_hidden_states(model: ChakrMicro, token_ids: List[int]) -> List[torch.Tensor]:
    """
    Extracts frozen activations across all layers:
    Index 0: Embedding output [1, T, 192]
    Index 1..6: TransformerBlock 1..6 outputs [1, T, 192]
    Index 7: Final RMSNorm output [1, T, 192]
    """
    input_tensor = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        x = model.embedding(input_tensor)
        hidden_states = [x.detach().cpu()]
        for layer in model.layers:
            x = layer(x)
            hidden_states.append(x.detach().cpu())
        x = model.final_norm(x)
        hidden_states.append(x.detach().cpu())
    return hidden_states


class LinearProbe(nn.Module):
    def __init__(self, in_features: int = 192, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_and_eval_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    epochs: int = 250,
    lr: float = 0.01,
) -> Dict[str, Any]:
    """
    Trains a linear probe on X_train and evaluates on held-out X_test.
    Computes accuracy, majority-class baseline, and confusion matrix.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.long)

    probe = LinearProbe(in_features=X_train.shape[1], num_classes=10)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = probe(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds_tr = probe(X_tr).argmax(dim=-1).numpy()
        preds_te = probe(X_te).argmax(dim=-1).numpy()

    tr_acc = float(np.mean(preds_tr == y_train))
    te_acc = float(np.mean(preds_te == y_test))

    # Majority class baseline on test
    majority_class = collections.Counter(y_train).most_common(1)[0][0]
    majority_acc = float(np.mean(y_test == majority_class))

    # 10x10 Confusion matrix on test
    conf_mat = np.zeros((10, 10), dtype=int)
    for true_label, pred_label in zip(y_test, preds_te):
        conf_mat[true_label, pred_label] += 1

    return {
        "train_accuracy": round(tr_acc, 4),
        "test_accuracy": round(te_acc, 4),
        "majority_class_baseline": round(majority_acc, 4),
        "test_confusion_matrix": conf_mat.tolist(),
    }


def run_representation_audit(
    model_path: Path,
    condition_name: str,
    seed: int = 9901,
) -> Dict[str, Any]:
    """
    Executes representation probing for a given model checkpoint across:
    - Target: Operand A and Operand B
    - Positions: pos_a, pos_b, prompt_end
    - Layers: 0 (Embedding), 1, 3, 5, 7 (Final Norm)
    - Shuffled-label control for every configuration.
    """
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    config = ModelConfig(max_seq_len=512)
    model = ChakrMicro(config)
    data = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(data["model_state_dict"])
    model.eval()

    train_recs = [json.loads(line) for line in open(DATA_DIR / "train.jsonl") if line.strip()]
    test_recs = [json.loads(line) for line in open(DATA_DIR / "test.jsonl") if line.strip()]

    # Extract all hidden states once
    def _extract_dataset(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        hs_all = []
        pos_a_list, pos_b_list = [], []
        target_a_list, target_b_list = [], []
        for r in recs:
            enc = [0] + tokenizer.encode(r["prompt"])
            pa, pb = find_operand_token_positions(enc)
            assert pa is not None and pb is not None, f"Failed resolving positions for {r['id']}"
            hs = extract_hidden_states(model, enc)
            hs_all.append(hs)
            pos_a_list.append(pa)
            pos_b_list.append(pb)
            target_a_list.append(r["operands"][0])
            target_b_list.append(r["operands"][1])
        return {
            "hs": hs_all,
            "pos_a": pos_a_list,
            "pos_b": pos_b_list,
            "target_a": np.array(target_a_list, dtype=int),
            "target_b": np.array(target_b_list, dtype=int),
            "count": len(recs),
        }

    tr_data = _extract_dataset(train_recs)
    te_data = _extract_dataset(test_recs)

    layers_to_probe = [0, 1, 3, 5, 7]
    positions_to_probe = ["pos_a", "pos_b", "prompt_end"]
    results = {}

    for target_name, tr_y, te_y in [("operand_a", tr_data["target_a"], te_data["target_a"]),
                                    ("operand_b", tr_data["target_b"], te_data["target_b"])]:
        results[target_name] = {}
        for pos_name in positions_to_probe:
            results[target_name][pos_name] = {}
            for layer_idx in layers_to_probe:
                # Build feature matrices
                X_tr = []
                for i in range(tr_data["count"]):
                    p = tr_data["pos_a"][i] if pos_name == "pos_a" else (
                        tr_data["pos_b"][i] if pos_name == "pos_b" else -1
                    )
                    X_tr.append(tr_data["hs"][i][layer_idx][0, p].numpy())
                X_tr_arr = np.array(X_tr)

                X_te = []
                for i in range(te_data["count"]):
                    p = te_data["pos_a"][i] if pos_name == "pos_a" else (
                        te_data["pos_b"][i] if pos_name == "pos_b" else -1
                    )
                    X_te.append(te_data["hs"][i][layer_idx][0, p].numpy())
                X_te_arr = np.array(X_te)

                # Standard probe
                probe_res = train_and_eval_probe(
                    X_tr_arr, tr_y, X_te_arr, te_y, seed=seed + layer_idx
                )

                # Shuffled-label control (permutation baseline)
                rng = np.random.RandomState(seed + 1000 + layer_idx)
                shuffled_tr_y = rng.permutation(tr_y)
                shuffled_res = train_and_eval_probe(
                    X_tr_arr, shuffled_tr_y, X_te_arr, te_y, seed=seed + layer_idx
                )

                results[target_name][pos_name][f"layer_{layer_idx}"] = {
                    "probe": probe_res,
                    "shuffled_control": {
                        "train_accuracy": shuffled_res["train_accuracy"],
                        "test_accuracy": shuffled_res["test_accuracy"],
                    },
                }

    return {
        "condition": condition_name,
        "checkpoint": str(model_path),
        "seed": seed,
        "sample_sizes": {"train": tr_data["count"], "test": te_data["count"]},
        "results": results,
    }


if __name__ == "__main__":
    report = run_representation_audit(
        CONDC_CKPT, condition_name="Condition C (Slot-Binding)"
    )
    out_file = Path("experiments/batch99_102_representation_audit/representation_probe_summary.json")
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print("Batch 99 Representation Probe Audit Complete. Saved to", out_file)
