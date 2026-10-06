"""
ChakrView Step 162: Minimal Reasoning Learnability Ladder.

Establishes the minimal learnable tasks from L0 to L6:
- L0: Token Identity (copy prompt symbol to output)
- L1: Direct Relation Recognition ('order: A > B -> first: A')
- L2: Direct Relation Inversion ('order: A > B -> second: B')
- L3: Binary Classification ('is A > B ? T' / 'is B > A ? F')
- L4: Two-Hop Transitive Relation ('A > B and B > C -> first: A')
- L5: Three-Hop Relation ('A > B, B > C, C > D -> first: A')
- L6: Four-Hop Relation ('A > B, B > C, C > D, D > E -> first: A')

For every level reports:
- train_accuracy
- validation_accuracy
- heldout_accuracy
- target_probability
- target_rank
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


@dataclass
class LevelEvaluationResult:
    level_id: str
    level_name: str
    train_accuracy: float
    validation_accuracy: float
    heldout_accuracy: float
    target_probability: float
    target_rank: int
    generalization_observed: bool


@dataclass
class LearnabilityLadderReport:
    report_id: str
    level_results: Dict[str, LevelEvaluationResult]
    l1_generalization_status: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "level_results": {k: asdict(v) for k, v in self.level_results.items()},
            "l1_generalization_status": self.l1_generalization_status,
            "summary": self.summary,
        }


class MinimalReasoningLearnabilityLadder:
    """
    Evaluates progressive minimal tasks from L0 to L6 on isolated candidate models.
    """

    def __init__(
        self,
        tokenizer: Optional[BPETokenizer] = None,
        device: str = "cpu",
    ) -> None:
        self.device = device
        if tokenizer is None:
            tok_dir = Path("data/experiments/vocab_4096")
            tok, _ = load_tokenizer_artifacts(tok_dir)
            self.tokenizer = tok
        else:
            self.tokenizer = tokenizer

    def _evaluate_set(
        self,
        model: ChakrMicro,
        items: List[Tuple[str, str]],
    ) -> Tuple[float, float, int]:
        model.eval()
        correct = 0
        probs: List[float] = []
        ranks: List[int] = []

        for p, t in items:
            p_ids = self.tokenizer.encode(p, add_bos=True, add_eos=False)
            t_ids = self.tokenizer.encode(t, add_bos=False, add_eos=False)
            if not t_ids:
                continue
            target_id = t_ids[0]

            with torch.no_grad():
                logits = model(torch.tensor([p_ids], device=self.device))
                pos_logits = logits[0, -1, :]
                pred_id = torch.argmax(pos_logits).item()
                soft = torch.softmax(pos_logits, dim=-1)
                p_val = soft[target_id].item()
                r_val = (pos_logits > pos_logits[target_id]).sum().item() + 1

            if pred_id == target_id:
                correct += 1
            probs.append(p_val)
            ranks.append(r_val)

        acc = correct / max(1, len(items))
        avg_prob = sum(probs) / max(1, len(probs))
        med_rank = sorted(ranks)[len(ranks) // 2] if ranks else 1
        return round(acc, 4), round(avg_prob, 6), med_rank

    def evaluate_ladder(
        self,
        candidate_model: ChakrMicro,
        train_steps: int = 50,
        lr: float = 1e-3,
    ) -> LearnabilityLadderReport:
        # Define clean symbols
        symbols_train = ["A", "B", "C", "D", "E"]
        symbols_heldout = ["X", "Y", "Z", "W", "V"]

        levels_data: Dict[str, Tuple[str, List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]] = {
            "L0": (
                "Token Identity",
                [(f"copy {s} = ", s) for s in ["1", "2", "3"]],
                [(f"copy {s} = ", s) for s in ["4", "5"]],
                [(f"copy {s} = ", s) for s in ["6", "7"]],
            ),
            "L1": (
                "Direct Relation Recognition",
                [(f"order: {symbols_train[i]} > {symbols_train[i+1]} -> first: ", symbols_train[i]) for i in range(len(symbols_train)-1)],
                [(f"order: {symbols_train[0]} > {symbols_train[2]} -> first: ", symbols_train[0])],
                [(f"compare: {symbols_train[i]} > {symbols_train[i+1]} -> first: ", symbols_train[i]) for i in range(len(symbols_train)-1)],
            ),
            "L2": (
                "Direct Relation Inversion",
                [(f"order: {symbols_train[i]} > {symbols_train[i+1]} -> second: ", symbols_train[i+1]) for i in range(len(symbols_train)-1)],
                [(f"order: {symbols_train[0]} > {symbols_train[2]} -> second: ", symbols_train[2])],
                [(f"order: {symbols_heldout[i]} > {symbols_heldout[i+1]} -> second: ", symbols_heldout[i+1]) for i in range(len(symbols_heldout)-1)],
            ),
            "L3": (
                "Binary Classification",
                [("is 2 > 1 ? ", "T"), ("is 1 > 2 ? ", "F"), ("is 3 > 2 ? ", "T"), ("is 2 > 3 ? ", "F")],
                [("is 4 > 3 ? ", "T"), ("is 3 > 4 ? ", "F")],
                [("is 8 > 7 ? ", "T"), ("is 7 > 8 ? ", "F"), ("is 9 > 8 ? ", "T"), ("is 8 > 9 ? ", "F")],
            ),
            "L4": (
                "Two-Hop Transitive",
                [(f"chain: {symbols_train[i]} > {symbols_train[i+1]} , {symbols_train[i+1]} > {symbols_train[i+2]} -> first: ", symbols_train[i]) for i in range(len(symbols_train)-2)],
                [(f"chain: {symbols_train[0]} > {symbols_train[1]} , {symbols_train[1]} > {symbols_train[3]} -> first: ", symbols_train[0])],
                [(f"chain: {symbols_heldout[i]} > {symbols_heldout[i+1]} , {symbols_heldout[i+1]} > {symbols_heldout[i+2]} -> first: ", symbols_heldout[i]) for i in range(len(symbols_heldout)-2)],
            ),
            "L5": (
                "Three-Hop Transitive",
                [(f"chain: {symbols_train[0]} > {symbols_train[1]} , {symbols_train[1]} > {symbols_train[2]} , {symbols_train[2]} > {symbols_train[3]} -> first: ", symbols_train[0])],
                [(f"chain: {symbols_train[1]} > {symbols_train[2]} , {symbols_train[2]} > {symbols_train[3]} , {symbols_train[3]} > {symbols_train[4]} -> first: ", symbols_train[1])],
                [(f"chain: {symbols_heldout[0]} > {symbols_heldout[1]} , {symbols_heldout[1]} > {symbols_heldout[2]} , {symbols_heldout[2]} > {symbols_heldout[3]} -> first: ", symbols_heldout[0])],
            ),
            "L6": (
                "Four-Hop Transitive",
                [(f"chain: {symbols_train[0]} > {symbols_train[1]} , {symbols_train[1]} > {symbols_train[2]} , {symbols_train[2]} > {symbols_train[3]} , {symbols_train[3]} > {symbols_train[4]} -> first: ", symbols_train[0])],
                [(f"chain: {symbols_train[0]} > {symbols_train[1]} , {symbols_train[1]} > {symbols_train[2]} , {symbols_train[2]} > {symbols_train[3]} , {symbols_train[3]} > {symbols_train[4]} -> first: ", symbols_train[0])],
                [(f"chain: {symbols_heldout[0]} > {symbols_heldout[1]} , {symbols_heldout[1]} > {symbols_heldout[2]} , {symbols_heldout[2]} > {symbols_heldout[3]} , {symbols_heldout[3]} > {symbols_heldout[4]} -> first: ", symbols_heldout[0])],
            ),
        }

        results: Dict[str, LevelEvaluationResult] = {}

        for lvl_id, (name, train_set, val_set, held_set) in levels_data.items():
            # Train a localized isolated copy for this level
            cand_lvl = copy.deepcopy(candidate_model)
            cand_lvl.train()
            optimizer = torch.optim.AdamW(cand_lvl.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()

            train_tensors = []
            for p, t in train_set:
                ids = self.tokenizer.encode(p + t, add_bos=True, add_eos=False)
                train_tensors.append(torch.tensor(ids, device=self.device))

            if train_tensors:
                for step in range(train_steps):
                    seq = train_tensors[step % len(train_tensors)]
                    inp = seq[:-1].unsqueeze(0)
                    tgt = seq[1:].unsqueeze(0)
                    optimizer.zero_grad()
                    logits = cand_lvl(inp)
                    # answer target position loss
                    loss = loss_fn(logits[0, -1, :].unsqueeze(0), tgt[0, -1].unsqueeze(0))
                    loss.backward()
                    optimizer.step()

            tr_acc, tr_prob, tr_rank = self._evaluate_set(cand_lvl, train_set)
            v_acc, v_prob, v_rank = self._evaluate_set(cand_lvl, val_set)
            h_acc, h_prob, h_rank = self._evaluate_set(cand_lvl, held_set)

            gen_obs = (h_acc > 0.0)
            results[lvl_id] = LevelEvaluationResult(
                level_id=lvl_id,
                level_name=name,
                train_accuracy=tr_acc,
                validation_accuracy=v_acc,
                heldout_accuracy=h_acc,
                target_probability=h_prob,
                target_rank=h_rank,
                generalization_observed=gen_obs,
            )

        l1_ok = results["L1"].heldout_accuracy > 0.0
        summary = (
            f"L1 generalization: {results['L1'].heldout_accuracy:.4f} (Observed: {l1_ok}). "
            f"L3 binary classification: {results['L3'].heldout_accuracy:.4f}."
        )

        return LearnabilityLadderReport(
            report_id="rep_step162_ladder",
            level_results=results,
            l1_generalization_status=l1_ok,
            summary=summary,
        )
