"""
ChakrView Step 148: Generalization & Transfer Learning Evaluator.

Features:
- Multi-domain transfer matrix across:
  DOMAIN_A (Coding: Python def/return),
  DOMAIN_B (Math: arithmetic operators),
  DOMAIN_C (Logic: implications)
- Tracks:
  - acquisition_delta: Improvement in newly trained domain
  - retention_delta: Preservation of earlier domain accuracy
  - transfer_delta: Zero-shot gain in unvisited domain
  - forgetting_delta: Performance degradation in historical domain
- Connects directly with anti-forgetting coordinator from Step 139.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.anti_forgetting import MultiDomainAntiForgettingCoordinator
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


@dataclass
class TransferLearningMetrics:
    domain_accuracies: Dict[str, float]
    acquisition_delta: float
    retention_delta: float
    transfer_delta: float
    forgetting_delta: float
    has_positive_transfer: bool
    retention_intact: bool


class GeneralizationTransferEvaluator:
    """
    Measures cross-domain acquisition, retention, and transfer performance.
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

    def evaluate_domain_accuracy(self, model: ChakrMicro, samples: List[Tuple[str, str]]) -> float:
        model.eval()
        correct = 0
        total = 0

        for prompt, target_str in samples:
            p_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            t_ids = self.tokenizer.encode(target_str, add_bos=False, add_eos=False)
            if not t_ids:
                continue
            target_id = t_ids[0]

            inp = torch.tensor([p_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = model(inp)
                pred_id = torch.argmax(logits[0, -1, :]).item()

            if pred_id == target_id:
                correct += 1
            total += 1

        return correct / max(1, total)

    def evaluate_multi_domain_transfer(
        self,
        baseline_model: ChakrMicro,
        candidate_model: ChakrMicro,
        domain_test_sets: Dict[str, List[Tuple[str, str]]],
    ) -> TransferLearningMetrics:
        """
        Computes acquisition, retention, and transfer deltas between baseline and candidate.
        """
        base_scores = {d: self.evaluate_domain_accuracy(baseline_model, data) for d, data in domain_test_sets.items()}
        cand_scores = {d: self.evaluate_domain_accuracy(candidate_model, data) for d, data in domain_test_sets.items()}

        domains = list(domain_test_sets.keys())
        dom_a = domains[0] if len(domains) > 0 else "DOMAIN_A"
        dom_b = domains[1] if len(domains) > 1 else "DOMAIN_B"
        dom_c = domains[2] if len(domains) > 2 else "DOMAIN_C"

        # Deltas
        acq_delta = round(cand_scores.get(dom_b, 0.0) - base_scores.get(dom_b, 0.0), 4)
        ret_delta = round(cand_scores.get(dom_a, 0.0) - base_scores.get(dom_a, 0.0), 4)
        trans_delta = round(cand_scores.get(dom_c, 0.0) - base_scores.get(dom_c, 0.0), 4)
        forget_delta = max(0.0, -ret_delta)

        return TransferLearningMetrics(
            domain_accuracies={d: round(cand_scores[d], 4) for d in domains},
            acquisition_delta=acq_delta,
            retention_delta=ret_delta,
            transfer_delta=trans_delta,
            forgetting_delta=round(forget_delta, 4),
            has_positive_transfer=(trans_delta >= 0.0),
            retention_intact=(forget_delta <= 0.05),
        )
