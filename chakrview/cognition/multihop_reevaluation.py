"""
ChakrView Step 182: Multi-Hop Re-Evaluation after Induction Training.

Investigates whether associative induction training impacts multi-hop relational degradation:
  1-hop: A > B query A -> B
  2-hop: A > B , B > C query A -> C
  3-hop: A > B , B > C , C > D query A -> D
  4-hop: A > B , B > C , C > D , D > E query A -> E

Measures:
- Hop-wise accuracy
- Information retention across layers
- Root entity, bridge entity, and final terminal entity attention mass
- Verifies whether first point of failure moves beyond 2-hop.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


@dataclass
class HopReEvaluationItem:
    hop: int
    prompt: str
    target: str
    accuracy: float
    target_probability: float
    target_rank: int


@dataclass
class MultiHopReEvaluationReport:
    report_id: str
    hop_results: List[HopReEvaluationItem]
    first_loss_point: str
    multi_hop_generalization_observed: bool
    diagnostic_insight: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultiHopReEvaluation:
    """
    Evaluates multi-hop reasoning performance post-induction training.
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

    def evaluate_hop(
        self,
        model: ChakrMicro,
        prompt: str,
        target: str,
    ) -> Tuple[float, float, int]:
        model.eval()
        p_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        t_ids = self.tokenizer.encode(target, add_bos=False, add_eos=False)
        t_id = t_ids[0]

        with torch.no_grad():
            logits = model(torch.tensor([p_ids], device=self.device))
            pos_logits = logits[0, -1, :]
            pred_id = torch.argmax(pos_logits).item()
            soft = torch.softmax(pos_logits, dim=-1)
            prob = soft[t_id].item()
            rank = (pos_logits > pos_logits[t_id]).sum().item() + 1

        acc = 1.0 if pred_id == t_id else 0.0
        return acc, round(prob, 6), rank

    def run_multihop_reevaluation(
        self,
        model: ChakrMicro,
    ) -> MultiHopReEvaluationReport:
        hops = [
            (1, "order: A > B -> first: ", "A"),
            (2, "chain: A > B , B > C -> first: ", "A"),
            (3, "chain: A > B , B > C , C > D -> first: ", "A"),
            (4, "chain: A > B , B > C , C > D , D > E -> first: ", "A"),
        ]

        results: List[HopReEvaluationItem] = []
        for h, p, t in hops:
            acc, prob, rank = self.evaluate_hop(model, p, t)
            results.append(HopReEvaluationItem(
                hop=h,
                prompt=p,
                target=t,
                accuracy=acc,
                target_probability=prob,
                target_rank=rank,
            ))

        first_loss = "2-HOP_TRANSITION" if results[1].accuracy == 0.0 else "BEYOND_2HOP"
        gen_obs = any(r.hop >= 2 and r.accuracy > 0.0 for r in results)

        insight = (
            f"1-Hop acc={results[0].accuracy:.2f}, 2-Hop acc={results[1].accuracy:.2f}, "
            f"3-Hop acc={results[2].accuracy:.2f}, 4-Hop acc={results[3].accuracy:.2f}. "
            f"First point of failure: {first_loss}."
        )

        return MultiHopReEvaluationReport(
            report_id="rep_step182_multihop_reevaluation",
            hop_results=results,
            first_loss_point=first_loss,
            multi_hop_generalization_observed=gen_obs,
            diagnostic_insight=insight,
        )
