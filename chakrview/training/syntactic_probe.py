"""
Syntactic Probe Evaluator and Benchmark Harness for Step 102.

Evaluates next-token prediction on frozen cloze probes without generating unconstrained text:
- Top-1 next-token accuracy
- Top-5 next-token accuracy
- Target token cross-entropy loss and log probability
- Category-level breakdown (python_syntax, delimiters, control_flow, natural_language)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer


@dataclass
class ProbeResult:
    probe_id: str
    category: str
    prompt: str
    target_token_id: int
    target_token_text: str
    predicted_top1_token_id: int
    predicted_top1_token_text: str
    top1_correct: bool
    top5_correct: bool
    target_log_prob: float
    top5_predicted_tokens: List[str]


@dataclass
class SyntacticProbeMetrics:
    total_probes: int
    top1_correct_count: int
    top1_accuracy: float
    top5_correct_count: int
    top5_accuracy: float
    mean_target_log_prob: float
    category_metrics: Dict[str, Dict[str, float]]
    probe_results: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SyntacticProbeEvaluator:
    """
    Evaluates next-token cloze probes deterministically on ChakrMicro.
    """

    def __init__(
        self,
        tokenizer: BPETokenizer,
        probes_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device
        if probes_path is None:
            probes_path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "step102_syntactic_probes.json"
        self.probes_path = Path(probes_path)
        with open(self.probes_path, "r", encoding="utf-8") as f:
            self.probes = json.load(f)

    def evaluate(self, model: ChakrMicro) -> SyntacticProbeMetrics:
        model.eval()
        model.to(self.device)

        results: List[ProbeResult] = []
        category_stats: Dict[str, Dict[str, Any]] = {}

        for probe in self.probes:
            pid = probe["probe_id"]
            cat = probe["category"]
            prompt = probe["prompt"]
            target_str = probe["target"]
            alts_str = probe.get("valid_alternatives", [])

            # Tokenize prompt and target
            p_tokens = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            t_tokens = self.tokenizer.encode(target_str, add_bos=False, add_eos=False)
            target_id = t_tokens[0]

            alt_ids = set()
            for a in alts_str:
                atoks = self.tokenizer.encode(a, add_bos=False, add_eos=False)
                if atoks:
                    alt_ids.add(atoks[0])

            valid_target_ids = {target_id} | alt_ids

            input_tensor = torch.tensor([p_tokens], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = model(input_tensor)  # [1, seq_len, vocab_size]
                last_logits = logits[0, -1, :]  # [vocab_size]
                log_probs = F.log_softmax(last_logits, dim=-1)

            top5_indices = torch.topk(last_logits, k=5).indices.tolist()
            top1_id = top5_indices[0]

            top1_correct = (top1_id in valid_target_ids)
            top5_correct = any(t in valid_target_ids for t in top5_indices)
            target_log_prob = float(log_probs[target_id].item())

            top5_texts = [self.tokenizer.decode([idx], errors="replace") for idx in top5_indices]

            res = ProbeResult(
                probe_id=pid,
                category=cat,
                prompt=prompt,
                target_token_id=target_id,
                target_token_text=self.tokenizer.decode([target_id], errors="replace"),
                predicted_top1_token_id=top1_id,
                predicted_top1_token_text=self.tokenizer.decode([top1_id], errors="replace"),
                top1_correct=top1_correct,
                top5_correct=top5_correct,
                target_log_prob=target_log_prob,
                top5_predicted_tokens=top5_texts,
            )
            results.append(res)

            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "top1": 0, "top5": 0}
            category_stats[cat]["total"] += 1
            if top1_correct:
                category_stats[cat]["top1"] += 1
            if top5_correct:
                category_stats[cat]["top5"] += 1

        total = len(results)
        top1_tot = sum(1 for r in results if r.top1_correct)
        top5_tot = sum(1 for r in results if r.top5_correct)
        mean_lp = sum(r.target_log_prob for r in results) / total if total > 0 else 0.0

        cat_metrics: Dict[str, Dict[str, float]] = {}
        for c, s in category_stats.items():
            tot = s["total"]
            cat_metrics[c] = {
                "total": tot,
                "top1_accuracy": round((s["top1"] / tot) * 100.0, 2) if tot > 0 else 0.0,
                "top5_accuracy": round((s["top5"] / tot) * 100.0, 2) if tot > 0 else 0.0,
            }

        return SyntacticProbeMetrics(
            total_probes=total,
            top1_correct_count=top1_tot,
            top1_accuracy=round((top1_tot / total) * 100.0, 2) if total > 0 else 0.0,
            top5_correct_count=top5_tot,
            top5_accuracy=round((top5_tot / total) * 100.0, 2) if total > 0 else 0.0,
            mean_target_log_prob=round(mean_lp, 4),
            category_metrics=cat_metrics,
            probe_results=[asdict(r) for r in results],
        )
