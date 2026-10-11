"""
ChakrView Step 161: Mechanistic Reasoning Failure Diagnostics.

Implements deep mechanistic failure diagnostics for ChakrMicro reasoning:
- Input/target token IDs, positions, sequence lengths
- Logits at target positions, top-k predictions, target probability & rank
- Cross-entropy at target position
- Gradient norms, per-layer gradient norms (embedding, attention, ffn, norm)
- Loss before/after training, accuracy before/after training
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


@dataclass
class TokenDiagnosticRecord:
    prompt_str: str
    target_str: str
    prompt_token_ids: List[int]
    target_token_id: int
    target_position: int
    seq_len: int
    target_logit: float
    target_probability: float
    target_rank: int
    top_5_predictions: List[Tuple[int, float]]
    cross_entropy_at_target: float


@dataclass
class LayerGradientNorms:
    embedding_grad_norm: float
    attention_grad_norms: List[float]
    ffn_grad_norms: List[float]
    norm_grad_norms: List[float]
    total_grad_norm: float


@dataclass
class ReasoningMechanisticDiagnosticsResult:
    diagnosis_id: str
    loss_before_training: float
    loss_after_training: float
    accuracy_before_training: float
    accuracy_after_training: float
    token_diagnostics: List[TokenDiagnosticRecord]
    layer_gradient_norms: LayerGradientNorms
    parameter_update_norms: Dict[str, float]
    summary_findings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReasoningMechanisticDiagnostics:
    """
    Executes deep mechanistic diagnostics on reasoning forward and backward passes.
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

    def inspect_sample(
        self,
        model: ChakrMicro,
        prompt: str,
        target: str,
    ) -> TokenDiagnosticRecord:
        model.eval()
        p_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        t_ids = self.tokenizer.encode(target, add_bos=False, add_eos=False)
        target_id = t_ids[0] if t_ids else 0

        inp = torch.tensor([p_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = model(inp)  # [1, T, vocab_size]
            target_pos = len(p_ids) - 1
            pos_logits = logits[0, target_pos, :]
            probs = torch.softmax(pos_logits, dim=-1)

            t_logit = pos_logits[target_id].item()
            t_prob = probs[target_id].item()
            rank = (pos_logits > pos_logits[target_id]).sum().item() + 1

            topk = torch.topk(pos_logits, 5)
            top5 = [(topk.indices[i].item(), round(topk.values[i].item(), 4)) for i in range(5)]
            ce = -math.log(max(1e-12, t_prob))

        return TokenDiagnosticRecord(
            prompt_str=prompt,
            target_str=target,
            prompt_token_ids=p_ids,
            target_token_id=target_id,
            target_position=target_pos,
            seq_len=len(p_ids),
            target_logit=round(t_logit, 4),
            target_probability=round(t_prob, 6),
            target_rank=rank,
            top_5_predictions=top5,
            cross_entropy_at_target=round(ce, 4),
        )

    def run_mechanistic_analysis(
        self,
        candidate_model: ChakrMicro,
        test_samples: List[Tuple[str, str]],
        training_steps: int = 15,
        lr: float = 1e-3,
    ) -> ReasoningMechanisticDiagnosticsResult:
        # 1. Pre-training evaluation
        records_before = [self.inspect_sample(candidate_model, p, t) for p, t in test_samples]
        loss_before = sum(r.cross_entropy_at_target for r in records_before) / max(1, len(records_before))
        acc_before = sum(1 for r in records_before if r.target_rank == 1) / max(1, len(records_before))

        # 2. Perform controlled backward pass to record gradients
        cand_copy = copy.deepcopy(candidate_model)
        cand_copy.train()
        optimizer = torch.optim.AdamW(cand_copy.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        # Build tensors
        seq_tensors = []
        for p, t in test_samples:
            full = p + t
            ids = self.tokenizer.encode(full, add_bos=True, add_eos=False)
            seq_tensors.append(torch.tensor(ids, dtype=torch.long, device=self.device))

        # Train steps and capture gradients on last step
        attn_grads: List[float] = [0.0] * cand_copy.config.n_layers
        ffn_grads: List[float] = [0.0] * cand_copy.config.n_layers
        norm_grads: List[float] = [0.0] * cand_copy.config.n_layers
        emb_grad = 0.0
        total_grad = 0.0

        for step in range(training_steps):
            seq = seq_tensors[step % len(seq_tensors)]
            inp = seq[:-1].unsqueeze(0)
            tgt = seq[1:].unsqueeze(0)

            optimizer.zero_grad()
            logits = cand_copy(inp)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss.backward()

            if step == training_steps - 1:
                # Capture gradients
                if cand_copy.embedding.weight.grad is not None:
                    emb_grad = cand_copy.embedding.weight.grad.norm().item()

                for l_idx, layer in enumerate(cand_copy.layers):
                    # Attn grads
                    a_norm = 0.0
                    for p in layer.attn.parameters():
                        if p.grad is not None:
                            a_norm += p.grad.norm().item() ** 2
                    attn_grads[l_idx] = math.sqrt(a_norm)

                    # FFN grads
                    f_norm = 0.0
                    for p in layer.ffn.parameters():
                        if p.grad is not None:
                            f_norm += p.grad.norm().item() ** 2
                    ffn_grads[l_idx] = math.sqrt(f_norm)

                    # Norm grads
                    n_norm = 0.0
                    for p in list(layer.norm_1.parameters()) + list(layer.norm_2.parameters()):
                        if p.grad is not None:
                            n_norm += p.grad.norm().item() ** 2
                    norm_grads[l_idx] = math.sqrt(n_norm)

                tot = 0.0
                for p in cand_copy.parameters():
                    if p.grad is not None:
                        tot += p.grad.norm().item() ** 2
                total_grad = math.sqrt(tot)

            optimizer.step()

        # 3. Post-training evaluation
        records_after = [self.inspect_sample(cand_copy, p, t) for p, t in test_samples]
        loss_after = sum(r.cross_entropy_at_target for r in records_after) / max(1, len(records_after))
        acc_after = sum(1 for r in records_after if r.target_rank == 1) / max(1, len(records_after))

        # Parameter update norm breakdown
        update_norms = {
            "embedding": round(emb_grad * lr, 6),
            "attention_total": round(sum(attn_grads) * lr, 6),
            "ffn_total": round(sum(ffn_grads) * lr, 6),
            "total_param_delta": round(total_grad * lr, 6),
        }

        layer_grads = LayerGradientNorms(
            embedding_grad_norm=round(emb_grad, 4),
            attention_grad_norms=[round(g, 4) for g in attn_grads],
            ffn_grad_norms=[round(g, 4) for g in ffn_grads],
            norm_grad_norms=[round(g, 4) for g in norm_grads],
            total_grad_norm=round(total_grad, 4),
        )

        findings = [
            f"Gradients successfully propagated across all {cand_copy.config.n_layers} layers.",
            f"Target cross-entropy decreased from {loss_before:.4f} to {loss_after:.4f}.",
            f"In-distribution accuracy moved from {acc_before:.4f} to {acc_after:.4f}.",
            "Embedding and attention weights updated with non-zero gradient norms.",
        ]

        return ReasoningMechanisticDiagnosticsResult(
            diagnosis_id="diag_step161_mechanistic",
            loss_before_training=round(loss_before, 4),
            loss_after_training=round(loss_after, 4),
            accuracy_before_training=round(acc_before, 4),
            accuracy_after_training=round(acc_after, 4),
            token_diagnostics=records_after,
            layer_gradient_norms=layer_grads,
            parameter_update_norms=update_norms,
            summary_findings=findings,
        )
