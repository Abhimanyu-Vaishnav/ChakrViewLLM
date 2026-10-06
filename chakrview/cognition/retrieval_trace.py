"""Step 220: Query -> Association -> Value Trace.

Instruments the internal retrieval path:
    query state
         ↓
    key matching
         ↓
    association state
         ↓
    value representation
         ↓
    output representation
         ↓
    logits

Measures at every stage:
- cosine to expected representation
- cosine to incorrect representations
- rank among candidate tokens / representations
- norm of the representation
- entropy of attention/distribution where applicable
- attention allocation across keys and values
- top competing representations

Identifies the FIRST point at which the correct signal is lost.
Produces a machine-readable trace artifact.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.cognition.value_representation_retrieval import (
    extract_contextual_representations,
    compute_representation_retrieval_metrics,
    get_default_tokenizer,
)


@dataclasses.dataclass
class StageTraceMetric:
    stage_name: str
    cosine_expected: float
    cosine_incorrect_mean: float
    margin: float
    rank: float
    norm: float
    entropy: float
    signal_retained: bool
    details: Dict[str, Any]


@dataclasses.dataclass
class RetrievalTraceReport:
    seed: int
    prompt: str
    query_key: str
    expected_value: str
    stages: List[StageTraceMetric]
    first_loss_stage: str
    cpu_runtime_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "prompt": self.prompt,
            "query_key": self.query_key,
            "expected_value": self.expected_value,
            "first_loss_stage": self.first_loss_stage,
            "cpu_runtime_ms": self.cpu_runtime_ms,
            "stages": [
                {
                    "stage_name": s.stage_name,
                    "cosine_expected": s.cosine_expected,
                    "cosine_incorrect_mean": s.cosine_incorrect_mean,
                    "margin": s.margin,
                    "rank": s.rank,
                    "norm": s.norm,
                    "entropy": s.entropy,
                    "signal_retained": s.signal_retained,
                    "details": s.details,
                }
                for s in self.stages
            ],
        }


def trace_retrieval_circuit(
    model: ChakrMicro,
    seed: int = 42,
    prompt: Optional[str] = None,
    pairs: Optional[List[Tuple[str, str]]] = None,
    query_key: Optional[str] = None,
    expected_val: Optional[str] = None,
) -> RetrievalTraceReport:
    """Traces the full 6-stage chain from query state to logits."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = get_default_tokenizer()
    model.eval()

    if pairs is None:
        pairs = [("P", "7"), ("Q", "8"), ("R", "9")]
    if query_key is None:
        query_key = pairs[0][0]
        expected_val = pairs[0][1]

    if prompt is None:
        parts = [f"|{k}| -> |{v}|" for k, v in pairs]
        prompt = "map " + " and ".join(parts) + f" query |{query_key}| -> |"

    t0 = time.time()
    token_ids = tok.encode(prompt, add_bos=True, add_eos=False)
    inp = torch.tensor([token_ids], dtype=torch.long)
    cos = nn.CosineSimilarity(dim=0)

    # Instrument forward pass
    with torch.no_grad():
        x = model.embedding(inp)
        for i, layer in enumerate(model.layers):
            x = layer(x, attention_mask=None, kv_cache=None, layer_idx=i)
        hidden = model.final_norm(x)[0]  # [T, d_model]
        logits = model.lm_head(hidden.unsqueeze(0))[0] # [T, vocab]

    extracted = extract_contextual_representations(
        model=model,
        tokenizer=tok,
        prompt=prompt,
        pairs=pairs,
        query_key=query_key,
    )

    k_reps = extracted["k_reps"]
    v_reps = extracted["v_reps"]
    h_query_key = extracted["query_key"]
    h_terminal = extracted["terminal_query"]

    stages: List[StageTraceMetric] = []
    first_loss = "NONE"

    # Stage 1: Query state (at query key position vs known keys)
    q_norm = float(torch.norm(h_query_key).item())
    key_sims = {k: float(cos(h_query_key, k_rep).item()) for k, k_rep in k_reps.items()}
    target_k_sim = key_sims.get(query_key, 0.0)
    incorr_k_sims = [s for k, s in key_sims.items() if k != query_key]
    avg_incorr_k = float(sum(incorr_k_sims) / len(incorr_k_sims)) if incorr_k_sims else 0.0
    margin_k = target_k_sim - max(incorr_k_sims) if incorr_k_sims else 0.0
    k_rank = 1 if all(target_k_sim >= s for s in incorr_k_sims) else 2
    s1_ok = (k_rank == 1 and margin_k > 0.0)

    stages.append(
        StageTraceMetric(
            stage_name="1_query_state",
            cosine_expected=target_k_sim,
            cosine_incorrect_mean=avg_incorr_k,
            margin=margin_k,
            rank=float(k_rank),
            norm=q_norm,
            entropy=0.0,
            signal_retained=s1_ok,
            details={"key_sims": key_sims},
        )
    )
    if not s1_ok and first_loss == "NONE":
        first_loss = "1_query_state"

    # Stage 2: Key matching
    # Softmax attention across candidate keys based on dot-product with query
    dot_scores = torch.tensor([torch.dot(h_query_key, k_reps[k]).item() for k, _ in pairs])
    attn_weights = F.softmax(dot_scores / (model.config.d_model ** 0.5), dim=0)
    ent_attn = float(-torch.sum(attn_weights * torch.log(attn_weights + 1e-12)).item())
    target_idx = [k for k, _ in pairs].index(query_key)
    target_attn = float(attn_weights[target_idx].item())
    max_other_attn = float(max([attn_weights[i].item() for i in range(len(pairs)) if i != target_idx]))
    s2_ok = (target_attn > max_other_attn)

    stages.append(
        StageTraceMetric(
            stage_name="2_key_matching",
            cosine_expected=target_attn,
            cosine_incorrect_mean=float(sum([attn_weights[i].item() for i in range(len(pairs)) if i != target_idx]) / (len(pairs) - 1)),
            margin=target_attn - max_other_attn,
            rank=1.0 if s2_ok else 2.0,
            norm=float(torch.norm(attn_weights).item()),
            entropy=ent_attn,
            signal_retained=s2_ok,
            details={"attn_distribution": attn_weights.tolist()},
        )
    )
    if not s2_ok and first_loss == "NONE":
        first_loss = "2_key_matching"

    # Stage 3: Association state
    pos_assoc = k_reps[query_key] * v_reps[expected_val]
    assoc_norm = float(torch.norm(pos_assoc).item())
    neg_sims = [float(cos(pos_assoc, k_reps[query_key] * v_reps[v]).item()) for _, v in pairs if v != expected_val]
    mean_neg = float(sum(neg_sims) / len(neg_sims)) if neg_sims else 0.0
    assoc_margin = 1.0 - mean_neg
    s3_ok = (assoc_margin >= 0.50)

    stages.append(
        StageTraceMetric(
            stage_name="3_association_state",
            cosine_expected=1.0,
            cosine_incorrect_mean=mean_neg,
            margin=assoc_margin,
            rank=1.0 if s3_ok else 2.0,
            norm=assoc_norm,
            entropy=0.0,
            signal_retained=s3_ok,
            details={"neg_assoc_sims": neg_sims},
        )
    )
    if not s3_ok and first_loss == "NONE":
        first_loss = "3_association_state"

    # Stage 4: Value representation retrieval
    # Evaluated at the terminal query prompt state h_terminal against candidate value vectors
    v_sim_corr, v_sim_incorr, v_margin, v_rank = compute_representation_retrieval_metrics(
        retrieved_rep=h_terminal,
        correct_value_rep=v_reps[expected_val],
        candidate_value_reps=v_reps,
        correct_val_key=expected_val,
    )
    s4_ok = (v_margin > 0.0 and v_rank < 2.0)

    stages.append(
        StageTraceMetric(
            stage_name="4_value_representation",
            cosine_expected=v_sim_corr,
            cosine_incorrect_mean=v_sim_incorr,
            margin=v_margin,
            rank=float(v_rank),
            norm=float(torch.norm(h_terminal).item()),
            entropy=0.0,
            signal_retained=s4_ok,
            details={"rank": v_rank},
        )
    )
    if not s4_ok and first_loss == "NONE":
        first_loss = "4_value_representation"

    # Stage 5: Output representation (pre-logit norm)
    out_norm = float(torch.norm(hidden[-1]).item())
    s5_ok = s4_ok  # Output representation inherits Stage 4 state

    stages.append(
        StageTraceMetric(
            stage_name="5_output_representation",
            cosine_expected=v_sim_corr,
            cosine_incorrect_mean=v_sim_incorr,
            margin=v_margin,
            rank=float(v_rank),
            norm=out_norm,
            entropy=0.0,
            signal_retained=s5_ok,
            details={"inherited_stage4": True},
        )
    )
    if not s5_ok and first_loss == "NONE":
        first_loss = "5_output_representation"

    # Stage 6: Vocabulary / Logits
    last_logits = logits[-1]
    exp_tok = tok.encode(expected_val, add_bos=False, add_eos=False)[0]
    p = F.softmax(last_logits, dim=-1)
    tok_prob = float(p[exp_tok].item())
    tok_ent = float(-torch.sum(p * torch.log(p + 1e-12)).item())
    sorted_idx = torch.argsort(last_logits, descending=True)
    tok_rank = int((sorted_idx == exp_tok).nonzero(as_tuple=True)[0].item()) + 1
    pred_tok = torch.argmax(last_logits).item()
    s6_ok = (pred_tok == exp_tok)

    stages.append(
        StageTraceMetric(
            stage_name="6_logits_output",
            cosine_expected=tok_prob,
            cosine_incorrect_mean=float((1.0 - tok_prob) / (p.numel() - 1)),
            margin=tok_prob - float(torch.topk(p, 2).values[1].item() if pred_tok == exp_tok else torch.max(p).item()),
            rank=float(tok_rank),
            norm=float(torch.norm(last_logits).item()),
            entropy=tok_ent,
            signal_retained=s6_ok,
            details={"pred_token_id": pred_tok, "expected_token_id": exp_tok, "rank": tok_rank},
        )
    )
    if not s6_ok and first_loss == "NONE":
        first_loss = "6_logits_output"

    cpu_ms = (time.time() - t0) * 1000.0

    return RetrievalTraceReport(
        seed=seed,
        prompt=prompt,
        query_key=query_key,
        expected_value=expected_val,
        stages=stages,
        first_loss_stage=first_loss,
        cpu_runtime_ms=cpu_ms,
    )
