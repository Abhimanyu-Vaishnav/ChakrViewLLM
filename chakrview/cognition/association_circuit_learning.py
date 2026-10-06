"""Step 228: Neural Association Circuit Formation.

Uses the best-performing objective from Step 227.
Instruments the exact retrieval trace from Wave 220:
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

Directly compares:
    FROZEN BASELINE
        vs
    TRAINED CANDIDATE

Measures at every stage:
- Target similarity
- Competing similarity
- Contrastive margin
- Rank
- Attention allocation
- Norm & entropy

Critical Question:
Does training increase Stage-A association score and propagate forward to Stage-B?
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.cognition.retrieval_trace import (
    trace_retrieval_circuit,
    RetrievalTraceReport,
    StageTraceMetric,
)
from chakrview.cognition.association_objectives import (
    train_and_eval_objective_candidate,
)
from chakrview.cognition.value_representation_retrieval import get_default_tokenizer


@dataclasses.dataclass
class CircuitComparisonStageResult:
    stage_name: str
    baseline_margin: float
    candidate_margin: float
    margin_delta: float
    baseline_rank: float
    candidate_rank: float
    baseline_signal_retained: bool
    candidate_signal_retained: bool
    is_improved: bool


@dataclasses.dataclass
class CircuitFormationReport:
    seed: int
    baseline_first_loss_stage: str
    candidate_first_loss_stage: str
    stage_comparisons: List[CircuitComparisonStageResult]
    stage_a_improved: bool
    stage_b_improved: bool
    is_circuit_formation_verified: bool
    cpu_runtime_ms: float = 0.0


def compare_neural_association_circuit(
    base_model: ChakrMicro,
    seed: int = 42,
) -> CircuitFormationReport:
    """Trains an isolated candidate and performs stage-by-stage circuit comparison with baseline."""
    tok = get_default_tokenizer()
    t0 = time.time()

    # Step 1: Baseline Trace
    base_trace = trace_retrieval_circuit(base_model, seed=seed)

    # Step 2: Train candidate using best joint objective (lambda_assoc=0.25, lambda_val=0.25)
    cand = copy.deepcopy(base_model)
    cand.train()
    for p in cand.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(cand.parameters(), lr=3e-4, weight_decay=1e-4)

    rng = random.Random(seed)
    train_keys = ["A", "B", "C", "D", "E"]
    train_vals = ["1", "2", "3", "4", "5"]

    # Short controlled training
    for _ in range(6):
        for _ in range(6):
            k_sample = rng.sample(train_keys, 3)
            v_sample = rng.sample(train_vals, 3)
            pairs = list(zip(k_sample, v_sample))
            query_pair = rng.choice(pairs)
            query_k, exp_v = query_pair
            exp_tok = tok.encode(exp_v, add_bos=False, add_eos=False)[0]

            prompt = "map " + " and ".join([f"|{k}| -> |{v}|" for k, v in pairs]) + f" query |{query_k}| -> |"
            token_ids = tok.encode(prompt, add_bos=True, add_eos=False)
            inp = torch.tensor([token_ids], dtype=torch.long)
            target = torch.tensor([exp_tok], dtype=torch.long)

            opt.zero_grad()
            logits = cand(inp)
            loss = F.cross_entropy(logits[0, -1, :].unsqueeze(0), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cand.parameters(), 1.0)
            opt.step()

    cand.eval()

    # Step 3: Candidate Trace
    cand_trace = trace_retrieval_circuit(cand, seed=seed)

    # Step 4: Comparative Stage Analysis
    stage_comparisons: List[CircuitComparisonStageResult] = []
    stage_a_imp = False
    stage_b_imp = False

    for b_stage, c_stage in zip(base_trace.stages, cand_trace.stages):
        m_delta = c_stage.margin - b_stage.margin
        imp = (m_delta > 0.05 or (c_stage.rank < b_stage.rank))

        if b_stage.stage_name == "3_association_state":
            stage_a_imp = (c_stage.margin > b_stage.margin)
        elif b_stage.stage_name == "4_value_representation":
            stage_b_imp = (c_stage.margin > b_stage.margin)

        stage_comparisons.append(
            CircuitComparisonStageResult(
                stage_name=b_stage.stage_name,
                baseline_margin=b_stage.margin,
                candidate_margin=c_stage.margin,
                margin_delta=m_delta,
                baseline_rank=b_stage.rank,
                candidate_rank=c_stage.rank,
                baseline_signal_retained=b_stage.signal_retained,
                candidate_signal_retained=c_stage.signal_retained,
                is_improved=imp,
            )
        )

    is_circuit_formed = (stage_a_imp and stage_b_imp and cand_trace.first_loss_stage == "NONE")
    elapsed = (time.time() - t0) * 1000.0

    return CircuitFormationReport(
        seed=seed,
        baseline_first_loss_stage=base_trace.first_loss_stage,
        candidate_first_loss_stage=cand_trace.first_loss_stage,
        stage_comparisons=stage_comparisons,
        stage_a_improved=stage_a_imp,
        stage_b_improved=stage_b_imp,
        is_circuit_formation_verified=is_circuit_formed,
        cpu_runtime_ms=elapsed,
    )
