"""Step 380: Adaptive Computation Objective & Computation/Accuracy Pareto Study.

Introduces and evaluates computation cost penalty:
total_loss = reasoning_loss + lambda_compute * computation_cost

Evaluates controlled values:
- lambda = 0.0 (unconstrained compute)
- lambda = 0.02 (small penalty)
- lambda = 0.08 (medium penalty)
- lambda = 0.25 (aggressive penalty)

Measures the Pareto tradeoff:
- Final reasoning accuracy
- Average reasoning cycles executed
- Premature halt rate
- Unnecessary continuation rate
- Efficiency / Pareto frontier analysis
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.cognition.adaptive_recurrent_reasoning_core import AdaptiveRecurrentReasoningCore
from chakrview.cognition.compositional_benchmark_environment import (
    CompositionalAssociativeEnvironment,
)
from chakrview.cognition.adaptive_learnability import (
    generate_mixed_hop_episode,
    train_and_eval_adaptive_learnability,
)


@dataclasses.dataclass
class ParetoTradeoffPoint:
    lambda_compute: float
    accuracy: float
    mean_cycles: float
    g4_acc: float
    premature_halt_rate: float
    unnecessary_rate: float
    pareto_score: float  # accuracy / (1 + 0.1 * mean_cycles)


@dataclasses.dataclass
class Step380ParetoReport:
    points: Dict[float, ParetoTradeoffPoint]
    best_lambda: float
    unconstrained_acc: float
    penalized_acc: float
    cycle_reduction_pct: float
    summary: str


def run_adaptive_computation_objective_study(
    seed: int = 42,
    train_steps: int = 20,
    eval_episodes: int = 6,
) -> Step380ParetoReport:
    """Executes Step 380 computation penalty tradeoff sweep."""
    torch.manual_seed(seed)
    lambdas = [0.0, 0.02, 0.08, 0.25]
    points: Dict[float, ParetoTradeoffPoint] = {}

    for lam in lambdas:
        core = AdaptiveRecurrentReasoningCore()
        _, rep_learn = train_and_eval_adaptive_learnability(
            core=core,
            seed=seed,
            train_steps=train_steps,
            eval_episodes_per_hop=eval_episodes,
            lambda_compute=lam,
        )

        acc = rep_learn.overall_accuracy
        m_c = rep_learn.mean_cycles_overall
        g4 = rep_learn.metrics_by_hop[2].accuracy  # 2-hop compositional proxy
        prem = sum(m.premature_halts for m in rep_learn.metrics_by_hop.values()) / 3.0
        unnec = sum(m.unnecessary_cycles for m in rep_learn.metrics_by_hop.values()) / 3.0
        score = acc / (1.0 + 0.1 * m_c)

        points[lam] = ParetoTradeoffPoint(
            lambda_compute=lam,
            accuracy=acc,
            mean_cycles=m_c,
            g4_acc=g4,
            premature_halt_rate=prem,
            unnecessary_rate=unnec,
            pareto_score=score,
        )

    best_lam = max(points.keys(), key=lambda k: points[k].pareto_score)
    u_acc = points[0.0].accuracy
    p_acc = points[best_lam].accuracy
    u_c = points[0.0].mean_cycles
    p_c = points[best_lam].mean_cycles
    reduction = ((u_c - p_c) / max(0.01, u_c)) * 100.0 if u_c > 0 else 0.0

    summary = (
        f"Computation Objective Pareto Study: Best lambda={best_lam} (Pareto Score={points[best_lam].pareto_score:.4f}). "
        f"Unconstrained Acc={u_acc:.2%} (AvgC={u_c:.2f}) vs Penalized Acc={p_acc:.2%} (AvgC={p_c:.2f}, "
        f"Cycle Reduction={reduction:+.1f}%)."
    )

    return Step380ParetoReport(
        points=points,
        best_lambda=best_lam,
        unconstrained_acc=u_acc,
        penalized_acc=p_acc,
        cycle_reduction_pct=reduction,
        summary=summary,
    )


if __name__ == "__main__":
    print("Step 380: Evaluating Adaptive Computation Objective Tradeoff...")
    p_rep = run_adaptive_computation_objective_study(seed=42, train_steps=15, eval_episodes=4)
    print("Report Summary:", p_rep.summary)
    for lam, pt in p_rep.points.items():
        print(f"  Lambda={lam:5.2f} -> Acc={pt.accuracy:.2%}, AvgC={pt.mean_cycles:.2f}, G4={pt.g4_acc:.2%}, Score={pt.pareto_score:.4f}")
