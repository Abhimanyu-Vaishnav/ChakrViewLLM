"""Step 496: Governed Meta-Cognitive Self-Evaluator Implementation."""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Dict, List, Optional, Any
import torch


class ReasoningStrategy(str, Enum):
    STRATEGY_DIRECT = "STRATEGY_DIRECT"
    STRATEGY_RECURSIVE = "STRATEGY_RECURSIVE"
    STRATEGY_MEMORY = "STRATEGY_MEMORY"
    STRATEGY_DEEPER_COMPUTATION = "STRATEGY_DEEPER_COMPUTATION"
    STRATEGY_RETRY = "STRATEGY_RETRY"
    STRATEGY_PEER_REQUEST = "STRATEGY_PEER_REQUEST"


@dataclasses.dataclass
class MetaCognitiveAssessment:
    current_state_norm: float
    confidence: float
    uncertainty: float
    sufficiency: bool
    failure_risk: float
    recommended_strategy: ReasoningStrategy
    rationale: str
    expected_value_compute: float
    expected_value_memory: float
    expected_value_peer: float

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d["recommended_strategy"] = self.recommended_strategy.value
        return d


class MetaCognitiveEvaluator:
    """
    Evaluates current reasoning trajectory and produces a governed assessment.
    Does NOT self-authorize resources.
    """
    def __init__(self, confidence_threshold: float = 0.75, uncertainty_threshold: float = 0.35):
        self.conf_thresh = confidence_threshold
        self.unc_thresh = uncertainty_threshold

    def evaluate(
        self,
        cycle_count: int,
        state_tensor: torch.Tensor,
        step_delta: float,
        epistemic_status: str,
        margin: float = 0.8,
        has_contradiction: bool = False,
    ) -> MetaCognitiveAssessment:
        s_norm = torch.norm(state_tensor).item() if torch.is_tensor(state_tensor) else float(state_tensor)
        
        # Calibration modeling
        if epistemic_status == "UNKNOWN":
            confidence = 0.20
            uncertainty = 0.80
            failure_risk = 0.90
            strategy = ReasoningStrategy.STRATEGY_MEMORY
            rationale = "Proposition unknown; querying persistent cognitive memory"
        elif epistemic_status == "CONFLICTING" or has_contradiction:
            confidence = 0.30
            uncertainty = 0.85
            failure_risk = 0.95
            strategy = ReasoningStrategy.STRATEGY_PEER_REQUEST
            rationale = "Contradictory evidence detected; requesting peer verification"
        elif cycle_count >= 4 and step_delta < 0.04:
            # State saturation / degradation condition discovered in Wave 483
            confidence = 0.55
            uncertainty = 0.50
            failure_risk = 0.65
            strategy = ReasoningStrategy.STRATEGY_DEEPER_COMPUTATION
            rationale = "Step delta attenuated at deep cycle; recommending deeper computation or retry"
        elif cycle_count == 1 and margin > 0.7:
            confidence = 0.95
            uncertainty = 0.08
            failure_risk = 0.05
            strategy = ReasoningStrategy.STRATEGY_DIRECT
            rationale = "High margin single-hop resolution; execution sufficient"
        else:
            confidence = 0.85
            uncertainty = 0.18
            failure_risk = 0.15
            strategy = ReasoningStrategy.STRATEGY_RECURSIVE
            rationale = "Multi-hop progression normal; standard recursive update"

        sufficiency = (confidence >= self.conf_thresh) and (uncertainty <= self.unc_thresh) and (failure_risk <= 0.30)
        
        ev_compute = max(0.0, failure_risk * 0.8 - 0.1)
        ev_memory = 0.9 if epistemic_status in ["UNKNOWN", "UNCERTAIN"] else 0.1
        ev_peer = 0.85 if has_contradiction or failure_risk > 0.6 else 0.2

        return MetaCognitiveAssessment(
            current_state_norm=round(s_norm, 4),
            confidence=round(confidence, 4),
            uncertainty=round(uncertainty, 4),
            sufficiency=sufficiency,
            failure_risk=round(failure_risk, 4),
            recommended_strategy=strategy,
            rationale=rationale,
            expected_value_compute=round(ev_compute, 4),
            expected_value_memory=round(ev_memory, 4),
            expected_value_peer=round(ev_peer, 4)
        )
