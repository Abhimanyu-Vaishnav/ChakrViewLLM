"""
ChakrView Step 126: Neural / Cognitive Integration Boundary.

Defines the exact architectural boundary between the neural core (ChakrMicro)
and higher-level cognitive layers:
ChakrMicro (Inference/Representation) -> NeuralCognitiveBridge -> Cognitive Layer / Tools

Critical Invariant:
Neural core has ZERO tool authority. Tool execution is exclusively governed by GovernedToolGate.
Model classes: CANONICAL_BASELINE, CANDIDATE_CHECKPOINT, EVALUATION_CHECKPOINT.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)


class ModelTier(str, enum.Enum):
    CANONICAL_BASELINE = "CANONICAL_BASELINE"
    CANDIDATE_CHECKPOINT = "CANDIDATE_CHECKPOINT"
    EVALUATION_CHECKPOINT = "EVALUATION_CHECKPOINT"


@dataclass
class NeuralCognitiveInferenceResult:
    predicted_tokens: List[int]
    confidence_score: float
    representation_embedding: Optional[List[float]] = None


class NeuralCognitiveBridge:
    """
    Governed bridge allowing cognitive workers to request neural representations,
    pattern scoring, or candidate rankings from ChakrMicro without granting the neural model
    direct execution or filesystem access.
    """

    def __init__(self, model_tier: ModelTier = ModelTier.CANONICAL_BASELINE) -> None:
        self.model_tier = model_tier
        self._model: Optional[ChakrMicro] = None

    def _ensure_model(self) -> ChakrMicro:
        if self._model is None:
            self._model = instantiate_frozen_baseline()
            # Verify immutable hash
            h = compute_model_hash(self._model)
            assert h == EXPECTED_WEIGHT_HASH, "Canonical baseline mutated!"
        return self._model

    def score_sequence_perplexity(self, input_ids: List[int]) -> float:
        """Computes sequence score without mutating weights or invoking tools."""
        model = self._ensure_model()
        model.eval()
        with torch.no_grad():
            inp = torch.tensor([input_ids[:128]], dtype=torch.long)
            logits = model(inp)
            # Pseudo-perplexity proxy
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), inp[:, 1:].reshape(-1))
            return float(torch.exp(loss).item())


    def verify_baseline_immutability(self) -> bool:
        model = self._ensure_model()
        return compute_model_hash(model) == EXPECTED_WEIGHT_HASH
