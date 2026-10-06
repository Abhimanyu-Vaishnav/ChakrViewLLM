"""
ChakrView Step 145: Neural Learning Observability & Baseline/Candidate Protocol.

Features:
- NeuralLearningExperiment: Governed contract capturing full experiment lineage, parameters, losses,
  and multi-dimensional before/after capability scores.
- CandidateIsolationManager: Ensures candidate model is copied from baseline, trained in isolation,
  evaluated against identical conditions, and never mutates the canonical baseline.
- Verifies candidate != baseline while confirming ΔW_baseline ≡ 0.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)


@dataclass
class NeuralLearningExperiment:
    experiment_id: str
    parent_model_id: str
    baseline_hash: str
    candidate_hash: str
    dataset_identity: str
    dataset_sha256: str
    tokenizer_identity: str
    tokenizer_sha256: str
    curriculum_identity: str
    seed: int
    optimizer_identity: str
    learning_rate: float
    batch_size: int
    sequence_length: int
    training_steps: int
    training_device: str
    parameter_count: int
    train_loss: float
    validation_loss: float
    held_out_score: float
    reasoning_score: float
    language_score: float
    generalization_score: float
    regression_score: float
    final_decision: str  # "PROMOTED_CANDIDATE", "REJECTED_CANDIDATE", "ROLLBACK"
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NeuralLearningExperiment:
        return cls(**data)


class CandidateIsolationManager:
    """
    Manages safe neural candidate generation, training isolation, and baseline integrity verification.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path
        if self.db_path:
            self._init_tables()

    def _init_tables(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS neural_learning_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    parent_model_id TEXT,
                    baseline_hash TEXT,
                    candidate_hash TEXT,
                    dataset_identity TEXT,
                    dataset_sha256 TEXT,
                    tokenizer_identity TEXT,
                    tokenizer_sha256 TEXT,
                    curriculum_identity TEXT,
                    seed INTEGER,
                    optimizer_identity TEXT,
                    learning_rate REAL,
                    batch_size INTEGER,
                    sequence_length INTEGER,
                    training_steps INTEGER,
                    training_device TEXT,
                    parameter_count INTEGER,
                    train_loss REAL,
                    validation_loss REAL,
                    held_out_score REAL,
                    reasoning_score REAL,
                    language_score REAL,
                    generalization_score REAL,
                    regression_score REAL,
                    final_decision TEXT,
                    created_at REAL,
                    record_json TEXT
                )
            """)
            conn.commit()

    def create_isolated_candidate(self) -> Tuple[ChakrMicro, str]:
        """
        Instantiates an isolated candidate initialized from the frozen baseline.
        Verifies baseline integrity during creation.
        """
        baseline = instantiate_frozen_baseline()
        base_hash = compute_model_hash(baseline)
        if base_hash != EXPECTED_WEIGHT_HASH:
            raise RuntimeError(f"Cannot branch candidate: baseline compromised ({base_hash})")

        # Deepcopy to create an isolated mutable candidate
        candidate = copy.deepcopy(baseline)
        for p in candidate.parameters():
            p.requires_grad = True

        return candidate, base_hash

    def verify_candidate_integrity(
        self,
        baseline: ChakrMicro,
        candidate: ChakrMicro,
    ) -> Tuple[bool, bool, str, str]:
        """
        Verifies:
        1. Baseline has NOT changed (hash == EXPECTED_WEIGHT_HASH)
        2. Candidate is different from baseline (candidate_hash != base_hash)
        """
        base_hash = compute_model_hash(baseline)
        cand_hash = compute_model_hash(candidate)

        baseline_intact = (base_hash == EXPECTED_WEIGHT_HASH)
        candidate_diverged = (cand_hash != base_hash)

        return baseline_intact, candidate_diverged, base_hash, cand_hash

    def record_experiment(self, exp: NeuralLearningExperiment) -> None:
        if not self.db_path:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO neural_learning_experiments
                (experiment_id, parent_model_id, baseline_hash, candidate_hash, dataset_identity, dataset_sha256,
                 tokenizer_identity, tokenizer_sha256, curriculum_identity, seed, optimizer_identity, learning_rate,
                 batch_size, sequence_length, training_steps, training_device, parameter_count, train_loss,
                 validation_loss, held_out_score, reasoning_score, language_score, generalization_score,
                 regression_score, final_decision, created_at, record_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp.experiment_id,
                exp.parent_model_id,
                exp.baseline_hash,
                exp.candidate_hash,
                exp.dataset_identity,
                exp.dataset_sha256,
                exp.tokenizer_identity,
                exp.tokenizer_sha256,
                exp.curriculum_identity,
                exp.seed,
                exp.optimizer_identity,
                exp.learning_rate,
                exp.batch_size,
                exp.sequence_length,
                exp.training_steps,
                exp.training_device,
                exp.parameter_count,
                exp.train_loss,
                exp.validation_loss,
                exp.held_out_score,
                exp.reasoning_score,
                exp.language_score,
                exp.generalization_score,
                exp.regression_score,
                exp.final_decision,
                exp.created_at,
                json.dumps(exp.to_dict()),
            ))
            conn.commit()
