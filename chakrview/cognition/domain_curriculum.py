"""
ChakrView Step 138: Domain Curriculum Engine.

Orchestrates multi-stage domain curricula:
FOUNDATION -> DOMAIN INTRODUCTION -> DOMAIN STRUCTURE -> DOMAIN REASONING -> DOMAIN APPLICATION -> EVALUATION -> GENERALIZATION

Tracks:
- curriculum_id, domain_id, dataset_identity, tokenizer_identity
- staged data samples with deterministic seeds
- held-out validation isolation
- candidate checkpoints and rollback manifests
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class CurriculumStage(str, enum.Enum):
    FOUNDATION = "FOUNDATION"
    DOMAIN_INTRO = "DOMAIN_INTRO"
    DOMAIN_STRUCTURE = "DOMAIN_STRUCTURE"
    DOMAIN_REASONING = "DOMAIN_REASONING"
    DOMAIN_APPLICATION = "DOMAIN_APPLICATION"
    EVALUATION = "EVALUATION"
    GENERALIZATION = "GENERALIZATION"


@dataclass
class DomainCurriculumSample:
    sample_id: str
    domain_id: str
    stage: CurriculumStage
    input_text: str
    target_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stage"] = self.stage.value
        return d


@dataclass
class CurriculumExecutionManifest:
    curriculum_id: str
    domain_id: str
    stages: List[CurriculumStage]
    dataset_checksum: str
    tokenizer_checksum: str
    seed: int
    candidate_checkpoint_id: str
    baseline_hash: str
    total_samples: int
    metrics_summary: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stages"] = [s.value for s in self.stages]
        return d


class DomainCurriculumEngine:
    """
    Manages structured curriculum progression and candidate checkpoint tracking.
    """

    def __init__(self, curriculum_id: str, domain_id: str, seed: int = 42) -> None:
        self.curriculum_id = curriculum_id
        self.domain_id = domain_id
        self.seed = seed
        self.stage_samples: Dict[CurriculumStage, List[DomainCurriculumSample]] = {
            s: [] for s in CurriculumStage
        }
        self.held_out_samples: List[DomainCurriculumSample] = []

    def add_sample(self, sample: DomainCurriculumSample, is_held_out: bool = False) -> None:
        if is_held_out:
            self.held_out_samples.append(sample)
        else:
            self.stage_samples[sample.stage].append(sample)

    def compute_dataset_checksum(self) -> str:
        hasher = hashlib.sha256()
        for stage in CurriculumStage:
            for s in self.stage_samples[stage]:
                hasher.update(s.input_text.encode("utf-8"))
                hasher.update(s.target_text.encode("utf-8"))
        for s in self.held_out_samples:
            hasher.update(s.input_text.encode("utf-8"))
            hasher.update(s.target_text.encode("utf-8"))
        return hasher.hexdigest()

    def get_stage_batches(self, stage: CurriculumStage, batch_size: int = 4) -> List[List[DomainCurriculumSample]]:
        samples = self.stage_samples.get(stage, [])
        return [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]

    def create_execution_manifest(
        self,
        candidate_checkpoint_id: str,
        baseline_hash: str,
        tokenizer_checksum: str,
        metrics: Optional[Dict[str, float]] = None,
    ) -> CurriculumExecutionManifest:
        total = sum(len(samples) for samples in self.stage_samples.values()) + len(self.held_out_samples)
        return CurriculumExecutionManifest(
            curriculum_id=self.curriculum_id,
            domain_id=self.domain_id,
            stages=list(CurriculumStage),
            dataset_checksum=self.compute_dataset_checksum(),
            tokenizer_checksum=tokenizer_checksum,
            seed=self.seed,
            candidate_checkpoint_id=candidate_checkpoint_id,
            baseline_hash=baseline_hash,
            total_samples=total,
            metrics_summary=metrics or {},
        )
