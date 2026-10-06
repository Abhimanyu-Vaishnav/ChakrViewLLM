"""
ChakrView Step 149: Neural Experience Learning Bridge.

Connects Step 143 Governed Experience Learning to actual candidate neural curriculum generation:
EXPERIENCE -> OBSERVE -> EVALUATE -> DIAGNOSE -> LESSON -> CURRICULUM PROPOSAL -> CANDIDATE TRAINING -> EVALUATION -> PROMOTE/ROLLBACK

Crucial Invariant:
No direct experience -> weights mutation.
A lesson triggers a reproducible synthetic curriculum proposal, which is then trained as an isolated candidate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chakrview.brain.model import ChakrMicro
from chakrview.cognition.experience_learning import (
    GovernedExperienceEpisode,
    ExperienceSignalOutcome,
    GovernedExperienceLearningEngine,
)
from chakrview.cognition.domain_curriculum import (
    DomainCurriculumSample,
    CurriculumStage,
)
from chakrview.cognition.neural_observability import CandidateIsolationManager


@dataclass
class ExperienceCurriculumProposal:
    proposal_id: str
    source_episode_id: str
    domain_id: str
    target_weakness: str
    proposed_samples: List[DomainCurriculumSample]
    sample_count: int
    validation_criteria: Dict[str, float]


class NeuralExperienceBridge:
    """
    Translates audited cognitive experience lessons into formal training curriculum proposals.
    """

    def __init__(self, experience_engine: GovernedExperienceLearningEngine) -> None:
        self.experience_engine = experience_engine

    def convert_episode_to_curriculum(
        self,
        episode: GovernedExperienceEpisode,
    ) -> ExperienceCurriculumProposal:
        """
        Synthesizes structured training samples specifically targeting the failure observed in the episode.
        """
        weakness = episode.diagnosis
        samples: List[DomainCurriculumSample] = []

        w_low = weakness.lower()
        if any(term in w_low for term in ["syntax", "not found", "does not exist", "defect", "error", "failed"]):
            # Generate boundary and syntactic validation samples
            samples.append(DomainCurriculumSample(
                sample_id=f"synth_{episode.episode_id}_01",
                domain_id=episode.domain_id,
                stage=CurriculumStage.DOMAIN_REASONING,
                input_text="def check_file_exists(path): if not path: raise FileNotFoundError",
                target_text="return verify_path(path)",
            ))
            samples.append(DomainCurriculumSample(
                sample_id=f"synth_{episode.episode_id}_02",
                domain_id=episode.domain_id,
                stage=CurriculumStage.DOMAIN_APPLICATION,
                input_text="try: open_file(path) except Exception:",
                target_text="return handle_safe_recovery()",
            ))
        else:
            samples.append(DomainCurriculumSample(
                sample_id=f"synth_{episode.episode_id}_generic",
                domain_id=episode.domain_id,
                stage=CurriculumStage.DOMAIN_INTRO,
                input_text=f"task_objective: {episode.task_id}",
                target_text=f"lesson: {episode.verified_lesson}",
            ))

        return ExperienceCurriculumProposal(
            proposal_id=f"prop_{episode.episode_id}",
            source_episode_id=episode.episode_id,
            domain_id=episode.domain_id,
            target_weakness=weakness,
            proposed_samples=samples,
            sample_count=len(samples),
            validation_criteria={"max_val_loss": 2.5, "min_heldout_acc": 0.50},
        )
