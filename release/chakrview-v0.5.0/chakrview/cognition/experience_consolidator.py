"""
Experience Consolidator.
Wave 521: Governed experience-consolidation mechanism.
Takes raw EpisodicExperience records and synthesizes AbstractExperience records
by detecting recurring structural patterns, successful/unsuccessful strategies,
and causal failure modes across multiple distinct episodes.

Guarantees:
- Does NOT concatenate raw texts.
- Does NOT create knowledge items from a single isolated observation.
- Discovers task families, computes consensus strategy utility, and flags contradictions.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import hashlib

from chakrview.cognition.adaptive_learning_contract import (
    EpisodicExperience,
    AbstractExperience,
    ProvenanceMetadata,
    AdaptiveLearningContract
)

class ExperienceConsolidator:
    def __init__(self, min_episodes_for_abstraction: int = 2):
        self.min_episodes = min_episodes_for_abstraction
        self.episode_buffer: List[EpisodicExperience] = []
        self.abstractions: Dict[str, AbstractExperience] = {}

    def ingest_episode(self, episode: EpisodicExperience) -> None:
        """Ingest a validated episodic experience."""
        AdaptiveLearningContract.verify_artifact_isolation(episode)
        self.episode_buffer.append(episode)

    def consolidate(self) -> List[AbstractExperience]:
        """
        Group episodes by task family/structure and derive abstractions.
        Only produces an AbstractExperience if >= min_episodes exist for that family.
        """
        family_groups = defaultdict(list)
        for ep in self.episode_buffer:
            family = ep.task_structure.get("task_family", "generic_relational")
            family_groups[family].append(ep)

        consolidated = []
        for family, episodes in family_groups.items():
            if len(episodes) < self.min_episodes:
                continue

            successes = [ep for ep in episodes if ep.success]
            failures = [ep for ep in episodes if not ep.success]

            # Analyze strategy success rates
            strat_counts = defaultdict(lambda: {"success": 0, "fail": 0})
            for ep in episodes:
                if ep.success:
                    strat_counts[ep.strategy_used]["success"] += 1
                else:
                    strat_counts[ep.strategy_used]["fail"] += 1

            best_strategy = "direct_retrieval"
            best_rate = -1.0
            for strat, counts in strat_counts.items():
                total = counts["success"] + counts["fail"]
                rate = counts["success"] / total if total > 0 else 0.0
                if rate > best_rate:
                    best_rate = rate
                    best_strategy = strat

            # Identify dominant failure cause
            failure_causes = defaultdict(int)
            for ep in failures:
                if ep.error_type:
                    failure_causes[ep.error_type] += 1
            dominant_failure = max(failure_causes.items(), key=lambda x: x[1])[0] if failure_causes else "none"

            success_pattern = f"Tasks of family '{family}' reliably succeed with strategy '{best_strategy}' (success rate: {best_rate:.2f})."
            failure_pattern = f"Recurring failure mode under low-step execution: '{dominant_failure}'."

            supporting_ids = [ep.episode_id for ep in successes]
            contradictory_ids = [ep.episode_id for ep in failures]

            # Calculate confidence
            confidence = len(successes) / len(episodes) if len(episodes) > 0 else 0.5

            abs_id = f"abs_{family}_{hashlib.sha256(family.encode()).hexdigest()[:8]}"
            abstract_exp = AbstractExperience(
                abstraction_id=abs_id,
                task_family=family,
                supporting_episode_ids=supporting_ids,
                contradictory_episode_ids=contradictory_ids,
                success_pattern=success_pattern,
                failure_pattern=failure_pattern,
                recommended_strategy=best_strategy,
                confidence=confidence,
                sample_count=len(episodes),
                provenance=ProvenanceMetadata()
            )
            AdaptiveLearningContract.verify_artifact_isolation(abstract_exp)
            self.abstractions[family] = abstract_exp
            consolidated.append(abstract_exp)

        return consolidated
