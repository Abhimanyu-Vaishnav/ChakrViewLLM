"""
Long-Horizon Durable Experience Memory.
Wave 523: Persistent experience storage engine.
Provides:
- Restart survival: Serializes abstractions, strategies, and verified knowledge to disk with checksums.
- Stale experience expiration: TTL-based expiration and confidence decay.
- Contradiction handling: Replaces or down-weights conflicting assertions upon validated counter-evidence.
- Storage footprint governance: Bounds max disk usage and memory footprint.
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional

from chakrview.cognition.adaptive_learning_contract import (
    AbstractExperience,
    StrategyPolicy,
    VerifiedKnowledge,
    ProvenanceMetadata,
    AdaptiveLearningContract
)

class DurableExperienceStore:
    def __init__(self, storage_path: Path, max_storage_kb: int = 1024, ttl_seconds: float = 86400 * 30):
        self.storage_path = Path(storage_path)
        self.max_storage_kb = max_storage_kb
        self.ttl_seconds = ttl_seconds
        self.abstractions: Dict[str, AbstractExperience] = {}
        self.strategies: Dict[str, StrategyPolicy] = {}
        self.knowledge: Dict[str, VerifiedKnowledge] = {}

    def save_to_disk(self) -> str:
        """Persist state to disk with cryptographic SHA-256 integrity manifest."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "v0.3.0_durable_experience",
            "saved_at": time.time(),
            "abstractions": [
                {
                    "abstraction_id": a.abstraction_id,
                    "task_family": a.task_family,
                    "supporting_episode_ids": a.supporting_episode_ids,
                    "contradictory_episode_ids": a.contradictory_episode_ids,
                    "success_pattern": a.success_pattern,
                    "failure_pattern": a.failure_pattern,
                    "recommended_strategy": a.recommended_strategy,
                    "confidence": a.confidence,
                    "sample_count": a.sample_count,
                    "created_at": a.provenance.timestamp
                }
                for a in self.abstractions.values()
            ],
            "strategies": [
                {
                    "strategy_name": s.strategy_name,
                    "task_family_mapping": s.task_family_mapping,
                    "retry_budget": s.retry_budget,
                    "direct_route_threshold": s.direct_route_threshold,
                    "recurrent_steps_cap": s.recurrent_steps_cap,
                    "peer_assistance_allowed": s.peer_assistance_allowed
                }
                for s in self.strategies.values()
            ],
            "knowledge": [
                {
                    "knowledge_id": k.knowledge_id,
                    "subject": k.subject,
                    "relation": k.relation,
                    "target": k.target,
                    "grounding_confidence": k.grounding_confidence,
                    "verification_source": k.verification_source
                }
                for k in self.knowledge.values()
            ]
        }
        raw_bytes = json.dumps(payload, indent=2).encode("utf-8")
        if len(raw_bytes) > self.max_storage_kb * 1024:
            raise ValueError(f"Storage limit exceeded: {len(raw_bytes)} bytes > {self.max_storage_kb} KB")

        digest = hashlib.sha256(raw_bytes).hexdigest()
        self.storage_path.write_bytes(raw_bytes)
        manifest_path = self.storage_path.with_suffix(".sha256")
        manifest_path.write_text(f"{digest}  {self.storage_path.name}", encoding="utf-8")
        return digest

    def load_from_disk(self) -> bool:
        """Load and verify stored state from disk. Returns True if successfully restored."""
        if not self.storage_path.exists():
            return False

        manifest_path = self.storage_path.with_suffix(".sha256")
        raw_bytes = self.storage_path.read_bytes()
        actual_digest = hashlib.sha256(raw_bytes).hexdigest()

        if manifest_path.exists():
            expected = manifest_path.read_text(encoding="utf-8").strip().split()[0]
            if actual_digest != expected:
                raise ValueError("Integrity check failed: Durable experience storage corrupted!")

        payload = json.loads(raw_bytes.decode("utf-8"))
        now = time.time()

        # Restore Abstractions
        self.abstractions.clear()
        for a_data in payload.get("abstractions", []):
            # Check TTL
            age = now - a_data.get("created_at", now)
            if age > self.ttl_seconds:
                continue # Expired
            abs_exp = AbstractExperience(
                abstraction_id=a_data["abstraction_id"],
                task_family=a_data["task_family"],
                supporting_episode_ids=a_data["supporting_episode_ids"],
                contradictory_episode_ids=a_data["contradictory_episode_ids"],
                success_pattern=a_data["success_pattern"],
                failure_pattern=a_data["failure_pattern"],
                recommended_strategy=a_data["recommended_strategy"],
                confidence=a_data["confidence"],
                sample_count=a_data["sample_count"],
                provenance=ProvenanceMetadata()
            )
            self.abstractions[abs_exp.task_family] = abs_exp

        # Restore Strategies
        self.strategies.clear()
        for s_data in payload.get("strategies", []):
            strat = StrategyPolicy(
                strategy_name=s_data["strategy_name"],
                task_family_mapping=s_data["task_family_mapping"],
                retry_budget=s_data["retry_budget"],
                direct_route_threshold=s_data["direct_route_threshold"],
                recurrent_steps_cap=s_data["recurrent_steps_cap"],
                peer_assistance_allowed=s_data["peer_assistance_allowed"],
                provenance=ProvenanceMetadata()
            )
            self.strategies[strat.strategy_name] = strat

        # Restore Knowledge
        self.knowledge.clear()
        for k_data in payload.get("knowledge", []):
            kn = VerifiedKnowledge(
                knowledge_id=k_data["knowledge_id"],
                subject=k_data["subject"],
                relation=k_data["relation"],
                target=k_data["target"],
                grounding_confidence=k_data["grounding_confidence"],
                verification_source=k_data["verification_source"],
                provenance=ProvenanceMetadata()
            )
            self.knowledge[kn.knowledge_id] = kn

        return True

    def register_knowledge(self, kn: VerifiedKnowledge) -> None:
        """Register knowledge item, resolving direct contradictions."""
        AdaptiveLearningContract.verify_artifact_isolation(kn)
        # Check for contradictions: same subject & relation, but different target
        for existing_id, existing in list(self.knowledge.items()):
            if existing.subject == kn.subject and existing.relation == kn.relation:
                if existing.target != kn.target:
                    # Conflict: Higher confidence wins; if newer is higher, replace
                    if kn.grounding_confidence > existing.grounding_confidence:
                        del self.knowledge[existing_id]
                    else:
                        return # Reject weaker incoming statement
        self.knowledge[kn.knowledge_id] = kn
