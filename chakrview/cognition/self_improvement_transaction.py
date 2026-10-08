"""
Self-Improvement Transaction with Complete State Rollback.
Wave 533 / 536: Bounded transactional governance for cognitive updates.

Rollback Completeness Invariant:
A transaction must snapshot and restore ALL affected governed state:
- strategies: Dict[str, Any]
- knowledge: Dict[str, Any]
- experiences: List[Any]
- trust_records: Dict[str, Any]
- metadata: Dict[str, Any]
- provenance: Dict[str, Any]
- curriculum: Dict[str, Any]
- candidate_state: Dict[str, Any]
- indexes: Dict[str, Any]
- checkpoints: List[str]

If an update is rejected or triggers regression, every field is restored
without leaving orphan state or mutated references.
"""

from typing import Dict, Any, Optional
import copy
import hashlib
import time

class SelfImprovementTransaction:
    def __init__(self, parent_state: Dict[str, Any]):
        self.transaction_id = f"tx_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:10]}"
        # Ensure deepcopy of complete state schema
        self.parent_snapshot = copy.deepcopy(parent_state)
        self.current_state = copy.deepcopy(parent_state)
        
        # Enforce canonical fields exist in working state
        for field_name in [
            "strategies", "knowledge", "experiences", "trust_records",
            "metadata", "provenance", "curriculum", "candidate_state",
            "indexes", "checkpoints"
        ]:
            if field_name not in self.current_state:
                self.current_state[field_name] = copy.deepcopy(self.parent_snapshot.get(field_name, {} if field_name != "experiences" and field_name != "checkpoints" else []))
            if field_name not in self.parent_snapshot:
                self.parent_snapshot[field_name] = copy.deepcopy(self.current_state[field_name])

        self.candidate_delta: Optional[Dict[str, Any]] = None
        self.is_committed = False
        self.is_rolled_back = False

    def apply_candidate(self, candidate_delta: Dict[str, Any]) -> None:
        """Apply candidate update to current working state across all governed facets."""
        self.candidate_delta = candidate_delta
        if "strategy_updates" in candidate_delta:
            self.current_state["strategies"].update(candidate_delta["strategy_updates"])
        if "knowledge_updates" in candidate_delta:
            self.current_state["knowledge"].update(candidate_delta["knowledge_updates"])
        if "experience_updates" in candidate_delta:
            if isinstance(self.current_state["experiences"], list):
                self.current_state["experiences"].extend(candidate_delta["experience_updates"])
        if "trust_updates" in candidate_delta:
            self.current_state["trust_records"].update(candidate_delta["trust_updates"])
        if "metadata_updates" in candidate_delta:
            self.current_state["metadata"].update(candidate_delta["metadata_updates"])
        if "provenance_updates" in candidate_delta:
            self.current_state["provenance"].update(candidate_delta["provenance_updates"])
        if "curriculum_updates" in candidate_delta:
            self.current_state["curriculum"].update(candidate_delta["curriculum_updates"])
        if "candidate_state_updates" in candidate_delta:
            self.current_state["candidate_state"].update(candidate_delta["candidate_state_updates"])
        if "index_updates" in candidate_delta:
            self.current_state["indexes"].update(candidate_delta["index_updates"])
        if "checkpoint_updates" in candidate_delta:
            if isinstance(self.current_state["checkpoints"], list):
                self.current_state["checkpoints"].extend(candidate_delta["checkpoint_updates"])

    def evaluate_and_finalize(self, held_out_gain: float, regression_detected: bool) -> Dict[str, Any]:
        """
        Enforce commit or complete rollback based on empirical evaluation.
        """
        if held_out_gain > 0.0 and not regression_detected:
            self.is_committed = True
            return {
                "transaction_id": self.transaction_id,
                "status": "COMMITTED",
                "reason": f"Positive held-out transfer gain (+{held_out_gain:.1f}%) and 0 regressions.",
                "final_state": self.current_state
            }
        else:
            # COMPLETE ROLLBACK to parent snapshot across all dimensions
            self.current_state = copy.deepcopy(self.parent_snapshot)
            self.is_rolled_back = True
            reason = "Regression detected" if regression_detected else "Zero or negative transfer gain"
            return {
                "transaction_id": self.transaction_id,
                "status": "ROLLED_BACK",
                "reason": f"Harmful or ineffective candidate rejected: {reason}.",
                "final_state": self.current_state
            }
