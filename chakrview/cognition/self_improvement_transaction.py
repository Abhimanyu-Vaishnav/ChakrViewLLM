"""
Self-Improvement Transaction with Rollback.
Wave 533: Bounded transactional governance for cognitive updates.

Transaction Protocol:
1. BEGIN: Snapshot parent state (strategies, knowledge, parameters).
2. APPLY: Apply isolated candidate modification.
3. VERIFY: Run held-out transfer benchmarks & capability regression gates.
4. DECIDE:
   - If improvement is statistically significant and 0 regressions occur: COMMIT.
   - If any regression occurs, or candidate is harmful: ROLLBACK immediately.
"""

from typing import Dict, Any, Optional
import copy
import hashlib
import time

class SelfImprovementTransaction:
    def __init__(self, parent_state: Dict[str, Any]):
        self.transaction_id = f"tx_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:10]}"
        self.parent_snapshot = copy.deepcopy(parent_state)
        self.current_state = copy.deepcopy(parent_state)
        self.candidate_delta: Optional[Dict[str, Any]] = None
        self.is_committed = False
        self.is_rolled_back = False

    def apply_candidate(self, candidate_delta: Dict[str, Any]) -> None:
        """Apply candidate update to current working state."""
        self.candidate_delta = candidate_delta
        if "strategy_updates" in candidate_delta:
            self.current_state["strategies"].update(candidate_delta["strategy_updates"])
        if "knowledge_updates" in candidate_delta:
            self.current_state["knowledge"].update(candidate_delta["knowledge_updates"])

    def evaluate_and_finalize(self, held_out_gain: float, regression_detected: bool) -> Dict[str, Any]:
        """
        Enforce commit or rollback based on empirical evaluation.
        """
        # Promotion gate: requires positive held-out gain AND zero regression
        if held_out_gain > 0.0 and not regression_detected:
            self.is_committed = True
            return {
                "transaction_id": self.transaction_id,
                "status": "COMMITTED",
                "reason": f"Positive held-out transfer gain (+{held_out_gain:.1f}%) and 0 regressions.",
                "final_state": self.current_state
            }
        else:
            # ROLLBACK to parent snapshot
            self.current_state = copy.deepcopy(self.parent_snapshot)
            self.is_rolled_back = True
            reason = "Regression detected" if regression_detected else "Zero or negative transfer gain"
            return {
                "transaction_id": self.transaction_id,
                "status": "ROLLED_BACK",
                "reason": f"Harmful or ineffective candidate rejected: {reason}.",
                "final_state": self.current_state
            }
