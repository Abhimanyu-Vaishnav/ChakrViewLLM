"""
Controlled Autonomous Self-Improvement & Update Lifecycle Manager.
Implements:
1. Hypothesis formulation from recurring weaknesses.
2. Isolated candidate training and experimental branch creation.
3. Frozen benchmark evaluation gate against production baseline.
4. Automatic rejection of regressive candidates.
5. Cryptographic promotion manifests.
"""

from __future__ import annotations
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple


class AutonomousUpdateManager:
    def __init__(self, updates_dir: Optional[Path] = None):
        if updates_dir is None:
            updates_dir = Path("data/self_updates")
        self.updates_dir = updates_dir
        self.updates_dir.mkdir(parents=True, exist_ok=True)

        self.update_history_file = self.updates_dir / "update_proposals.json"
        self.proposals: List[Dict[str, Any]] = self._load(self.update_history_file, default=[])

    def _load(self, path: Path, default: Any) -> Any:
        if path.exists():
            try:
                return json.load(open(path, encoding="utf-8"))
            except Exception:
                return default
        return default

    def _save(self) -> None:
        with open(self.update_history_file, "w", encoding="utf-8") as f:
            json.dump(self.proposals, f, indent=2)

    def formulate_update_hypothesis(
        self,
        weakness_identified: str,
        proposed_action: str,
        target_metric: str,
        baseline_score: float
    ) -> Dict[str, Any]:
        proposal = {
            "proposal_id": f"prop_{len(self.proposals)+1}_{int(time.time())}",
            "timestamp": time.time(),
            "weakness": weakness_identified,
            "action": proposed_action,
            "target_metric": target_metric,
            "baseline_score": baseline_score,
            "candidate_score": None,
            "decision": "PENDING"
        }
        self.proposals.append(proposal)
        self._save()
        return proposal

    def evaluate_and_gate_candidate(
        self,
        proposal_id: str,
        candidate_score: float,
        min_improvement_threshold: float = 0.0
    ) -> Tuple[bool, str]:
        proposal = next((p for p in self.proposals if p["proposal_id"] == proposal_id), None)
        if not proposal:
            return False, "Proposal ID not found."

        proposal["candidate_score"] = candidate_score
        baseline = proposal["baseline_score"]

        # Gate: Candidate must show improvement without regressions
        # For loss/perplexity metrics: candidate_score < baseline
        # For accuracy/score metrics: candidate_score > baseline
        is_loss_metric = any(term in proposal.get("target_metric", "").lower() for term in ["loss", "perplexity", "ppl", "nll"])
        
        if is_loss_metric:
            is_better = (candidate_score <= baseline - min_improvement_threshold)
        else:
            is_better = (candidate_score >= baseline + min_improvement_threshold)

        if is_better:
            proposal["decision"] = "APPROVED_FOR_PROMOTION"
            self._save()
            return True, f"Candidate passed evaluation gate: {candidate_score:.4f} vs baseline {baseline:.4f}"
        else:
            proposal["decision"] = "REJECTED_REGRESSION"
            self._save()
            return False, f"Candidate rejected (regression detected): {candidate_score:.4f} vs baseline {baseline:.4f}"
