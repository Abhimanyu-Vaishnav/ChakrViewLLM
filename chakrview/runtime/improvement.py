"""
Controlled Self-Improvement Architecture for ChakrView (Step 9).

Enforces human-auditable, gated self-improvement:
Observe -> Evaluate -> Identify Weakness -> Propose -> Sandbox -> Compare -> Gate -> Promote -> Checkpoint.

Guarantees the model never autonomously modifies its own weights or executable code.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from typing import Dict, List, Optional, Any


class ChangeType(str, Enum):
    """Classification of system modification targets."""
    KNOWLEDGE = "knowledge"          # Vector indexes, docs (Low Risk)
    SKILL = "skill"                  # Prompt templates, tool policies (Low Risk)
    CONFIGURATION = "configuration"  # Hyperparameters, context budget (Low Risk)
    ADAPTER = "adapter"              # LoRA / prefix tuning weights (Medium Risk)
    WEIGHTS = "weights"              # Base neural network parameters (High Risk)
    CODE = "code"                    # Software / execution engine logic (Critical Risk)


class RiskLevel(str, Enum):
    """Assessed risk level of a proposed change."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProposalStatus(str, Enum):
    """Lifecycle stages of an ImprovementProposal."""
    DRAFT = "draft"
    PROPOSED = "proposed"
    EVALUATING = "evaluating"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    QUARANTINED = "quarantined"
    ROLLED_BACK = "rolled_back"


@dataclass
class ImprovementProposal:
    """
    Formal specification of a proposed system change.
    
    Attributes:
        proposal_id: Unique string identifier.
        target: Target subsystem or module name.
        change_type: Target category.
        parent_version: Active version ID against which the proposal is made.
        evidence: Empirical data (metrics, loss, regression output) motivating the proposal.
        evaluation_suite: Benchmark or test suite required to validate candidate.
        expected_change: Documented hypothesis of behavioral/performance impact.
        risk_level: Assessed risk category.
        status: Current lifecycle state.
        created_at: ISO timestamp of creation.
        approved_by: Identity of human reviewer or authoritative gating agent.
        artifact_hash: SHA-256 digest of candidate update payload.
        rejection_reason: Explanation if rejected or quarantined.
        metadata: Optional execution logs or metrics.
    """
    proposal_id: str
    target: str
    change_type: ChangeType
    parent_version: str
    evidence: Dict[str, Any]
    evaluation_suite: str
    expected_change: str
    risk_level: RiskLevel
    status: ProposalStatus = ProposalStatus.DRAFT
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    approved_by: Optional[str] = None
    artifact_hash: Optional[str] = None
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["change_type"] = self.change_type.value
        data["risk_level"] = self.risk_level.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImprovementProposal":
        data = dict(data)
        if isinstance(data.get("change_type"), str):
            data["change_type"] = ChangeType(data["change_type"])
        if isinstance(data.get("risk_level"), str):
            data["risk_level"] = RiskLevel(data["risk_level"])
        if isinstance(data.get("status"), str):
            data["status"] = ProposalStatus(data["status"])
        return cls(**data)

    def can_promote(self) -> bool:
        """
        Verify whether the proposal meets invariants for promotion.
        High and Critical changes require explicit human approval and verified artifact hash.
        """
        if self.status != ProposalStatus.APPROVED:
            return False
        if not self.artifact_hash:
            return False
        if self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) and not self.approved_by:
            return False
        return True

    def submit_for_evaluation(self) -> None:
        if self.status not in (ProposalStatus.DRAFT, ProposalStatus.PROPOSED):
            raise ValueError(f"Cannot evaluate proposal in state '{self.status}'.")
        self.status = ProposalStatus.EVALUATING

    def approve(self, approver: str) -> None:
        if self.status != ProposalStatus.EVALUATING:
            raise ValueError(f"Cannot approve proposal in state '{self.status}'. Must be evaluating.")
        self.status = ProposalStatus.APPROVED
        self.approved_by = approver

    def reject(self, reason: str) -> None:
        self.status = ProposalStatus.REJECTED
        self.rejection_reason = reason

    def quarantine(self, reason: str) -> None:
        self.status = ProposalStatus.QUARANTINED
        self.rejection_reason = reason


class ImprovementProposalManager:
    """
    Manages proposal registry, lifecycle persistence, and gating rules.
    """

    def __init__(self, storage_dir: Optional[Path | str] = None) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir else None
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._proposals: Dict[str, ImprovementProposal] = {}

    def create_proposal(
        self,
        proposal_id: str,
        target: str,
        change_type: ChangeType,
        parent_version: str,
        evidence: Dict[str, Any],
        evaluation_suite: str,
        expected_change: str,
        risk_level: RiskLevel,
        artifact_hash: Optional[str] = None,
    ) -> ImprovementProposal:
        """Create and register a new proposal in DRAFT state."""
        if proposal_id in self._proposals:
            raise ValueError(f"Proposal ID '{proposal_id}' already exists.")
        
        # Self-governance invariant: Code updates default to CRITICAL risk
        if change_type == ChangeType.CODE and risk_level != RiskLevel.CRITICAL:
            risk_level = RiskLevel.CRITICAL
            
        proposal = ImprovementProposal(
            proposal_id=proposal_id,
            target=target,
            change_type=change_type,
            parent_version=parent_version,
            evidence=evidence,
            evaluation_suite=evaluation_suite,
            expected_change=expected_change,
            risk_level=risk_level,
            artifact_hash=artifact_hash,
            status=ProposalStatus.DRAFT,
        )
        self._proposals[proposal_id] = proposal
        self._persist(proposal)
        return proposal

    def get_proposal(self, proposal_id: str) -> Optional[ImprovementProposal]:
        return self._proposals.get(proposal_id)

    def list_proposals(
        self,
        status: Optional[ProposalStatus] = None,
        change_type: Optional[ChangeType] = None,
    ) -> List[ImprovementProposal]:
        proposals = list(self._proposals.values())
        if status:
            proposals = [p for p in proposals if p.status == status]
        if change_type:
            proposals = [p for p in proposals if p.change_type == change_type]
        return proposals

    def _persist(self, proposal: ImprovementProposal) -> None:
        if self.storage_dir:
            path = self.storage_dir / f"{proposal.proposal_id}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(proposal.to_dict(), f, indent=2)
