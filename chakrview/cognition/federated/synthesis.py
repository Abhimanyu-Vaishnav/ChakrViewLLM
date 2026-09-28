"""
Federated Synthesis & Consensus Layer (Step 26).

Synthesizes multi-agent evidence, analyses, and critiques into a unified candidate
while explicitly preserving minority opinions and epistemic uncertainty.

CRITICAL INVARIANTS:
1. NON-AUTHORITATIVE: The synthesizer produces a candidate proposal, NOT a sovereign command.
2. NO SILENT OVERWRITES: Unresolved conflicts and minority opinions are preserved in the candidate.
3. DECISION ROUTING: Maps evidence state to bounded DecisionState without fabricating certainty.
"""

from typing import Dict, List, Optional, Any
import uuid

from chakrview.cognition.federated.models import (
    FederatedSynthesisCandidate,
    FederatedConflictRecord,
    ConflictState,
    AgentMessage,
    AgentRole,
)
from chakrview.cognition.unified.models import DecisionState


class FederatedSynthesizer:
    """
    Deterministic synthesizer assembling multi-agent contributions into a cohesive proposal.
    """

    def synthesize(
        self,
        task_id: str,
        objective: str,
        messages: List[AgentMessage],
        corroborated_evidence: List[Dict[str, Any]],
        conflicts: List[FederatedConflictRecord],
        critical_counter_evidence: List[Dict[str, Any]],
        is_capability_task: bool = False,
    ) -> FederatedSynthesisCandidate:
        """
        Assemble multi-agent outputs, evidence, and conflicts into a FederatedSynthesisCandidate.
        """
        cid = f"fsyn_{uuid.uuid4().hex[:10]}"
        analyst_claims: List[str] = []
        researcher_facts: List[str] = []
        critic_caveats: List[str] = []
        minority_opinions: List[Dict[str, Any]] = []

        for msg in messages:
            p = msg.payload
            role = p.get("role", "")
            if role == AgentRole.ANALYST.value and p.get("claim"):
                analyst_claims.append(p["claim"])
            elif role == AgentRole.RESEARCHER.value:
                for mem in p.get("retrieved_memories", []):
                    content = mem.get("content")
                    if content and content not in researcher_facts:
                        researcher_facts.append(content)
            elif role == AgentRole.CRITIC.value:
                if p.get("counter_evidence"):
                    critic_caveats.append(f"{len(p['counter_evidence'])} counter-evidence items flagged")

        # Ingest unresolved conflicts as minority opinions
        unresolved_conflicts = [c for c in conflicts if c.state in (ConflictState.CONFLICT, ConflictState.UNRESOLVED)]
        for conf in unresolved_conflicts:
            for cl in conf.claims:
                minority_opinions.append({
                    "agent_id": cl.get("agent_id", "conflicting_agent"),
                    "claim": cl.get("claim"),
                    "conflict_id": conf.conflict_id,
                })

        # Construct primary synthesis string
        parts = []
        if researcher_facts:
            parts.append(f"Factual basis: {'; '.join(researcher_facts[:3])}.")
        if analyst_claims:
            parts.append(f"Analysis: {analyst_claims[-1]}")
        else:
            parts.append(f"Federated assessment concluded for: {objective}")

        primary_synthesis = " ".join(parts)

        # Determine recommended DecisionState & uncertainty notes
        uncertainty_notes: Optional[str] = None
        recommended_decision = DecisionState.ANSWER
        confidence = 0.85

        if is_capability_task:
            recommended_decision = DecisionState.CAPABILITY_REQUIRED
            confidence = 0.90
            uncertainty_notes = "External capability execution required; routed to CapabilityGate."
        elif unresolved_conflicts:
            recommended_decision = DecisionState.ANSWER_WITH_UNCERTAINTY
            confidence = 0.50
            uncertainty_notes = f"Unresolved disagreement across {len(unresolved_conflicts)} conflict records."
        elif not corroborated_evidence and not researcher_facts:
            # Check if query is factual/analytical but lacks evidence
            words = objective.split()
            if len(words) > 4 and any(w.lower().rstrip("?:;,.") in ("what", "why", "how", "who", "when", "calculate", "prove") for w in words):
                recommended_decision = DecisionState.INSUFFICIENT_INFORMATION
                confidence = 0.25
                uncertainty_notes = "No corroborated evidence was retrieved to support an authoritative conclusion."
        elif critical_counter_evidence:
            recommended_decision = DecisionState.ANSWER_WITH_UNCERTAINTY
            confidence = 0.60
            uncertainty_notes = "Counter-evidence identified during critical examination."

        return FederatedSynthesisCandidate(
            candidate_id=cid,
            task_id=task_id,
            primary_synthesis=primary_synthesis,
            supporting_evidence=corroborated_evidence,
            unresolved_contradictions=[c.to_dict() for c in unresolved_conflicts],
            minority_opinions=minority_opinions,
            uncertainty_notes=uncertainty_notes,
            recommended_decision_state=recommended_decision,
            confidence=confidence,
        )
