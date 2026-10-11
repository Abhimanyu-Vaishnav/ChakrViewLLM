"""
Federated Evidence Aggregator (Step 26).

Enforces strict epistemic distinction across agent communications:
    AGENT CLAIM         != EVIDENCE
    INTERPRETATION      != FACT
    ASSUMPTION          != TRUTH
    RECOMMENDATION      != DECISION

CRITICAL PRINCIPLES:
1. No Blind Majority Voting:
   "Three agents said X" does NOT increase truth confidence without empirical corroboration.
2. Provenance Tracking:
   Every evidence item preserves its source attribution, confidence, and verification state.
3. Uncorroborated assertions remain flagged as claims, not ground evidence.
"""

from typing import Dict, List, Optional, Any, Set
from chakrview.cognition.federated.models import AgentMessage, AgentRole


class EvidenceCategory:
    GROUND_EVIDENCE = "GROUND_EVIDENCE"
    AGENT_CLAIM = "AGENT_CLAIM"
    INTERPRETATION = "INTERPRETATION"
    ASSUMPTION = "ASSUMPTION"
    RECOMMENDATION = "RECOMMENDATION"
    COUNTER_EVIDENCE = "COUNTER_EVIDENCE"


class FederatedEvidenceAggregator:
    """
    Deterministic aggregator analyzing agent messages and categorizing claims vs verified evidence.
    """

    def __init__(self) -> None:
        self.evidence_pool: List[Dict[str, Any]] = []
        self.claims: List[Dict[str, Any]] = []
        self.assumptions: List[Dict[str, Any]] = []
        self.counter_evidence: List[Dict[str, Any]] = []

    def ingest_message(self, message: AgentMessage) -> None:
        """Categorize and ingest assertions from an AgentMessage."""
        payload = message.payload
        role = payload.get("role", "")
        sender = message.sender_agent_id

        # 1. Ground Evidence from Researcher / Memory
        if role == AgentRole.RESEARCHER.value and "retrieved_memories" in payload:
            for mem in payload["retrieved_memories"]:
                self.evidence_pool.append({
                    "category": EvidenceCategory.GROUND_EVIDENCE,
                    "content": mem.get("content", ""),
                    "source": "continual_memory",
                    "sender_agent_id": sender,
                    "confidence": mem.get("confidence", 0.7),
                    "verification_status": mem.get("verification_status", "UNVERIFIED"),
                    "provenance": message.provenance,
                    "memory_id": mem.get("memory_id"),
                })

        # 2. Inferences & Claims from Analyst
        if role == AgentRole.ANALYST.value and "claim" in payload:
            claim_text = payload["claim"]
            if claim_text:
                self.claims.append({
                    "category": EvidenceCategory.INTERPRETATION if payload.get("success") else EvidenceCategory.AGENT_CLAIM,
                    "content": claim_text,
                    "sender_agent_id": sender,
                    "confidence": payload.get("confidence", 0.5),
                    "subproblems_resolved": payload.get("subproblems_resolved", 0),
                    "provenance": message.provenance,
                })

        # 3. Assumptions and Counter-Evidence from Critic
        if role == AgentRole.CRITIC.value:
            for a in payload.get("assumptions", []):
                self.assumptions.append({
                    "category": EvidenceCategory.ASSUMPTION,
                    "content": a.get("statement", ""),
                    "sender_agent_id": sender,
                    "plausibility": a.get("plausibility", 0.5),
                    "falsifiable": a.get("falsifiable", True),
                })
            for ce in payload.get("counter_evidence", []):
                self.counter_evidence.append({
                    "category": EvidenceCategory.COUNTER_EVIDENCE,
                    "content": ce.get("description", ""),
                    "sender_agent_id": sender,
                    "status": ce.get("status", "UNCONFIRMED"),
                    "source": ce.get("source", "critical_analysis"),
                })

    def get_corroborated_evidence(self) -> List[Dict[str, Any]]:
        """Filter evidence items that possess ground-truth or high-confidence memory backing."""
        return [
            e for e in self.evidence_pool
            if e.get("verification_status") in ("VERIFIED", "EXTERNAL_VERIFIED") or e.get("confidence", 0) >= 0.80
        ]

    def get_summary(self) -> Dict[str, Any]:
        """Produce an auditable evidence accounting summary."""
        return {
            "ground_evidence_count": len(self.evidence_pool),
            "claims_count": len(self.claims),
            "assumptions_count": len(self.assumptions),
            "counter_evidence_count": len(self.counter_evidence),
            "corroborated_evidence_count": len(self.get_corroborated_evidence()),
        }

    def clear(self) -> None:
        self.evidence_pool.clear()
        self.claims.clear()
        self.assumptions.clear()
        self.counter_evidence.clear()
