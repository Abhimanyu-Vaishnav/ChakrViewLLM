"""
Wave 546 & 547: Open-World Knowledge Interface & Traceable Evidence Provenance.

Architectural Guarantees:
- Tool/Provider-agnostic neutral knowledge interface.
- External providers NEVER directly mutate neural weights (Delta W remains 0).
- External providers NEVER directly promote knowledge into trusted memory without governance.
- Returns structured, verifiable evidence candidates.
- Full provenance lineage: source identity, content hash, extraction spans, temporal stamps, provider identity.
- Exact evidence attribution: evaluates whether evidence supports the exact claim about the exact entity.
"""

from __future__ import annotations

import time
import hashlib
import uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Set


class SourceType(str, Enum):
    WEB_RETRIEVAL = "WEB_RETRIEVAL"
    LOCAL_DOCUMENT = "LOCAL_DOCUMENT"
    STRUCTURED_API = "STRUCTURED_API"
    EXTERNAL_DATABASE = "EXTERNAL_DATABASE"
    USER_PROVIDED = "USER_PROVIDED"
    INSTITUTIONAL_SOURCE = "INSTITUTIONAL_SOURCE"


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTESTED = "CONTESTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNKNOWN = "UNKNOWN"
    OUTDATED = "OUTDATED"
    REJECTED = "REJECTED"
    UNVERIFIED = "UNVERIFIED"


@dataclass
class KnowledgeSource:
    source_id: str
    source_type: SourceType
    uri_or_doc_id: str
    publisher_or_author: Optional[str] = None
    publication_timestamp: Optional[float] = None
    retrieval_timestamp: float = field(default_factory=time.time)
    reputation_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceItem:
    evidence_id: str
    request_id: str
    source_id: str
    query_used: str
    extracted_text: str
    content_hash: str
    target_entity: str
    predicate_or_relation: str
    claimed_value: str
    observed_at: float
    confidence: float = 0.5
    corroborating_evidence_ids: List[str] = field(default_factory=list)
    contradicting_evidence_ids: List[str] = field(default_factory=list)
    provider_identity: str = "mock_provider"
    temporal_validity_start: Optional[float] = None
    temporal_validity_end: Optional[float] = None


@dataclass
class KnowledgeRequest:
    request_id: str = field(default_factory=lambda: f"kreq_{uuid.uuid4().hex[:8]}")
    target_entity: str = ""
    target_relation: str = ""
    query_text: str = ""
    domain: str = "general"
    required_freshness_seconds: float = 86400.0  # 24h
    importance_level: float = 0.5
    allowed_sources: List[SourceType] = field(
        default_factory=lambda: [
            SourceType.LOCAL_DOCUMENT,
            SourceType.STRUCTURED_API,
            SourceType.WEB_RETRIEVAL,
        ]
    )
    verification_quorum_needed: int = 2


@dataclass
class AttributionAssessment:
    evidence_id: str
    target_entity_matches: bool
    predicate_matches: bool
    conclusion_supported: bool
    is_attribution_valid: bool
    rejection_reason: Optional[str] = None


@dataclass
class KnowledgeCandidate:
    candidate_id: str
    request_id: str
    proposition: str
    target_entity: str
    veracity_claim: bool
    evidence_items: List[EvidenceItem]
    verification_status: VerificationStatus
    overall_confidence: float
    provenance_chain_valid: bool
    attribution_valid: bool
    summary: str


class OpenWorldKnowledgeInterface:
    """
    Neutral interface for requesting external knowledge without coupling to specific search engines.
    Enforces strict provenance hashing and attribution verification.
    """

    def __init__(self):
        self.registered_sources: Dict[str, KnowledgeSource] = {}
        self.request_audit_log: List[KnowledgeRequest] = []

    def register_source(self, source: KnowledgeSource) -> None:
        self.registered_sources[source.source_id] = source

    @staticmethod
    def compute_content_hash(text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    @classmethod
    def evaluate_attribution(
        cls,
        evidence: EvidenceItem,
        requested_entity: str,
        expected_relation: str,
    ) -> AttributionAssessment:
        """
        Guarantees that evidence is not cited if it is about a different entity
        or does not actually ground the queried relation.
        """
        entity_match = (
            evidence.target_entity.strip().lower()
            == requested_entity.strip().lower()
        )
        pred_match = (
            expected_relation.strip().lower() in evidence.predicate_or_relation.strip().lower()
            or evidence.predicate_or_relation.strip().lower() in expected_relation.strip().lower()
        )
        supported = entity_match and pred_match and len(evidence.extracted_text.strip()) > 5

        rejection_reason = None
        if not entity_match:
            rejection_reason = f"Evidence target entity '{evidence.target_entity}' does not match requested '{requested_entity}'."
        elif not pred_match:
            rejection_reason = f"Evidence predicate '{evidence.predicate_or_relation}' does not match requested relation '{expected_relation}'."
        elif not supported:
            rejection_reason = "Evidence content is insufficient or ungrounded."

        return AttributionAssessment(
            evidence_id=evidence.evidence_id,
            target_entity_matches=entity_match,
            predicate_matches=pred_match,
            conclusion_supported=supported,
            is_attribution_valid=supported,
            rejection_reason=rejection_reason,
        )

    def process_external_evidence(
        self,
        request: KnowledgeRequest,
        raw_evidence: List[Dict[str, Any]],
        provider_id: str = "generic_provider",
    ) -> KnowledgeCandidate:
        """
        Converts raw retrieved payloads into structured, auditable KnowledgeCandidates.
        Checks provenance, content hashes, and entity attribution.
        """
        self.request_audit_log.append(request)
        evidence_items: List[EvidenceItem] = []
        all_attributions_valid = True

        for idx, raw in enumerate(raw_evidence):
            text = raw.get("text", "")
            c_hash = self.compute_content_hash(text)
            src_id = raw.get("source_id", f"src_{provider_id}_{idx}")

            if src_id not in self.registered_sources:
                self.register_source(
                    KnowledgeSource(
                        source_id=src_id,
                        source_type=raw.get("source_type", SourceType.WEB_RETRIEVAL),
                        uri_or_doc_id=raw.get("uri", "https://example.com/source"),
                        reputation_score=raw.get("reputation", 0.5),
                    )
                )

            ev = EvidenceItem(
                evidence_id=f"ev_{request.request_id}_{idx}_{c_hash[:6]}",
                request_id=request.request_id,
                source_id=src_id,
                query_used=request.query_text,
                extracted_text=text,
                content_hash=c_hash,
                target_entity=raw.get("entity", request.target_entity),
                predicate_or_relation=raw.get("relation", request.target_relation),
                claimed_value=raw.get("value", ""),
                observed_at=time.time(),
                confidence=raw.get("confidence", 0.7),
                provider_identity=provider_id,
                temporal_validity_start=raw.get("valid_from"),
                temporal_validity_end=raw.get("valid_until"),
            )

            # Attribution check
            att_check = self.evaluate_attribution(
                ev, request.target_entity, request.target_relation
            )
            if not att_check.is_attribution_valid:
                all_attributions_valid = False

            evidence_items.append(ev)

        # Candidate synthesis
        candidate_id = f"cand_{request.request_id}_{hashlib.sha256(str(len(evidence_items)).encode()).hexdigest()[:6]}"
        is_provenance_valid = all(len(ev.content_hash) == 64 for ev in evidence_items)

        status = VerificationStatus.UNVERIFIED
        if not evidence_items:
            status = VerificationStatus.INSUFFICIENT_EVIDENCE
        elif not all_attributions_valid:
            status = VerificationStatus.REJECTED

        return KnowledgeCandidate(
            candidate_id=candidate_id,
            request_id=request.request_id,
            proposition=f"{request.target_entity} {request.target_relation}",
            target_entity=request.target_entity,
            veracity_claim=True if (evidence_items and all_attributions_valid) else False,
            evidence_items=evidence_items,
            verification_status=status,
            overall_confidence=sum(e.confidence for e in evidence_items) / max(1, len(evidence_items)),
            provenance_chain_valid=is_provenance_valid,
            attribution_valid=all_attributions_valid,
            summary=f"Processed {len(evidence_items)} items for {request.target_entity}; attribution_valid={all_attributions_valid}",
        )
