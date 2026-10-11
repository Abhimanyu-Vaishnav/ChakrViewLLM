from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any
import uuid
import time

class KnowledgeAcquisitionStatus(Enum):
    """
    Authoritative verification states for acquired knowledge.
    """
    VERIFIED = "verified"
    SUPPORTED = "supported"
    CONTESTED = "contested"
    STALE = "stale"
    INSUFFICIENT = "insufficient"
    UNKNOWN = "unknown"
    ABSTAIN = "abstain"

class InvestigationScope(Enum):
    QUICK = auto()
    DEEP = auto()
    EXHAUSTIVE = auto()

@dataclass
class SourceMetadata:
    source_identity: str
    source_type: str
    reliability_score: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InvestigationSource:
    source_id: str
    uri: str
    metadata: SourceMetadata

@dataclass
class EvidenceItem:
    evidence_id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:8]}")
    request_id: str = ""
    source_id: str = ""
    content: str = ""
    confidence: float = 0.0
    status: KnowledgeAcquisitionStatus = KnowledgeAcquisitionStatus.UNKNOWN
    retrieved_at: float = field(default_factory=time.time)
    corroborating_evidence_ids: List[str] = field(default_factory=list)
    contradicting_evidence_ids: List[str] = field(default_factory=list)

@dataclass
class InvestigationRequest:
    request_id: str = field(default_factory=lambda: f"inv_{uuid.uuid4().hex[:8]}")
    user_query: str = ""
    required_information: List[str] = field(default_factory=list)
    scope: InvestigationScope = InvestigationScope.QUICK
    freshness_requirement_seconds: float = 86400.0  # 24 hours
    allowed_source_types: List[str] = field(default_factory=lambda: ["local", "document", "web"])
    maximum_evidence_budget: int = 5
    created_at: float = field(default_factory=time.time)

@dataclass
class InvestigationResult:
    result_id: str = field(default_factory=lambda: f"res_{uuid.uuid4().hex[:8]}")
    request_id: str = ""
    evidence_items: List[EvidenceItem] = field(default_factory=list)
    overall_status: KnowledgeAcquisitionStatus = KnowledgeAcquisitionStatus.UNKNOWN
    generated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
