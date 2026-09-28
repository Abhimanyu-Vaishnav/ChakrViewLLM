"""
Thought Representation for ChakrView Deliberation (Step 21).

Provides strongly typed, auditable, and immutable thought steps:
- Structured computational representation (NOT free-form unmonitored text).
- Explicit purposes (OBSERVE, INTERPRET, QUESTION, HYPOTHESIZE, etc.).
- Provenance and lineage references linking inputs, evidence, and verification.
- Tamper-resistant immutability: historical steps cannot be silently modified.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Dict, List, Optional, Any
import uuid


class ThoughtPurpose(str, Enum):
    """Explicit functional purpose of a deliberate thought step."""
    OBSERVE = "OBSERVE"                  # Ingest input or environment signal
    INTERPRET = "INTERPRET"              # Extract meaning / constraints / semantics
    QUESTION = "QUESTION"                # Formulate an inquiry or identify an unknown
    HYPOTHESIZE = "HYPOTHESIZE"          # Propose candidate explanation or approach
    COMPARE = "COMPARE"                  # Compare competing alternatives or assertions
    INFER = "INFER"                      # Apply deduction / rule / constraint check
    PLAN = "PLAN"                        # Outline structured step progression
    CRITIQUE = "CRITIQUE"                # Evaluate candidate against criteria and evidence
    VERIFY = "VERIFY"                    # Test outcome against expected invariants
    REVISE = "REVISE"                    # Formulate corrective direction upon weakness
    DECIDE = "DECIDE"                    # Select path or action based on verified evidence
    STOP = "STOP"                        # Conclude deliberation when stopping policy met


@dataclass(frozen=True)
class ThoughtStep:
    """
    Immutable, inspectable record of a single computational deliberation step.
    
    Architectural Rules:
    1. frozen=True guarantees historical thoughts cannot be silently mutated.
    2. NOT a raw chain-of-thought dump: structured summary of intent and outcome.
    3. Traces input lineages and evidence references for verifiable provenance.
    """
    thought_id: str = field(default_factory=lambda: f"th_{uuid.uuid4().hex[:10]}")
    step_index: int = 0
    purpose: ThoughtPurpose = ThoughtPurpose.INTERPRET
    content: str = ""
    input_references: List[str] = field(default_factory=list)
    hypothesis_reference: Optional[str] = None
    evidence_references: List[str] = field(default_factory=list)
    candidate_action: Optional[str] = None
    result: Optional[Any] = None
    uncertainty: Optional[float] = None
    confidence_available: bool = False
    verification_state: str = "UNVERIFIED"  # "PASS", "FAIL", "UNVERIFIED"
    provenance: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["purpose"] = self.purpose.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThoughtStep":
        d = dict(data)
        if "purpose" in d and isinstance(d["purpose"], str):
            d["purpose"] = ThoughtPurpose(d["purpose"])
        return cls(**d)
