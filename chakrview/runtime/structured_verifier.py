"""
Structured Entity-Attribute Claim Verifier for ChakrView.
Location: chakrview/runtime/structured_verifier.py

Implements:
- StructuredClaim dataclass (subject, attribute, value, unit).
- VerificationVerdict Enum: [SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE, CONFLICTING_EVIDENCE, INVALID_CLAIM].
- StructuredVerifier: Matches entity and attribute claims against indexed document evidence.
- Explicit normalization for numbers, storage units, currencies, and time intervals.
- Distinguishes absence of evidence (INSUFFICIENT_EVIDENCE) from contradiction (CONTRADICTED).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import re
import time
from typing import Any, Dict, List, Optional, Tuple


class VerificationVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    INVALID_CLAIM = "INVALID_CLAIM"


@dataclass
class StructuredClaim:
    claim_id: str
    subject_entity: str
    attribute_name: str
    claimed_value: str
    expected_unit: Optional[str] = None
    comparison_mode: str = "auto"  # "auto", "exact", "numeric", "string"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationRecord:
    claim_id: str
    verdict: VerificationVerdict
    confidence: float
    explanation: str
    subject_entity: str
    attribute_name: str
    claimed_value: str
    evidence_value: Optional[str] = None
    source_doc_id: Optional[str] = None
    supporting_text: Optional[str] = None
    conflicting_sources: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


class StructuredVerifier:
    """
    Compares explicit entity-attribute claims against indexed document chunks.
    """

    def __init__(self, chunks: Optional[List[Any]] = None) -> None:
        self.chunks = chunks or []

    def set_chunks(self, chunks: List[Any]) -> None:
        self.chunks = chunks

    @staticmethod
    def normalize_numeric_string(val_str: str) -> Optional[float]:
        """Extract and parse numeric component from string."""
        if not val_str:
            return None
        clean = val_str.replace(",", "").strip()
        # Remove currency symbols or common units for pure numeric value
        clean = re.sub(r"[\$€£]", "", clean)
        m = re.search(r"[-+]?\d+(?:\.\d+)?", clean)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
        return None

    @staticmethod
    def normalize_bytes(val_str: str) -> Optional[int]:
        """Convert representations like '10 MB', '10485760 bytes' to integer bytes."""
        if not val_str:
            return None
        s = val_str.lower().strip()
        m = re.search(r"(\d+(?:\.\d+)?)\s*(mb|kb|gb|bytes|b)\b", s)
        if m:
            qty = float(m.group(1))
            unit = m.group(2)
            if unit == "mb":
                return int(qty * 1024 * 1024)
            elif unit == "kb":
                return int(qty * 1024)
            elif unit == "gb":
                return int(qty * 1024 * 1024 * 1024)
            elif unit in ("bytes", "b"):
                return int(qty)
        # Check pure digits
        m_dig = re.search(r"^\d+$", s)
        if m_dig:
            return int(m_dig.group(0))
        return None

    @staticmethod
    def normalize_time_ms(val_str: str) -> Optional[float]:
        """Convert '50 milliseconds', '50 ms', '2 seconds' to milliseconds."""
        if not val_str:
            return None
        s = val_str.lower().strip()
        m = re.search(r"(\d+(?:\.\d+)?)\s*(milliseconds|millisecond|ms|seconds|second|s)\b", s)
        if m:
            qty = float(m.group(1))
            unit = m.group(2)
            if "ms" in unit or "milli" in unit:
                return qty
            elif "s" in unit:
                return qty * 1000.0
        return None

    def _extract_attribute_from_chunk(
        self, chunk_text: str, entity: str, attribute: str
    ) -> Optional[Tuple[str, str]]:
        """
        Extracts candidate attribute value and matching sentence from chunk text.
        Returns: (extracted_value_string, matching_sentence_text) or None.
        """
        e_lower = entity.lower().strip()
        a_lower = attribute.lower().strip()
        c_lower = chunk_text.lower()
        # Check if entity is mentioned in chunk
        # Require entity words to match title or header if possible, or require full entity match
        e_words = [w for w in re.findall(r"\b[a-zA-Z0-9_\.-]+\b", e_lower) if len(w) > 2]
        if not e_words:
            return None
        
        # If multi-word entity, check if all words or phrase appears in chunk
        if len(e_words) > 1:
            if e_lower not in c_lower and not all(w in c_lower for w in e_words):
                return None
        else:
            if not any(w in c_lower for w in e_words):
                return None

        # Lines or bullet points in chunk text
        lines = [line.strip() for line in chunk_text.splitlines() if line.strip()]
        for line in lines:
            line_lower = line.lower()
            # Attribute words (keywords like 'k1', 'b', 'price', 'port', 'diameter', 'heartbeat', 'retention')
            a_words = [w for w in re.findall(r"\b[a-zA-Z0-9_\.-]+\b", a_lower) if len(w) >= 1]
            # Must match at least 60% of attribute words or a specific identifier like 'k1', 'b'
            matched_words = [w for w in a_words if w in line_lower]
            if len(matched_words) >= max(1, int(len(a_words) * 0.5)):
                # If entity is also in this line or nearby, prioritize it
                # Extract value following colon or dash
                # Patterns:
                # - "parameter k1: 1.5."
                # - "Starter Tier: $150 per month, includes up to 2 concurrent worker processes."
                # - "TCP port 8443."
                # - "diameter: 16 distributed participant nodes."
                
                # Check for "Attribute: Value"
                if ":" in line:
                    parts = line.split(":", 1)
                    prefix = parts[0].strip()
                    suffix = parts[1].strip()

                    # Avoid treating markdown headers like "# BM25 Lexical Knowledge Index" as attribute lines
                    if prefix.startswith("#"):
                        continue

                    prefix_lower = prefix.lower()
                    # Check if attribute keywords match prefix closely
                    if all(w in prefix_lower for w in a_words) or (len(a_words) > 1 and sum(1 for w in a_words if w in prefix_lower) >= 2):
                        # Extract first number/currency/unit from suffix
                        m_val = re.search(r"([\$€£]?\d+(?:\.\d+)?(?:\s*[a-zA-Z%]+)?)", suffix)
                        if m_val:
                            return m_val.group(1).strip(), line

                # Check if attribute is in line and followed by colon or value
                m_after = re.search(rf"(?:{'|'.join(re.escape(w) for w in a_words)}).*?[:=-]\s*([\$€£]?\d+(?:\.\d+)?(?:\s*[a-zA-Z%]+)?)", line, re.I)
                if m_after:
                    return m_after.group(1).strip(), line

        return None

    def verify_claim(self, claim: StructuredClaim) -> VerificationRecord:
        t0 = time.perf_counter()

        if not claim.subject_entity or not claim.attribute_name or not claim.claimed_value:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return VerificationRecord(
                claim_id=claim.claim_id,
                verdict=VerificationVerdict.INVALID_CLAIM,
                confidence=1.0,
                explanation="Claim is invalid: missing subject_entity, attribute_name, or claimed_value.",
                subject_entity=claim.subject_entity,
                attribute_name=claim.attribute_name,
                claimed_value=claim.claimed_value,
                latency_ms=elapsed,
            )

        extracted_evidences = []
        for chunk in self.chunks:
            # chunk may be a DocumentChunk or dict
            text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
            doc_id = chunk.doc_id if hasattr(chunk, "doc_id") else chunk.get("doc_id", "doc")
            res = self._extract_attribute_from_chunk(text, claim.subject_entity, claim.attribute_name)
            if res:
                extracted_val, sent_text = res
                extracted_evidences.append({
                    "val_str": extracted_val,
                    "doc_id": doc_id,
                    "sent_text": sent_text,
                })

        elapsed = (time.perf_counter() - t0) * 1000.0

        if not extracted_evidences:
            # Absence of evidence is NEVER contradiction!
            return VerificationRecord(
                claim_id=claim.claim_id,
                verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
                confidence=0.95,
                explanation=f"No indexed documents contain verified evidence for entity '{claim.subject_entity}' and attribute '{claim.attribute_name}'.",
                subject_entity=claim.subject_entity,
                attribute_name=claim.attribute_name,
                claimed_value=claim.claimed_value,
                latency_ms=elapsed,
            )

        # Check for conflicting evidence among multiple sources
        unique_vals = set(e["val_str"] for e in extracted_evidences)
        if len(unique_vals) > 1:
            # Verify if they normalize to the same value
            norm_set = set()
            for v in unique_vals:
                n = self.normalize_numeric_string(v)
                norm_set.add(n if n is not None else v.lower())
            if len(norm_set) > 1:
                return VerificationRecord(
                    claim_id=claim.claim_id,
                    verdict=VerificationVerdict.CONFLICTING_EVIDENCE,
                    confidence=0.88,
                    explanation=f"Multiple indexed documents disagree on the value of '{claim.attribute_name}'.",
                    subject_entity=claim.subject_entity,
                    attribute_name=claim.attribute_name,
                    claimed_value=claim.claimed_value,
                    evidence_value=list(unique_vals)[0],
                    conflicting_sources=extracted_evidences,
                    latency_ms=elapsed,
                )

        # Compare claim value to evidence value
        evidence = extracted_evidences[0]
        ev_val_str = evidence["val_str"]
        doc_id = evidence["doc_id"]
        sent_text = evidence["sent_text"]

        is_match = False

        # Attempt Byte Normalization
        claimed_bytes = self.normalize_bytes(claim.claimed_value)
        ev_bytes = self.normalize_bytes(ev_val_str)
        if claimed_bytes is not None and ev_bytes is not None:
            is_match = (claimed_bytes == ev_bytes)
        else:
            # Attempt Time Normalization
            claimed_time = self.normalize_time_ms(claim.claimed_value)
            ev_time = self.normalize_time_ms(ev_val_str)
            if claimed_time is not None and ev_time is not None:
                is_match = (claimed_time == ev_time)
            else:
                # Attempt Pure Numeric Normalization
                c_num = self.normalize_numeric_string(claim.claimed_value)
                e_num = self.normalize_numeric_string(ev_val_str)
                if c_num is not None and e_num is not None:
                    is_match = (abs(c_num - e_num) < 1e-5)
                else:
                    # String comparison
                    is_match = (claim.claimed_value.lower().strip() == ev_val_str.lower().strip())

        if is_match:
            return VerificationRecord(
                claim_id=claim.claim_id,
                verdict=VerificationVerdict.SUPPORTED,
                confidence=0.96,
                explanation=f"Claim is supported: verified '{claim.attribute_name}' = '{ev_val_str}' in '{doc_id}'.",
                subject_entity=claim.subject_entity,
                attribute_name=claim.attribute_name,
                claimed_value=claim.claimed_value,
                evidence_value=ev_val_str,
                source_doc_id=doc_id,
                supporting_text=sent_text,
                latency_ms=elapsed,
            )
        else:
            return VerificationRecord(
                claim_id=claim.claim_id,
                verdict=VerificationVerdict.CONTRADICTED,
                confidence=0.95,
                explanation=f"Claim is contradicted: claimed '{claim.claimed_value}', but evidence in '{doc_id}' establishes '{ev_val_str}'.",
                subject_entity=claim.subject_entity,
                attribute_name=claim.attribute_name,
                claimed_value=claim.claimed_value,
                evidence_value=ev_val_str,
                source_doc_id=doc_id,
                supporting_text=sent_text,
                latency_ms=elapsed,
            )
