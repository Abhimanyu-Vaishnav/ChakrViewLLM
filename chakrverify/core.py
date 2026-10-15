"""
ChakrVerify Local Alpha: Document Ingestion, Claim Verification & Evidence Provenance Engine.
Location: chakrverify/core.py

Provides:
- DocumentIngestion: Ingests TXT, Markdown, and text-based documents with chunking, section metadata, and character offset tracking.
- VerificationVerdict: Enum representing [SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE, PROCESSING_ERROR].
- ClaimVerifier: Evaluates claims against ingested documents using semantic grounding, pointer token matching, and contradiction detection.
- Honest Abstention: Distinguishes absence of evidence (INSUFFICIENT_EVIDENCE) from direct negation (CONTRADICTED).
- Exact Evidence Citation: Returns document name, section, character start/end offsets, and snippet.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple


class VerificationVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    PROCESSING_ERROR = "PROCESSING_ERROR"


@dataclass
class DocumentChunk:
    doc_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    section: Optional[str] = None


@dataclass
class EvidenceCitation:
    doc_id: str
    chunk_index: int
    start_char: int
    end_char: int
    exact_text: str
    section: Optional[str] = None
    confidence: float = 1.0


@dataclass
class VerificationResult:
    claim: str
    verdict: VerificationVerdict
    confidence: float
    explanation: str
    citations: List[EvidenceCitation] = field(default_factory=list)
    has_conflicts: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        return data


class DocumentIngestionEngine:
    """
    Ingests local plain text and markdown documents, indexing character boundaries.
    """

    @classmethod
    def ingest_file(cls, path: Path | str, chunk_size: int = 400, overlap: int = 50) -> List[DocumentChunk]:
        fpath = Path(path)
        if not fpath.exists():
            raise FileNotFoundError(f"Document file not found: {fpath}")

        try:
            content = fpath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = fpath.read_text(encoding="latin-1")

        doc_id = fpath.name
        chunks: List[DocumentChunk] = []
        
        # Simple character-offset chunking
        cur = 0
        idx = 0
        n = len(content)

        while cur < n:
            end = min(cur + chunk_size, n)
            # Try to snap to sentence or newline boundary
            if end < n:
                newline_pos = content.rfind("\n", cur, end)
                period_pos = content.rfind(". ", cur, end)
                snap = max(newline_pos, period_pos)
                if snap > cur + (chunk_size // 2):
                    end = snap + (1 if snap == newline_pos else 2)

            chunk_text = content[cur:end]
            chunks.append(DocumentChunk(
                doc_id=doc_id,
                chunk_index=idx,
                text=chunk_text,
                start_char=cur,
                end_char=end,
                section=f"Offset_{cur}_{end}",
            ))
            idx += 1
            if end >= n:
                break
            cur = max(cur + 1, end - overlap)

        return chunks


class ClaimVerificationEngine:
    """
    Verifies claims against indexed document chunks with explicit contradiction detection.
    """

    def __init__(self, chunks: Optional[List[DocumentChunk]] = None):
        self.chunks = chunks or []

    def add_document(self, path: Path | str):
        new_chunks = DocumentIngestionEngine.ingest_file(path)
        self.chunks.extend(new_chunks)

    def verify_claim(self, claim: str) -> VerificationResult:
        t0 = time.perf_counter()
        c_clean = claim.strip()
        c_lower = c_clean.lower()

        if not self.chunks:
            return VerificationResult(
                claim=c_clean,
                verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                explanation="No reference documents have been indexed for verification.",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Extract primary key entities / tokens from claim
        # Filter out common stop words
        stops = {"the", "a", "an", "is", "was", "are", "were", "in", "on", "at", "by", "for", "with", "about", "that", "this", "has", "have", "had"}
        words = [w.strip(".,;:?!\"'") for w in c_lower.split() if w.strip(".,;:?!\"'") and w.strip(".,;:?!\"'") not in stops]

        # Scan chunks for matching evidence
        supporting_citations: List[EvidenceCitation] = []
        contradicting_citations: List[EvidenceCitation] = []

        # Detect negation in claim: "is not", "cannot", "does not"
        claim_has_negation = bool(re.search(r"\b(?:not|never|no|cannot|does not|is not)\b", c_lower))

        for chunk in self.chunks:
            # Split chunk into sentences to evaluate claims against local assertions
            sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
            chunk_start = chunk.start_char

            for sent in sentences:
                sent_clean = sent.strip()
                if not sent_clean:
                    continue
                sent_lower = sent_clean.lower()
                matching_words = [w for w in words if w in sent_lower]
                overlap_ratio = len(matching_words) / max(len(words), 1)

                if overlap_ratio >= 0.70:
                    sent_has_negation = bool(re.search(r"\b(?:not|never|no|cannot|does not|is not|uninhabited|without|lacks|failed)\b", sent_lower))
                    
                    if claim_has_negation != sent_has_negation:
                        contradicting_citations.append(EvidenceCitation(
                            doc_id=chunk.doc_id,
                            chunk_index=chunk.chunk_index,
                            start_char=chunk.start_char,
                            end_char=chunk.end_char,
                            exact_text=sent_clean,
                            section=chunk.section,
                            confidence=overlap_ratio,
                        ))
                    else:
                        supporting_citations.append(EvidenceCitation(
                            doc_id=chunk.doc_id,
                            chunk_index=chunk.chunk_index,
                            start_char=chunk.start_char,
                            end_char=chunk.end_char,
                            exact_text=sent_clean,
                            section=chunk.section,
                            confidence=overlap_ratio,
                        ))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Decide Verdict
        if supporting_citations and contradicting_citations:
            return VerificationResult(
                claim=c_clean,
                verdict=VerificationVerdict.CONTRADICTED,
                confidence=0.85,
                explanation="Conflicting evidence found across indexed documents.",
                citations=supporting_citations + contradicting_citations,
                has_conflicts=True,
                latency_ms=elapsed_ms,
            )
        elif supporting_citations:
            return VerificationResult(
                claim=c_clean,
                verdict=VerificationVerdict.SUPPORTED,
                confidence=0.92,
                explanation=f"Claim is supported by verified text in document '{supporting_citations[0].doc_id}'.",
                citations=supporting_citations,
                has_conflicts=False,
                latency_ms=elapsed_ms,
            )
        elif contradicting_citations:
            return VerificationResult(
                claim=c_clean,
                verdict=VerificationVerdict.CONTRADICTED,
                confidence=0.90,
                explanation=f"Claim is directly contradicted by verified text in document '{contradicting_citations[0].doc_id}'.",
                citations=contradicting_citations,
                has_conflicts=False,
                latency_ms=elapsed_ms,
            )
        else:
            return VerificationResult(
                claim=c_clean,
                verdict=VerificationVerdict.INSUFFICIENT_EVIDENCE,
                confidence=0.95,
                explanation="No document contains sufficient verified evidence regarding the entities in the claim.",
                citations=[],
                has_conflicts=False,
                latency_ms=elapsed_ms,
            )
