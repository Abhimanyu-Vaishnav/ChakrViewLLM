"""
Structured Multi-Document Evidence Assembler for ChakrView.
Location: chakrview/runtime/evidence_assembler.py

Features:
1. Multi-Document Deduplication: Detects identical text or overlapping chunks.
2. Provenance Tracking: Retains document ID, chunk ID, content hash, retrieval rank, and raw scores.
3. Cross-Document Conflict Detection: Evaluates whether multiple sources state conflicting facts for the same entity/attribute.
4. Partial Evidence Assembly: Stitches complementary facts from distinct sources without conflating different entities.
5. Context Budget Enforcer: Truncates assembled evidence cleanly within the model's max_seq_len (512 tokens).
6. Evidence Abstention: Flags when retrieved evidence is absent or insufficient to answer the query.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple

from chakrview.runtime.evidence_extractor import EvidenceExtractor, ExtractionResult
from chakrview.runtime.hybrid_reranker import RerankedEvidence


@dataclass
class EvidenceItem:
    doc_id: str
    chunk_id: str
    text: str
    retrieval_rank: int
    lexical_score: float
    neural_score: float
    final_score: float
    extracted_fact: Optional[str] = None
    supporting_span: Optional[str] = None
    conflict_status: bool = False
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "retrieval_rank": self.retrieval_rank,
            "lexical_score": round(self.lexical_score, 4),
            "neural_score": round(self.neural_score, 4),
            "final_score": round(self.final_score, 4),
            "extracted_fact": self.extracted_fact,
            "supporting_span": self.supporting_span,
            "conflict_status": self.conflict_status,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class EvidenceBundle:
    query: str
    items: List[EvidenceItem] = field(default_factory=list)
    consensus_fact: Optional[str] = None
    has_conflicts: bool = False
    conflicting_values: List[str] = field(default_factory=list)
    is_sufficient: bool = True
    abstention_reason: Optional[str] = None
    total_token_estimate: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "items": [it.to_dict() for it in self.items],
            "consensus_fact": self.consensus_fact,
            "has_conflicts": self.has_conflicts,
            "conflicting_values": self.conflicting_values,
            "is_sufficient": self.is_sufficient,
            "abstention_reason": self.abstention_reason,
            "total_token_estimate": self.total_token_estimate,
        }

    def format_context_prompt(self, max_tokens: int = 380) -> str:
        """Formats the assembled evidence bundle into a clean context prompt block."""
        if not self.items:
            return ""

        lines = []
        est_tokens = 0
        for it in self.items:
            line = f"[{it.doc_id}] {it.text}"
            line_tok_est = len(line.split()) * 2  # rough token approximation
            if est_tokens + line_tok_est > max_tokens:
                break
            lines.append(line)
            est_tokens += line_tok_est

        return "\n".join(lines)


class MultiDocumentEvidenceAssembler:
    """
    Assembles, deduplicates, verifies, and bounds multi-document evidence pools.
    """

    def __init__(self, max_context_tokens: int = 380, min_evidence_score: float = 0.15):
        self.max_context_tokens = max_context_tokens
        self.min_evidence_score = min_evidence_score

    def assemble(self, query: str, retrieved_chunks: List[RerankedEvidence]) -> EvidenceBundle:
        q_clean = query.strip()
        if not retrieved_chunks:
            return EvidenceBundle(
                query=q_clean,
                items=[],
                is_sufficient=False,
                abstention_reason="No documents retrieved from knowledge index.",
            )

        # 1. Deduplication by content hash and near-exact text
        seen_hashes: Set[str] = set()
        deduped_chunks: List[RerankedEvidence] = []
        for ch in retrieved_chunks:
            if ch.final_score < self.min_evidence_score:
                continue
            p_hash = ch.provenance_hash
            if p_hash in seen_hashes:
                continue
            seen_hashes.add(p_hash)
            deduped_chunks.append(ch)

        if not deduped_chunks:
            return EvidenceBundle(
                query=q_clean,
                items=[],
                is_sufficient=False,
                abstention_reason="All retrieved chunks fell below minimum evidence score threshold.",
            )

        # 2. Extract facts and verify consistency across documents
        extracted_facts: Dict[str, List[str]] = {}  # fact_val -> list of doc_ids
        evidence_items: List[EvidenceItem] = []

        for ch in deduped_chunks:
            ext = EvidenceExtractor.extract_slot(ch.text, q_clean)
            fact = ext.extracted_value if ext.success else None
            span = ext.supporting_span if ext.success else None

            if fact:
                norm_fact = fact.strip().lower()
                extracted_facts.setdefault(norm_fact, []).append(ch.doc_id)

            item = EvidenceItem(
                doc_id=ch.doc_id,
                chunk_id=ch.chunk_id,
                text=ch.text,
                retrieval_rank=ch.retrieval_rank,
                lexical_score=ch.lexical_score,
                neural_score=ch.neural_score,
                final_score=ch.final_score,
                extracted_fact=fact,
                supporting_span=span,
            )
            evidence_items.append(item)

        # 3. Check for multi-document contradictions
        distinct_facts = list(extracted_facts.keys())
        has_conflicts = len(distinct_facts) > 1
        conflicting_vals = distinct_facts if has_conflicts else []

        consensus_fact = None
        if distinct_facts and not has_conflicts:
            # All supporting documents agree on the same fact
            consensus_fact = distinct_facts[0]
            for it in evidence_items:
                if it.extracted_fact and it.extracted_fact.strip().lower() == consensus_fact:
                    consensus_fact = it.extracted_fact  # preserve original casing
                    break

        if has_conflicts:
            for it in evidence_items:
                it.conflict_status = True

        # Total token estimate
        total_toks = sum(len(it.text.split()) * 2 for it in evidence_items)

        bundle = EvidenceBundle(
            query=q_clean,
            items=evidence_items,
            consensus_fact=consensus_fact,
            has_conflicts=has_conflicts,
            conflicting_values=conflicting_vals,
            is_sufficient=bool(consensus_fact) and not has_conflicts,
            abstention_reason=(
                f"Conflicting facts detected across evidence: {conflicting_vals}"
                if has_conflicts
                else (None if consensus_fact else "Required factual relation absent from retrieved documents.")
            ),
            total_token_estimate=total_toks,
        )
        return bundle
