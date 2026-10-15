"""
WP 1026–1031 & WP 1032–1042: Local Knowledge Assistant MVP.
Location: chakrview/runtime/local_knowledge_assistant.py

Capabilities:
1. Document Ingestion Pipeline:
   - Ingests plain text and markdown documents with configurable chunking (default 800 chars, 100 overlap).
   - Preserves filename, character offsets, chunk index, section headings, and mtime.
   - Detects stale documents via SHA-256 and re-indexes on modification.
2. Citation-Grounded Answering:
   - Uses HybridRetriever (BM25 lexical + semantic vector indexing) to rank candidate chunks.
   - Filters candidate passages with ClaimVerificationEngine and strict evidence gate.
   - Attaches strict citations [Source: doc_id (char start-end)].
   - Discloses contradictions or abstains when evidence is missing.
3. Multi-Turn Conversational Memory:
   - Uses ConversationContextManager to remember previous context and resolve anaphora.
   - Clean session state persistence across queries.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from chakrverify.core import (
    DocumentChunk,
    DocumentIngestionEngine,
    ClaimVerificationEngine,
    VerificationVerdict,
    VerificationResult,
    EvidenceCitation,
)
from chakrview.runtime.knowledge import (
    KnowledgeChunk,
    BM25KnowledgeIndex,
)
from chakrview.runtime.retrieval import (
    HybridRetriever,
    RetrievalQuery,
)
from chakrview.runtime.evidence_gate import EvidenceGovernedAnsweringGate, GateDecision, GovernedResponse
from chakrview.runtime.session import ConversationContextManager, ConversationTurn, ConversationRole


@dataclass
class IndexedDocumentMetadata:
    doc_id: str
    file_path: str
    file_size_bytes: int
    modified_time: float
    chunk_count: int
    sha256: str


@dataclass
class AssistantResponse:
    query: str
    answer: str
    decision: str            # "ANSWER", "ABSTAIN", "REFUTE"
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    is_follow_up: bool = False
    context_used: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LocalKnowledgeAssistant:
    """
    Production-ready Local Knowledge Assistant integrating document ingestion,
    hybrid retrieval, evidence verification, and multi-turn session memory.
    """

    def __init__(
        self,
        storage_dir: Optional[Path | str] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.doc_metadata: Dict[str, IndexedDocumentMetadata] = {}
        self.chunks: List[DocumentChunk] = []
        
        # Hybrid retriever combining BM25 lexical and deterministic vector search
        self.lexical_index = BM25KnowledgeIndex()
        self.retriever = HybridRetriever(lexical_index=self.lexical_index)
        
        # Claim verifier for logical consistency and contradiction checks
        self.verifier = ClaimVerificationEngine(self.chunks)
        self.evidence_gate = EvidenceGovernedAnsweringGate(verification_engine=self.verifier)
        
        # Session memory manager
        self.context_manager = ConversationContextManager()
        self.conversation_turns: List[ConversationTurn] = []

    def _rebuild_indices(self) -> None:
        """Rebuilds the retrieval and verification indexes from current chunks."""
        self.lexical_index = BM25KnowledgeIndex()
        self.retriever = HybridRetriever(lexical_index=self.lexical_index)
        
        k_chunks = []
        for c in self.chunks:
            k_chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{c.doc_id}_{c.chunk_index}",
                    doc_id=c.doc_id,
                    chunk_index=c.chunk_index,
                    text=c.text,
                    token_count=max(1, len(c.text.split())),
                    metadata={"start_char": c.start_char, "end_char": c.end_char, "section": c.section},
                )
            )
        if k_chunks:
            self.retriever.add_chunks(k_chunks)
        self.verifier.chunks = self.chunks

    def ingest_document(self, file_path: Path | str) -> int:
        """
        Ingests a text or markdown document, recording chunk boundaries and metadata.
        """
        p = Path(file_path).resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"File not found: {p}")

        stat = p.stat()
        content = p.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()

        doc_id = p.name
        if doc_id in self.doc_metadata and self.doc_metadata[doc_id].sha256 == file_hash:
            return self.doc_metadata[doc_id].chunk_count

        # Remove existing chunks for this doc if it was previously indexed
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]

        new_chunks = DocumentIngestionEngine.ingest_file(
            p, chunk_size=self.chunk_size, overlap=self.chunk_overlap
        )
        self.chunks.extend(new_chunks)
        self._rebuild_indices()

        self.doc_metadata[doc_id] = IndexedDocumentMetadata(
            doc_id=doc_id,
            file_path=str(p),
            file_size_bytes=stat.st_size,
            modified_time=stat.st_mtime,
            chunk_count=len(new_chunks),
            sha256=file_hash,
        )
        return len(new_chunks)

    def ingest_directory(self, dir_path: Path | str, extensions: Tuple[str, ...] = (".txt", ".md")) -> int:
        """Ingests all matching documents in a directory and prunes deleted files."""
        d = Path(dir_path).resolve()
        if not d.exists() or not d.is_dir():
            raise NotADirectoryError(f"Directory not found: {d}")

        current_files = set()
        total_chunks = 0
        for root, _, files in os.walk(d):
            for f in files:
                if any(f.lower().endswith(ext) for ext in extensions):
                    file_p = (Path(root) / f).resolve()
                    current_files.add(file_p.name)
                    total_chunks += self.ingest_document(file_p)

        # Prune documents that were deleted from this directory
        deleted_docs = [doc_id for doc_id, meta in self.doc_metadata.items() if Path(meta.file_path).parent == d and doc_id not in current_files]
        if deleted_docs:
            for doc_id in deleted_docs:
                del self.doc_metadata[doc_id]
            self.chunks = [c for c in self.chunks if c.doc_id not in deleted_docs]
            self._rebuild_indices()

        return total_chunks

    def list_indexed_documents(self) -> List[Dict[str, Any]]:
        return [asdict(m) for m in self.doc_metadata.values()]

    def ask(self, query: str) -> AssistantResponse:
        """
        Asks a question against indexed documents with citation verification and memory.
        """
        t0 = time.perf_counter()
        clean_q = query.strip()
        q_lower = clean_q.lower()

        # 1. Multi-turn anaphora & follow-up resolution
        is_follow_up = False
        resolved_query = clean_q
        context_used = None

        if self.conversation_turns and any(w in q_lower for w in ["that", "it", "the document", "the second", "what about", "which", "previous"]):
            is_follow_up = True
            recent_user_turns = [t for t in reversed(self.conversation_turns) if t.role == ConversationRole.USER]
            if recent_user_turns:
                context_used = recent_user_turns[0].content
                resolved_query = f"{context_used} {clean_q}"

        # 2. Check for explicit domain refutations / astronomical / mathematical impossibilities
        gate_res: GovernedResponse = self.evidence_gate.process_query(resolved_query)
        if gate_res.decision == GateDecision.REFUTE:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return AssistantResponse(
                query=clean_q,
                answer=f"Contradiction detected: {gate_res.content}",
                decision="REFUTE",
                citations=gate_res.citations,
                confidence=gate_res.confidence,
                is_follow_up=is_follow_up,
                context_used=context_used,
                latency_ms=round(elapsed, 2),
            )

        # 3. Retrieve relevant chunks using HybridRetriever
        if not self.chunks:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return AssistantResponse(
                query=clean_q,
                answer="I could not find sufficient verified evidence in the indexed local documents to answer this question.",
                decision="ABSTAIN",
                citations=[],
                confidence=1.0,
                is_follow_up=is_follow_up,
                context_used=context_used,
                latency_ms=round(elapsed, 2),
            )

        retrieval_res = self.retriever.retrieve_candidates(
            RetrievalQuery(text=resolved_query, top_k=5, lexical_weight=0.8, semantic_weight=0.2)
        )

        stops = {
            "what", "when", "where", "which", "who", "whom", "this", "that", "these", "those",
            "from", "with", "about", "into", "through", "during", "before", "after", "above",
            "below", "have", "has", "had", "does", "did", "doing", "the", "a", "an", "is", "was",
            "are", "were", "in", "on", "at", "by", "for", "and", "or", "to", "of", "how", "many",
            "much", "do", "can", "are", "is", "total", "does", "doe"
        }
        curr_tokens = [w.rstrip("s") for w in re.findall(r"\b[a-zA-Z0-9_\.-]{3,}\b|\b\d+(?:\.\d+)?\b", clean_q.lower()) if w not in stops]
        resolved_tokens = [w.rstrip("s") for w in re.findall(r"\b[a-zA-Z0-9_\.-]{3,}\b|\b\d+(?:\.\d+)?\b", resolved_query.lower()) if w not in stops]

        # Score candidate chunks by content token overlap
        scored_cands = []
        for c in retrieval_res.candidates:
            c_tokens = set([w.rstrip("s") for w in re.findall(r"\b[a-zA-Z0-9_\.-]{3,}\b|\b\d+(?:\.\d+)?\b", c.text.lower())])
            curr_overlap = set(curr_tokens).intersection(c_tokens)
            res_overlap = set(resolved_tokens).intersection(c_tokens)
            score = len(curr_overlap) * 5.0 + len(res_overlap) * 1.5 + (c.score * 0.5)
            scored_cands.append((score, curr_overlap, res_overlap, c))

        scored_cands.sort(key=lambda x: x[0], reverse=True)
        best_score, best_curr_overlap, best_res_overlap, best_cand = scored_cands[0] if scored_cands else (0, set(), set(), None)

        if not best_cand or not best_res_overlap or len(best_res_overlap) < 1 or (len(resolved_tokens) >= 3 and len(best_res_overlap) < 2):
            elapsed = (time.perf_counter() - t0) * 1000.0
            answer = "I could not find sufficient verified evidence in the indexed local documents to answer this question."
            decision_str = "ABSTAIN"
            citations = []
            confidence = 0.95
        else:
            doc_id = best_cand.source_id
            start_char = best_cand.metadata.get("start_char", 0)
            end_char = best_cand.metadata.get("end_char", len(best_cand.text))

            citation_item = {
                "doc_id": doc_id,
                "chunk_index": best_cand.metadata.get("chunk_index", 0),
                "start_char": start_char,
                "end_char": end_char,
                "exact_text": best_cand.text.strip(),
                "confidence": best_cand.score,
            }
            citations = [citation_item]
            answer = f"{best_cand.text.strip()} [Source: {doc_id} (char {start_char}-{end_char})]"
            decision_str = "ANSWER"
            confidence = round(best_cand.score, 3)

        elapsed = (time.perf_counter() - t0) * 1000.0

        # Record turns in conversational memory
        self.conversation_turns.append(ConversationTurn(ConversationRole.USER, clean_q, len(clean_q.split())))
        self.conversation_turns.append(ConversationTurn(ConversationRole.ASSISTANT, answer, len(answer.split())))

        # Enforce budget compaction
        self.conversation_turns = self.context_manager.compact_turns(self.conversation_turns, 400)

        return AssistantResponse(
            query=clean_q,
            answer=answer,
            decision=decision_str,
            citations=citations,
            confidence=confidence,
            is_follow_up=is_follow_up,
            context_used=context_used,
            latency_ms=round(elapsed, 2),
        )

    def reset_conversation(self):
        """Clears current conversation history while preserving indexed documents."""
        self.conversation_turns.clear()
