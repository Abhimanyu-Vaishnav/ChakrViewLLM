"""
Knowledge Abstraction Layer for ChakrView (Step 9).

Decouples external, domain, and user knowledge from neural model weights.
Allows specialized instances to query documents without base model retraining.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


def compute_sha256_text(text: str) -> str:
    """Compute deterministic SHA-256 digest of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class KnowledgeSource:
    """
    Metadata representation of an external knowledge origin.
    
    Attributes:
        source_id: Unique identifier for the source.
        source_type: Category of source ("file", "directory", "url", "api").
        uri: Location or locator of the source.
        title: Human-readable title.
        metadata: Arbitrary user or system metadata.
        checksum: Optional cryptographic hash of the raw source data.
        created_at: ISO timestamp of registration.
        updated_at: ISO timestamp of last modification.
    """
    source_id: str
    source_type: str
    uri: str
    title: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeSource":
        return cls(**data)


@dataclass
class KnowledgeDocument:
    """
    An ingested document belonging to a knowledge source.
    
    Attributes:
        doc_id: Unique identifier for the document.
        source_id: Parent KnowledgeSource ID.
        title: Title of document.
        content: Extracted plain text content.
        metadata: Document-level attributes (author, domain, mime-type).
        content_hash: SHA-256 digest of content for integrity and deduplication.
    """
    doc_id: str
    source_id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = compute_sha256_text(self.content)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeDocument":
        return cls(**data)


@dataclass
class KnowledgeChunk:
    """
    A segmented portion of a KnowledgeDocument suitable for context insertion.
    
    Attributes:
        chunk_id: Unique identifier for the chunk.
        doc_id: Parent document identifier.
        chunk_index: 0-indexed position within the document.
        text: Segment text content.
        token_count: Estimated or exact token count.
        metadata: Chunk-level metadata (e.g. section, headings).
        chunk_hash: SHA-256 digest of the chunk text.
    """
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_hash: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_hash:
            self.chunk_hash = compute_sha256_text(self.text)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeChunk":
        return cls(**data)


class KnowledgeIndex(ABC):
    """
    Abstract interface for indexing and searching knowledge chunks.
    """

    @abstractmethod
    def add_chunks(self, chunks: List[KnowledgeChunk]) -> int:
        """Add knowledge chunks to the index. Returns count added."""
        pass

    @abstractmethod
    def remove_document(self, doc_id: str) -> int:
        """Remove all chunks associated with a document. Returns count removed."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeChunk, float]]:
        """Search index and return (chunk, relevance_score) tuples ordered by score."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of chunks currently indexed."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from the index."""
        pass


class InMemoryKnowledgeIndex(KnowledgeIndex):
    """
    Deterministic, dependency-free in-memory knowledge index using term overlap scoring.
    
    Ideal for local execution, unit tests, and lightweight domain-specific memory.
    """

    def __init__(self, index_id: str = "default_in_memory") -> None:
        self.index_id = index_id
        self._chunks: Dict[str, KnowledgeChunk] = {}

    def add_chunks(self, chunks: List[KnowledgeChunk]) -> int:
        added = 0
        for chunk in chunks:
            if chunk.chunk_id not in self._chunks:
                self._chunks[chunk.chunk_id] = chunk
                added += 1
            else:
                self._chunks[chunk.chunk_id] = chunk
        return added

    def remove_document(self, doc_id: str) -> int:
        to_remove = [
            cid for cid, chunk in self._chunks.items() if chunk.doc_id == doc_id
        ]
        for cid in to_remove:
            del self._chunks[cid]
        return len(to_remove)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeChunk, float]]:
        if not query.strip() or not self._chunks:
            return []

        query_tokens = set(query.lower().split())
        scored: List[Tuple[KnowledgeChunk, float]] = []

        for chunk in self._chunks.values():
            chunk_tokens = set(chunk.text.lower().split())
            if not chunk_tokens:
                continue
            # Jaccard overlap score
            intersection = query_tokens.intersection(chunk_tokens)
            union = query_tokens.union(chunk_tokens)
            score = len(intersection) / len(union) if union else 0.0
            if score > 0.0:
                scored.append((chunk, score))

        # Sort descending by score, tie-break by chunk_id for determinism
        scored.sort(key=lambda x: (x[1], x[0].chunk_id), reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()


class Retriever(ABC):
    """
    Abstract interface for retrieving knowledge relevant to a prompt or query.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeChunk, float]]:
        """Retrieve relevant knowledge chunks with scores."""
        pass


class SimpleRetriever(Retriever):
    """
    Reference retriever operating over a KnowledgeIndex.
    """

    def __init__(self, index: KnowledgeIndex, min_score: float = 0.0) -> None:
        self.index = index
        self.min_score = min_score

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeChunk, float]]:
        results = self.index.search(query, top_k=top_k)
        return [(chunk, score) for chunk, score in results if score >= self.min_score]


class ContextProvider:
    """
    Formats retrieved knowledge chunks into structured prompt context.
    
    Respects context token budgets and maintains deterministic boundary markers.
    """

    def __init__(
        self,
        retriever: Retriever,
        max_context_tokens: int = 256,
        header: str = "--- KNOWLEDGE CONTEXT ---",
        footer: str = "-------------------------",
    ) -> None:
        self.retriever = retriever
        self.max_context_tokens = max_context_tokens
        self.header = header
        self.footer = footer

    def build_context(self, query: str, top_k: int = 3) -> Tuple[str, List[KnowledgeChunk]]:
        """
        Retrieve chunks and build formatted context string within token budget.
        
        Returns:
            Tuple of (formatted_context_string, list_of_included_chunks).
        """
        results = self.retriever.retrieve(query, top_k=top_k)
        if not results:
            return "", []

        included_chunks: List[KnowledgeChunk] = []
        context_parts: List[str] = [self.header]
        accumulated_tokens = 0

        for chunk, _ in results:
            # Approximate token count by word length if exact count not given
            chunk_tokens = chunk.token_count if chunk.token_count > 0 else len(chunk.text.split())
            if accumulated_tokens + chunk_tokens > self.max_context_tokens and included_chunks:
                # Do not exceed token budget if at least one chunk is already included
                break

            context_parts.append(f"[Source: {chunk.doc_id} | Chunk {chunk.chunk_index}]")
            context_parts.append(chunk.text.strip())
            accumulated_tokens += chunk_tokens
            included_chunks.append(chunk)

        context_parts.append(self.footer)
        return "\n".join(context_parts), included_chunks


class KnowledgeAdapter(ABC):
    """
    Abstract bridge between knowledge retrieval and runtime inference execution.
    """

    @abstractmethod
    def augment_prompt(self, user_prompt: str) -> str:
        """Augment user prompt with relevant external knowledge."""
        pass
