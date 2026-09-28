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
import math
from pathlib import Path
import re
from typing import Dict, List, Optional, Any, Tuple, Union


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


@dataclass
class KnowledgeProvenance:
    """
    Structured provenance and citation tracking for retrieved knowledge.
    
    Attributes:
        doc_id: Unique document identifier.
        chunk_id: Unique chunk identifier.
        source_id: Knowledge source identifier.
        content_hash: Cryptographic SHA-256 hash of the chunk content.
        retrieval_score: Score assigned by the retriever.
        text_preview: Brief human-readable snippet.
        metadata: Additional provenance metadata (file path, titles, timestamps).
    """
    doc_id: str
    chunk_id: str
    source_id: str
    content_hash: str
    retrieval_score: float
    text_preview: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeProvenance":
        return cls(**data)


class DocumentChunker:
    """
    Deterministic chunking engine for KnowledgeDocuments.
    
    Segments text into bounded, overlapping chunks with stable identifiers,
    token estimation, and SHA-256 cryptographic provenance.
    """
    def __init__(
        self,
        chunk_size: int = 64,       # Target words per chunk (~80-90 tokens)
        chunk_overlap: int = 16,     # Overlap words between consecutive chunks
        min_chunk_size: int = 8,     # Minimum words to form a standalone chunk
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap must be in [0, chunk_size-1], got {chunk_overlap}"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, doc: KnowledgeDocument) -> List[KnowledgeChunk]:
        """Chunk a KnowledgeDocument deterministically."""
        return self.chunk_text(
            text=doc.content,
            doc_id=doc.doc_id,
            source_id=doc.source_id,
            metadata={"title": doc.title, **doc.metadata},
        )

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        source_id: str = "default_source",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[KnowledgeChunk]:
        """
        Split raw text into deterministic, overlapping chunks.
        """
        clean_text = text.strip()
        if not clean_text:
            return []

        words = clean_text.split()
        if not words:
            return []

        meta = dict(metadata or {})
        chunks: List[KnowledgeChunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunk_idx = 0

        for start in range(0, len(words), step):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]

            # If remaining words are less than min_chunk_size and chunks already exist,
            # fold them into the preceding chunk to prevent tiny fragments
            if len(chunk_words) < self.min_chunk_size and chunks:
                last_chunk = chunks[-1]
                combined_words = last_chunk.text.split() + chunk_words
                last_chunk.text = " ".join(combined_words)
                last_chunk.token_count = max(1, int(len(combined_words) * 1.3))
                last_chunk.chunk_hash = compute_sha256_text(last_chunk.text)
                break

            chunk_str = " ".join(chunk_words)
            chunk_id = f"{doc_id}_c{chunk_idx:04d}"
            token_estimate = max(1, int(len(chunk_words) * 1.3))

            chunk_meta = dict(meta)
            chunk_meta["source_id"] = source_id
            chunk_meta["start_word"] = start
            chunk_meta["end_word"] = end

            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=chunk_idx,
                    text=chunk_str,
                    token_count=token_estimate,
                    metadata=chunk_meta,
                )
            )
            chunk_idx += 1
            if end >= len(words):
                break

        return chunks


class DocumentIngester:
    """
    Lightweight, production-oriented document ingestion pipeline.
    
    Supports .txt, .md, and in-memory strings.
    Sanitizes content, creates KnowledgeDocument, and produces deterministic chunks.
    """
    def __init__(self, chunker: Optional[DocumentChunker] = None) -> None:
        self.chunker = chunker or DocumentChunker()

    def ingest_text(
        self,
        text: str,
        title: str = "",
        doc_id: Optional[str] = None,
        source_id: str = "default_source",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[KnowledgeDocument, List[KnowledgeChunk]]:
        """Ingest plain text or markdown string."""
        if not text or not text.strip():
            raise ValueError("Cannot ingest empty text document.")

        meta = dict(metadata or {})
        assigned_id = doc_id or f"doc_{compute_sha256_text(text)[:12]}"
        assigned_title = title or f"Document {assigned_id}"

        doc = KnowledgeDocument(
            doc_id=assigned_id,
            source_id=source_id,
            title=assigned_title,
            content=text.strip(),
            metadata=meta,
        )
        chunks = self.chunker.chunk_document(doc)
        return doc, chunks

    def ingest_file(
        self,
        file_path: Union[str, Path],
        doc_id: Optional[str] = None,
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[KnowledgeDocument, List[KnowledgeChunk]]:
        """Ingest a .txt, .md, or plaintext file from filesystem."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a regular file: {path}")

        suffix = path.suffix.lower()
        if suffix not in (".txt", ".md", ".markdown", ".rst", ".json", ".log"):
            raise ValueError(
                f"Unsupported file format: '{suffix}'. Supported formats: .txt, .md, .markdown, .json, .log"
            )

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="utf-8", errors="replace")

        src_id = source_id or f"src_file_{path.stem}"
        meta = dict(metadata or {})
        meta["file_path"] = str(path.resolve())
        meta["file_extension"] = suffix
        meta["file_size_bytes"] = path.stat().st_size

        return self.ingest_text(
            text=content,
            title=path.name,
            doc_id=doc_id or f"doc_{path.stem}",
            source_id=src_id,
            metadata=meta,
        )

    def ingest_directory(
        self,
        dir_path: Union[str, Path],
        extensions: Tuple[str, ...] = (".txt", ".md"),
        source_id: Optional[str] = None,
    ) -> List[Tuple[KnowledgeDocument, List[KnowledgeChunk]]]:
        """Ingest all matching files in a directory recursively."""
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {path}")

        results: List[Tuple[KnowledgeDocument, List[KnowledgeChunk]]] = []
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    res = self.ingest_file(file_path, source_id=source_id)
                    results.append(res)
                except ValueError:
                    # Skip empty or whitespace-only files safely
                    continue
        return results


class BM25KnowledgeIndex(KnowledgeIndex):
    """
    Deterministic Okapi BM25 Lexical Knowledge Index.
    
    Provides term-frequency saturation and length-normalized ranking
    without external vector databases or neural embedding dependencies.
    """
    def __init__(
        self,
        index_id: str = "bm25_default",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.index_id = index_id
        self.k1 = k1
        self.b = b
        self._chunks: Dict[str, KnowledgeChunk] = {}
        self._doc_lens: Dict[str, int] = {}
        self._tf: Dict[str, Dict[str, int]] = {}
        self._df: Dict[str, int] = {}
        self._total_doc_len: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Deterministic term extraction."""
        return re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())

    def add_chunks(self, chunks: List[KnowledgeChunk]) -> int:
        added = 0
        for chunk in chunks:
            cid = chunk.chunk_id
            if cid in self._chunks:
                self._remove_chunk_stats(cid)

            terms = self._tokenize(chunk.text)
            term_freqs: Dict[str, int] = {}
            for t in terms:
                term_freqs[t] = term_freqs.get(t, 0) + 1

            for t in term_freqs:
                self._df[t] = self._df.get(t, 0) + 1

            doc_len = len(terms)
            self._doc_lens[cid] = doc_len
            self._total_doc_len += doc_len
            self._tf[cid] = term_freqs
            self._chunks[cid] = chunk
            added += 1

        return added

    def _remove_chunk_stats(self, chunk_id: str) -> None:
        if chunk_id in self._tf:
            for t in self._tf[chunk_id]:
                if t in self._df:
                    self._df[t] -= 1
                    if self._df[t] <= 0:
                        del self._df[t]
            del self._tf[chunk_id]
        if chunk_id in self._doc_lens:
            self._total_doc_len -= self._doc_lens[chunk_id]
            del self._doc_lens[chunk_id]

    def remove_document(self, doc_id: str) -> int:
        to_remove = [cid for cid, chunk in self._chunks.items() if chunk.doc_id == doc_id]
        for cid in to_remove:
            self._remove_chunk_stats(cid)
            del self._chunks[cid]
        return len(to_remove)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeChunk, float]]:
        if not query.strip() or not self._chunks:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        N = len(self._chunks)
        avgdl = (self._total_doc_len / N) if N > 0 else 1.0

        scores: Dict[str, float] = {}

        for t in query_terms:
            if t not in self._df:
                continue
            df = self._df[t]
            # Standard Okapi BM25 IDF with smoothing
            idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))

            for cid, freqs in self._tf.items():
                if t in freqs:
                    tf = freqs[t]
                    dl = self._doc_lens.get(cid, 0)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                    tf_component = (tf * (self.k1 + 1.0)) / denom if denom > 0 else 0.0
                    scores[cid] = scores.get(cid, 0.0) + (idf * tf_component)

        scored_chunks: List[Tuple[KnowledgeChunk, float]] = []
        for cid, score in scores.items():
            if score > 0.0:
                scored_chunks.append((self._chunks[cid], float(score)))

        # Deterministic sorting: highest score first, tie-break by chunk_id
        scored_chunks.sort(key=lambda x: (x[1], x[0].chunk_id), reverse=True)
        return scored_chunks[:top_k]

    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._doc_lens.clear()
        self._tf.clear()
        self._df.clear()
        self._total_doc_len = 0


class LexicalRetriever(Retriever):
    """
    Standard Lexical Retriever over a KnowledgeIndex (BM25 or InMemory).
    """
    def __init__(self, index: KnowledgeIndex, min_score: float = 0.0) -> None:
        self.index = index
        self.min_score = min_score

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[KnowledgeChunk, float]]:
        results = self.index.search(query, top_k=top_k)
        return [(chunk, score) for chunk, score in results if score >= self.min_score]
