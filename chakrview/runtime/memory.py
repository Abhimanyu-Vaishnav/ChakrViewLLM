"""
Conversational State & Short-Term Working Memory Subsystem for ChakrView (Step 12).

Provides runtime-owned conversational memory, session management, and working
memory ranking while keeping the neural core strictly stateless and frozen.

Key Architectural Guarantees:
- Model remains stateless; runtime owns conversation sessions.
- In-memory by default: zero hidden or automatic disk persistence (privacy-first).
- Strict data/instruction separation: memory items are passive DATA, not instructions.
- Deterministic memory ranking: relevance + importance + recency with deterministic tie-breaking.
- Bounded token footprint: enforces eviction when exceeding short-term working memory budgets.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Dict, List, Optional, Tuple, Any, Set


class MemoryType(str, Enum):
    """Categorization of short-term working memory items."""
    FACT = "fact"
    PREFERENCE = "preference"
    TASK_STATE = "task_state"
    CONSTRAINT = "constraint"
    SUMMARY = "summary"


@dataclass
class ConversationTurn:
    """
    An immutable record of a single turn in a multi-turn conversation.
    
    Attributes:
        turn_id: Unique string identifier for the turn.
        role: Role of the speaker ('system', 'user', 'assistant', 'tool').
        text: Utterance text content.
        sequence_number: Deterministic 0-indexed turn index within the session.
        token_estimate: Estimated token count for context budgeting.
        timestamp_utc: ISO UTC timestamp.
        metadata: Custom metadata (e.g. tools invoked, citations).
    """
    turn_id: str
    role: str
    text: str
    sequence_number: int
    token_estimate: int = 0
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        return cls(**data)


@dataclass
class MemoryItem:
    """
    A single distilled unit of short-term working memory.
    
    Attributes:
        memory_id: Unique identifier for the memory item.
        content: Memory statement or distilled observation.
        memory_type: Category (FACT, PREFERENCE, TASK_STATE, CONSTRAINT, SUMMARY).
        importance: Numerical priority in [0.0, 1.0].
        token_estimate: Estimated token footprint for context budgeting.
        source_turn_id: Optional identifier of the turn where this memory originated.
        created_sequence: Sequence index at which memory was created.
        last_access_sequence: Sequence index at which memory was most recently accessed.
        metadata: Additional provenance metadata.
    """
    memory_id: str
    content: str
    memory_type: MemoryType
    importance: float = 0.5
    token_estimate: int = 0
    source_turn_id: Optional[str] = None
    created_sequence: int = 0
    last_access_sequence: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.importance <= 1.0):
            raise ValueError(f"Memory importance must be in [0.0, 1.0], got {self.importance}")
        if self.token_estimate <= 0 and self.content:
            # Estimate: words * 1.3
            self.token_estimate = max(1, int(len(self.content.split()) * 1.3))

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["memory_type"] = self.memory_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        data = dict(data)
        if isinstance(data.get("memory_type"), str):
            data["memory_type"] = MemoryType(data["memory_type"])
        return cls(**data)


class WorkingMemory:
    """
    Deterministic short-term working memory store.
    
    Responsibilities:
    - Maintains prioritized memory items.
    - Deterministic ranking: score = w_rel*relevance + w_imp*importance + w_rec*recency + w_type*type_weight
    - Enforces memory token budgets by evicting low-priority items.
    - Tie-breaking: (-score, -importance, memory_id).
    """

    TYPE_WEIGHTS = {
        MemoryType.CONSTRAINT: 1.0,
        MemoryType.PREFERENCE: 0.9,
        MemoryType.TASK_STATE: 0.8,
        MemoryType.FACT: 0.6,
        MemoryType.SUMMARY: 0.5,
    }

    def __init__(self, max_tokens: int = 120, max_token_budget: Optional[int] = None) -> None:
        self.max_tokens = max_token_budget if max_token_budget is not None else max_tokens
        self._memories: Dict[str, MemoryItem] = {}

    def __len__(self) -> int:
        return len(self._memories)

    def __iter__(self):
        return iter(self._memories.values())

    def __contains__(self, memory_id: str) -> bool:
        return memory_id in self._memories

    def add(self, item: MemoryItem) -> MemoryItem:
        """Add a pre-constructed MemoryItem directly and enforce budget."""
        self._memories[item.memory_id] = item
        self.enforce_budget(self.max_tokens, current_sequence=item.created_sequence)
        return item

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve memory item by ID."""
        return self._memories.get(memory_id)

    @property
    def total_tokens(self) -> int:
        """Sum of estimated tokens across all active memories."""
        return sum(m.token_estimate for m in self._memories.values())

    def total_token_count(self) -> int:
        return self.total_tokens

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType,
        importance: float = 0.5,
        source_turn_id: Optional[str] = None,
        sequence: int = 0,
        memory_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryItem:
        """Add a new memory item and enforce budget."""
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("Memory content cannot be empty.")

        mid = memory_id or f"mem_{len(self._memories):04d}_{sequence}"
        item = MemoryItem(
            memory_id=mid,
            content=clean_content,
            memory_type=memory_type,
            importance=importance,
            source_turn_id=source_turn_id,
            created_sequence=sequence,
            last_access_sequence=sequence,
            metadata=dict(metadata or {}),
        )
        self._memories[mid] = item
        self.enforce_budget(self.max_tokens, current_sequence=sequence)
        return item

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        sequence: Optional[int] = None,
    ) -> Optional[MemoryItem]:
        """Update an existing memory item."""
        item = self._memories.get(memory_id)
        if item is None:
            return None
        if content is not None:
            item.content = content.strip()
            item.token_estimate = max(1, int(len(item.content.split()) * 1.3))
        if importance is not None:
            if not (0.0 <= importance <= 1.0):
                raise ValueError(f"Importance must be in [0.0, 1.0], got {importance}")
            item.importance = importance
        if sequence is not None:
            item.last_access_sequence = sequence
        return item

    def remove_memory(self, memory_id: str) -> bool:
        """Remove a memory item by ID."""
        if memory_id in self._memories:
            del self._memories[memory_id]
            return True
        return False

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve memory item by ID."""
        return self._memories.get(memory_id)

    def list_memories(self, memory_type: Optional[MemoryType] = None) -> List[MemoryItem]:
        """List all memories, optionally filtered by type."""
        if memory_type is None:
            return list(self._memories.values())
        return [m for m in self._memories.values() if m.memory_type == memory_type]

    def count(self) -> int:
        """Total number of stored memory items."""
        return len(self._memories)

    def clear(self) -> None:
        """Clear all working memory items."""
        self._memories.clear()

    def retrieve_relevant(
        self,
        query: str,
        current_sequence: int = 0,
        top_k: int = 5,
        min_score: float = 0.0,
        active_domain: Optional[str] = None,
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Rank and retrieve relevant memories deterministically.
        
        Score components:
        - relevance: Jaccard term overlap between query and content
        - importance: memory.importance (0 to 1)
        - recency: 1.0 / (1.0 + max(0, current_sequence - last_access_sequence))
        - type_weight: type prior (constraints and preferences score higher)
        """
        if not self._memories:
            return []

        query_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", query.lower()))

        scored_items: List[Tuple[MemoryItem, float]] = []

        for item in self._memories.values():
            content_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", item.content.lower()))
            # Jaccard overlap
            if query_tokens and content_tokens:
                intersection = query_tokens.intersection(content_tokens)
                union = query_tokens.union(content_tokens)
                relevance = len(intersection) / len(union) if union else 0.0
            else:
                relevance = 0.0

            # Recency
            recency_diff = max(0, current_sequence - item.last_access_sequence)
            recency = 1.0 / (1.0 + 0.1 * recency_diff)

            # Type prior
            type_weight = self.TYPE_WEIGHTS.get(item.memory_type, 0.5)

            # Combined score: 40% relevance, 30% importance, 15% recency, 15% type
            score = (
                0.40 * relevance
                + 0.30 * item.importance
                + 0.15 * recency
                + 0.15 * type_weight
            )

            if score >= min_score:
                scored_items.append((item, float(score)))

        # Deterministic sorting: highest score first, highest importance first, tie-break alphabetical on memory_id
        scored_items.sort(key=lambda x: (-x[1], -x[0].importance, x[0].memory_id))

        # Update last_access_sequence for retrieved memories
        for item, _ in scored_items[:top_k]:
            item.last_access_sequence = current_sequence

        return scored_items[:top_k]

    def enforce_budget(self, max_tokens: int, current_sequence: int = 0) -> int:
        """
        Evict lowest-priority memories when total tokens exceed max_tokens.
        Returns the number of evicted memory items.
        """
        if self.total_tokens <= max_tokens:
            return 0

        # Rank all memories by baseline priority: importance * type_weight * recency
        ranked = []
        for item in self._memories.values():
            recency_diff = max(0, current_sequence - item.last_access_sequence)
            recency = 1.0 / (1.0 + 0.1 * recency_diff)
            type_weight = self.TYPE_WEIGHTS.get(item.memory_type, 0.5)
            priority = item.importance * type_weight * recency
            ranked.append((item, priority))

        # Sort ascending by priority: lowest priority items evicted first
        ranked.sort(key=lambda x: (x[1], x[0].importance, x[0].memory_id))

        evicted_count = 0
        for item, _ in ranked:
            if self.total_tokens <= max_tokens:
                break
            del self._memories[item.memory_id]
            evicted_count += 1

        return evicted_count


@dataclass
class ConversationState:
    """
    Runtime-owned state container for an active multi-turn conversation session.
    
    Attributes:
        session_id: Unique session identifier string.
        turns: Ordered list of ConversationTurn records.
        working_memory: WorkingMemory instance managing short-term recall.
        active_skill_id: Active capability skill identifier.
        active_domain: Active capability domain.
        sequence_counter: Monotonically increasing turn counter.
        metadata: Session-level metadata.
    """
    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    active_skill_id: str = "skill_general_v1"
    active_domain: str = "general"
    sequence_counter: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": [t.to_dict() for t in self.turns],
            "working_memory": [m.to_dict() for m in self.working_memory.list_memories()],
            "active_skill_id": self.active_skill_id,
            "active_domain": self.active_domain,
            "sequence_counter": self.sequence_counter,
            "metadata": self.metadata,
        }


class ConversationStore:
    """
    In-memory session registry and conversation coordinator.
    
    Privacy Contract:
    - Pure in-memory storage: zero automatic disk writes.
    - Zero network transmission.
    - Explicit lifecycle: create_session, append_turn, clear, delete.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, ConversationState] = {}

    def create_session(
        self,
        session_id: Optional[str] = None,
        active_skill_id: str = "skill_general_v1",
        active_domain: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationState:
        """Create a new isolated conversation session."""
        sid = session_id or f"session_{len(self._sessions):04d}"
        if sid in self._sessions:
            raise ValueError(f"Session '{sid}' already exists.")

        state = ConversationState(
            session_id=sid,
            active_skill_id=active_skill_id,
            active_domain=active_domain,
            metadata=dict(metadata or {}),
        )
        self._sessions[sid] = state
        return state

    def get_session(self, session_id: str) -> Optional[ConversationState]:
        """Retrieve active conversation session by ID."""
        return self._sessions.get(session_id)

    def get_or_create_session(self, session_id: str) -> ConversationState:
        """Retrieve existing session or instantiate a new one."""
        sess = self.get_session(session_id)
        if sess is None:
            sess = self.create_session(session_id=session_id)
        return sess

    def append_turn(
        self,
        session_id: str,
        role: str,
        text: str,
        token_estimate: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Append a user, assistant, or tool turn to the session."""
        sess = self.get_session(session_id)
        if sess is None:
            raise KeyError(f"Session '{session_id}' not found.")

        seq = sess.sequence_counter
        sess.sequence_counter += 1

        tid = f"{session_id}_t{seq:04d}"
        tokens = token_estimate if token_estimate > 0 else max(1, int(len(text.split()) * 1.3))

        turn = ConversationTurn(
            turn_id=tid,
            role=role,
            text=text.strip(),
            sequence_number=seq,
            token_estimate=tokens,
            metadata=dict(metadata or {}),
        )
        sess.turns.append(turn)
        return turn

    def get_recent_turns(
        self,
        session_id: str,
        max_turns: Optional[int] = None,
        n: Optional[int] = None,
    ) -> List[ConversationTurn]:
        """Retrieve recent conversation turns in chronological order."""
        sess = self.get_session(session_id)
        if sess is None:
            return []
        limit = n if n is not None else max_turns
        if limit is None or limit <= 0:
            return list(sess.turns)
        return sess.turns[-limit:]

    def clear_session(self, session_id: str) -> bool:
        """Reset conversation turns and working memory while keeping the session."""
        sess = self.get_session(session_id)
        if sess is None:
            return False
        sess.turns.clear()
        sess.working_memory.clear()
        sess.sequence_counter = 0
        return True

    def delete_session(self, session_id: str) -> bool:
        """Completely destroy a conversation session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[str]:
        """List all active session identifiers."""
        return list(self._sessions.keys())

    def snapshot(self, session_id: str) -> Dict[str, Any]:
        """Create a complete serializable snapshot of session state."""
        sess = self.get_session(session_id)
        if sess is None:
            raise KeyError(f"Session '{session_id}' not found.")
        return sess.to_dict()


class MemoryExtractor:
    """
    Deterministic rule-based short-term memory extractor.
    
    Identifies explicit user constraints, preferences, and facts without an LLM.
    """

    PATTERNS: List[Tuple[re.Pattern, MemoryType, float]] = [
        (re.compile(r"\b(?:always|never|must not|do not|constraint:|must)\s+(.+)", re.IGNORECASE), MemoryType.CONSTRAINT, 0.95),
        (re.compile(r"\b(?:i prefer|prefer to|my preference is|call me|i like)\s+(.+)", re.IGNORECASE), MemoryType.PREFERENCE, 0.85),
        (re.compile(r"\b(?:current goal is|working on task|task:|task is|our task is|goal is|goal:)\s+(.+)", re.IGNORECASE), MemoryType.TASK_STATE, 0.80),
        (re.compile(r"\b(?:remember that|note that|fact:|the fact is)\s+(.+)", re.IGNORECASE), MemoryType.FACT, 0.75),
    ]

    @classmethod
    def extract_from_text(
        cls,
        text: str,
        session: Optional[ConversationState] = None,
        source_turn_id: Optional[str] = None,
        sequence_num: int = 0,
    ) -> List[MemoryItem]:
        """
        Extract deterministic memory candidates and optionally add to session working memory.
        """
        extracted = []
        seq = session.sequence_counter if session is not None else sequence_num
        for pat, mem_type, imp in cls.PATTERNS:
            match = pat.search(text)
            if match:
                content_str = match.group(0).strip().rstrip(".!?;")
                mid = f"mem_ext_{len(extracted)}_{seq}"
                item = MemoryItem(
                    memory_id=mid,
                    content=content_str,
                    memory_type=mem_type,
                    importance=imp,
                    source_turn_id=source_turn_id,
                    created_sequence=seq,
                    last_access_sequence=seq,
                )
                if session is not None:
                    session.working_memory.add(item)
                extracted.append(item)
        return extracted


class ConversationSummarizer:
    """
    Deterministic rolling conversation summarizer.
    
    Compresses older turns into a structured summary MemoryItem when the history
    exceeds threshold, maintaining provenance to source turns without external models.
    """

    @classmethod
    def maybe_summarize(
        cls,
        session: Optional[ConversationState] = None,
        state: Optional[ConversationState] = None,
        max_turns_threshold: int = 8,
        turns_to_retain: int = 4,
        keep_recent: Optional[int] = None,
    ) -> Optional[MemoryItem]:
        """
        If turns count exceeds threshold, compress older turns into a summary MemoryItem.
        """
        target_session = session or state
        if target_session is None:
            raise ValueError("A ConversationState session must be provided.")
        retain = keep_recent if keep_recent is not None else turns_to_retain
        if len(target_session.turns) <= max_turns_threshold:
            return None

        cutoff = len(target_session.turns) - retain
        turns_to_compress = target_session.turns[:cutoff]

        user_topics = []
        for t in turns_to_compress:
            if t.role == "user":
                words = t.text.split()[:8]
                user_topics.append(" ".join(words))

        summary_text = (
            f"Compressed dialogue (turns 0-{cutoff-1}): Discussed "
            + "; ".join(user_topics[:3])
            + (f" and {len(user_topics)-3} more topics." if len(user_topics) > 3 else ".")
        )

        source_turn_ids = [t.turn_id for t in turns_to_compress]

        item = target_session.working_memory.add_memory(
            content=summary_text,
            memory_type=MemoryType.SUMMARY,
            importance=0.6,
            source_turn_id=turns_to_compress[0].turn_id if turns_to_compress else None,
            sequence=target_session.sequence_counter,
            metadata={"compressed_turns": source_turn_ids, "cutoff": cutoff},
        )

        target_session.turns = target_session.turns[cutoff:]
        return item

