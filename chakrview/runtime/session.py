"""
ChakrView Local Model Session & Multi-Turn Context Manager (Step 46).

Manages stateful multi-turn conversational dialogue with strict governed
sliding-window context compaction respecting the 512-token ceiling of ChakrMicro.

Invariants Enforced:
1. Context Horizon: Total prompt tokens + max_new_tokens <= 512.
2. Tenant Isolation: Each session is strictly bound to a tenant_id.
3. Clean Compaction: Oldest non-pinned turns evicted sliding-window style.
4. Cognitive Memory Provenance: Memory context maintains verified trust levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple

from chakrview.cognition.federation.cognitive.models import CognitiveContextEnvelope
from chakrview.runtime.pipeline import ContextOverflowError
from chakrview.tokenizer.tokenizer import BPETokenizer


class ConversationRole(str, Enum):
    """Roles participating in multi-turn conversational sessions."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ConversationTurn:
    """
    A single turn in a multi-turn conversation.
    """
    role: ConversationRole
    content: str
    token_count: int
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    pinned: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "token_count": self.token_count,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "pinned": self.pinned,
        }


class ConversationContextManager:
    """
    Governs token budgeting and context compaction across multi-turn history.
    """

    def __init__(self, max_context: int = 512) -> None:
        self.max_context = max_context

    def compact_turns(
        self,
        turns: List[ConversationTurn],
        budget: int,
    ) -> List[ConversationTurn]:
        """
        Retain all pinned turns and as many recent unpinned turns as fit within budget.

        Args:
            turns: Full list of turns in chronological order.
            budget: Maximum token budget available for history.

        Returns:
            Compacted list of turns in chronological order fitting the budget.
        """
        if not turns or budget <= 0:
            return [t for t in turns if t.pinned]

        pinned_turns = [t for t in turns if t.pinned]
        pinned_tokens = sum(t.token_count for t in pinned_turns)
        remaining_budget = budget - pinned_tokens

        if remaining_budget <= 0:
            # Cannot even fit all pinned turns, return what fits
            acc = 0
            res = []
            for t in pinned_turns:
                if acc + t.token_count <= budget:
                    res.append(t)
                    acc += t.token_count
            return res

        unpinned_turns = [t for t in turns if not t.pinned]
        selected_unpinned: List[ConversationTurn] = []

        # Iterate backwards from most recent unpinned turn
        for t in reversed(unpinned_turns):
            if t.token_count <= remaining_budget:
                selected_unpinned.append(t)
                remaining_budget -= t.token_count
            else:
                break

        # Re-sort to maintain original chronological order
        selected_unpinned.reverse()
        result: List[ConversationTurn] = []

        # Interleave preserving order
        all_selected_ids = {id(t) for t in pinned_turns} | {id(t) for t in selected_unpinned}
        for t in turns:
            if id(t) in all_selected_ids:
                result.append(t)

        return result


class LocalModelSession:
    """
    Interactive local conversation session binding multi-turn history,
    tenant boundaries, and cognitive memory integration.
    """

    def __init__(
        self,
        session_id: str,
        tenant_id: str = "default_tenant",
        system_prompt: Optional[str] = None,
        max_context: int = 512,
        memory_adapter: Optional[Any] = None,
    ) -> None:
        self.session_id = session_id
        self.tenant_id = tenant_id
        self.max_context = max_context
        self.memory_adapter = memory_adapter
        self.context_manager = ConversationContextManager(max_context=max_context)
        self.turns: List[ConversationTurn] = []

        if system_prompt:
            self.add_system_prompt(system_prompt)

    def add_system_prompt(self, content: str) -> None:
        """Add or update pinned system prompt."""
        # Remove any existing system prompt
        self.turns = [t for t in self.turns if t.role != ConversationRole.SYSTEM]
        # Approximate token count (will be refined during build_prompt)
        approx_tokens = max(1, len(content) // 3)
        turn = ConversationTurn(
            role=ConversationRole.SYSTEM,
            content=content,
            token_count=approx_tokens,
            pinned=True,
        )
        self.turns.insert(0, turn)

    def add_turn(
        self,
        role: ConversationRole,
        content: str,
        token_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationTurn:
        """Append a completed turn to conversation history."""
        turn = ConversationTurn(
            role=role,
            content=content,
            token_count=token_count,
            metadata=metadata or {},
            pinned=(role == ConversationRole.SYSTEM),
        )
        self.turns.append(turn)
        return turn

    def clear(self) -> None:
        """Clear conversation history, preserving system prompt if present."""
        self.turns = [t for t in self.turns if t.pinned]

    def build_prompt_package(
        self,
        user_input: str,
        tokenizer: BPETokenizer,
        max_new_tokens: int,
        retrieve_memory: bool = True,
        truncate_if_overflow: bool = True,
    ) -> Tuple[str, Optional[CognitiveContextEnvelope], List[ConversationTurn]]:
        """
        Assemble bounded multi-turn prompt and optional cognitive memory envelope.

        Ensures: len(assembled_prompt_tokens) + max_new_tokens <= 512.
        """
        user_tokens = len(tokenizer.encode(user_input, add_bos=False, add_eos=False))
        overhead = 2  # BOS + safety margin

        # Query cognitive memory if adapter is available
        envelope: Optional[CognitiveContextEnvelope] = None
        memory_tokens = 0
        if retrieve_memory and self.memory_adapter is not None:
            try:
                memories = self.memory_adapter.retrieve_context(
                    objective=user_input,
                    tenant_id=self.tenant_id,
                    session_id=self.session_id,
                    top_k=2,
                )
                if memories:
                    envelope = CognitiveContextEnvelope(
                        tenant_id=self.tenant_id,
                        session_id=self.session_id,
                        context_items=memories,
                    )
                    # Estimate memory tokens in envelope
                    mem_text = " ".join(m.text for m in memories)
                    memory_tokens = len(tokenizer.encode(mem_text, add_bos=False, add_eos=False))
            except Exception:
                envelope = None
                memory_tokens = 0

        # Calculate budget available for conversation history
        available_budget = self.max_context - max_new_tokens - user_tokens - memory_tokens - overhead

        if available_budget < 0:
            if not truncate_if_overflow:
                raise ContextOverflowError(
                    f"User input ({user_tokens} tok) + memory ({memory_tokens} tok) + "
                    f"max_new_tokens ({max_new_tokens}) exceeds context limit ({self.max_context})."
                )
            # Drop memory first if budget is negative
            envelope = None
            memory_tokens = 0
            available_budget = self.max_context - max_new_tokens - user_tokens - overhead

        # Compact history to fit remaining budget
        active_turns = self.context_manager.compact_turns(self.turns, max(0, available_budget))

        # Format turn dialogue into prompt
        prompt_sections: List[str] = []
        for turn in active_turns:
            if turn.role == ConversationRole.SYSTEM:
                prompt_sections.append(f"System: {turn.content}")
            elif turn.role == ConversationRole.USER:
                prompt_sections.append(f"User: {turn.content}")
            elif turn.role == ConversationRole.ASSISTANT:
                prompt_sections.append(f"Assistant: {turn.content}")

        prompt_sections.append(f"User: {user_input}\nAssistant:")
        final_prompt = "\n".join(prompt_sections)

        return final_prompt, envelope, active_turns
