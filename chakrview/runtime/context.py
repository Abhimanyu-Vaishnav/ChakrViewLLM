"""
Context Assembly & Prompt Budget Allocator for ChakrView (Step 11 & Step 12).

Provides deterministic context composition respecting the strict ModelConfig
maximum sequence horizon (T <= 512).

Architectural Contracts:
- Strictly enforces 512-token ceiling:
    System Tokens + Memory Tokens + Knowledge Tokens + History Tokens + User Query Tokens + Generation Budget <= 512
- Enforces unambiguous instruction-data boundaries:
    --- WORKING MEMORY START ---
    ...
    --- WORKING MEMORY END ---
    --- KNOWLEDGE CONTEXT START ---
    ...
    --- KNOWLEDGE CONTEXT END ---
    --- CONVERSATION HISTORY START ---
    ...
    --- CONVERSATION HISTORY END ---
    --- USER QUERY ---
    ...
- Working memory and external knowledge are strictly passive DATA, never executable instructions.
- Conversation history drops oldest turns first when budget is constrained.
- Preserves fine-grained provenance citations for all injected knowledge.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union

from chakrview.runtime.knowledge import KnowledgeChunk, KnowledgeProvenance
from chakrview.runtime.memory import MemoryItem, MemoryType, ConversationTurn


@dataclass
class ContextBudget:
    """
    Token budget allocator enforcing sequence length constraints.
    
    Attributes:
        max_context: Total model sequence length ceiling (512 in ChakrMicro).
        generation_budget: Reserved token quota for autoregressive generation.
        max_system_tokens: Maximum tokens allocated for system/skill policy prompts.
        max_query_tokens: Maximum tokens allocated for user query.
        max_memory_tokens: Maximum tokens allocated for working memory context.
        max_knowledge_tokens: Maximum tokens allocated for external knowledge context.
        max_history_tokens: Maximum tokens allocated for recent conversation turns.
    """
    max_context: int = 512
    generation_budget: int = 64
    max_system_tokens: int = 64
    max_query_tokens: int = 128
    max_memory_tokens: int = 140
    max_knowledge_tokens: int = 240
    max_history_tokens: int = 100

    def __post_init__(self) -> None:
        if self.max_context > 512:
            raise ValueError(f"max_context cannot exceed model limit 512, got {self.max_context}")
        if self.generation_budget <= 0:
            raise ValueError(f"generation_budget must be positive, got {self.generation_budget}")
        if self.max_prompt_tokens <= 0:
            raise ValueError("generation_budget must be strictly less than max_context.")

    @property
    def max_prompt_tokens(self) -> int:
        """Total tokens available for the input prompt before generation."""
        return self.max_context - self.generation_budget


@dataclass
class AssembledContext:
    """
    Structured outcome of the prompt context assembly process.
    
    Attributes:
        full_prompt: Assembled string ready for model tokenization.
        system_prompt: Resolved system/policy component.
        knowledge_context: Formatted knowledge block (if any).
        user_query: User query component.
        provenance: Citations for all knowledge chunks included in the prompt.
        estimated_prompt_tokens: Estimated or measured prompt token count.
        knowledge_used: True if at least one knowledge chunk was injected.
        truncated: True if knowledge, memory, history, or query was truncated to fit budget.
        memory_context: Formatted working memory block (if any).
        conversation_history: Formatted conversation history block (if any).
        memories_used: List of MemoryItem objects included in prompt.
        turns_included: List of ConversationTurn objects included in prompt.
    """
    full_prompt: str
    system_prompt: str
    knowledge_context: str
    user_query: str
    provenance: List[KnowledgeProvenance] = field(default_factory=list)
    estimated_prompt_tokens: int = 0
    knowledge_used: bool = False
    truncated: bool = False
    memory_context: str = ""
    conversation_history: str = ""
    memories_used: List[MemoryItem] = field(default_factory=list)
    turns_included: List[ConversationTurn] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "full_prompt": self.full_prompt,
            "system_prompt": self.system_prompt,
            "knowledge_context": self.knowledge_context,
            "user_query": self.user_query,
            "provenance": [p.to_dict() for p in self.provenance],
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "knowledge_used": self.knowledge_used,
            "truncated": self.truncated,
            "memory_context": self.memory_context,
            "conversation_history": self.conversation_history,
            "memories_used": [m.to_dict() for m in self.memories_used],
            "turns_included": [t.to_dict() for t in self.turns_included],
        }


class PromptContextBuilder:
    """
    Deterministic context builder and prompt assembler for ChakrView.
    
    Separates system instructions, passive data context (memory, RAG),
    conversation history, and user queries with unambiguous boundary markers.
    """

    MEMORY_START = "--- WORKING MEMORY START ---"
    MEMORY_END = "--- WORKING MEMORY END ---"
    KNOWLEDGE_START = "--- KNOWLEDGE CONTEXT START ---"
    KNOWLEDGE_END = "--- KNOWLEDGE CONTEXT END ---"
    HISTORY_START = "--- CONVERSATION HISTORY START ---"
    HISTORY_END = "--- CONVERSATION HISTORY END ---"
    QUERY_HEADER = "--- USER QUERY ---"

    def __init__(self, default_budget: Optional[ContextBudget] = None) -> None:
        self.default_budget = default_budget or ContextBudget()

    @staticmethod
    def estimate_tokens(text: str, tokenizer: Optional[Any] = None) -> int:
        """Estimate token count using tokenizer if available, otherwise word count * 1.3."""
        if not text:
            return 0
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            try:
                return len(tokenizer.encode(text, add_bos=False, add_eos=False))
            except Exception:
                pass
        return max(1, int(len(text.split()) * 1.3))

    def build_prompt(
        self,
        user_query: str,
        system_prompt: str = "",
        retrieved_chunks: Optional[List[Tuple[KnowledgeChunk, float]]] = None,
        working_memories: Optional[Union[List[MemoryItem], List[Tuple[MemoryItem, float]]]] = None,
        conversation_turns: Optional[List[ConversationTurn]] = None,
        budget: Optional[ContextBudget] = None,
        tokenizer: Optional[Any] = None,
        unified_candidates: Optional[List[Any]] = None,
    ) -> AssembledContext:
        """
        Assemble a bounded prompt string within the configured token budget.
        
        Strict priority order when budget is exceeded:
        1. Current User Query & System Instructions (strictly preserved)
        2. Relevant Working Memory (up to max_memory_tokens)
        3. Relevant RAG Knowledge (up to max_knowledge_tokens)
        4. Recent Conversation Turns (up to max_history_tokens; oldest turns evicted first)
        
        Guarantees:
            prompt_tokens + generation_budget <= max_context (512 tokens)
        """
        b = budget or self.default_budget
        truncated = False

        # Partition unified retrieval candidates if provided (Step 13)
        effective_chunks = list(retrieved_chunks or [])
        effective_memories = list(working_memories or [])

        if unified_candidates:
            for cand in unified_candidates:
                cand_type = getattr(cand, "source_type", None)
                src_val = cand_type.value if hasattr(cand_type, "value") else str(cand_type)
                if src_val in ("knowledge", "document"):
                    chunk = KnowledgeChunk(
                        chunk_id=cand.candidate_id,
                        doc_id=cand.source_id,
                        chunk_index=cand.metadata.get("chunk_index", 0),
                        text=cand.text,
                        token_count=cand.token_count,
                        metadata=dict(cand.metadata),
                        chunk_hash=cand.metadata.get("chunk_hash", ""),
                    )
                    effective_chunks.append((chunk, cand.score))
                elif src_val in ("memory", "conversation_summary"):
                    mem_type_str = cand.metadata.get("memory_type", "fact")
                    try:
                        m_type = MemoryType(mem_type_str)
                    except Exception:
                        m_type = MemoryType.FACT
                    mem_item = MemoryItem(
                        memory_id=cand.candidate_id,
                        content=cand.text,
                        memory_type=m_type,
                        importance=cand.metadata.get("importance", 0.7),
                        token_estimate=cand.token_count,
                        source_turn_id=cand.metadata.get("source_turn_id"),
                        created_sequence=cand.metadata.get("created_sequence", 0),
                        metadata=dict(cand.metadata),
                    )
                    effective_memories.append((mem_item, cand.score))

        # 1. Budget System Prompt
        sys_clean = system_prompt.strip()
        sys_tokens = self.estimate_tokens(sys_clean, tokenizer)
        if sys_tokens > b.max_system_tokens:
            words = sys_clean.split()
            keep_words = max(1, int(b.max_system_tokens / 1.3))
            sys_clean = " ".join(words[:keep_words])
            sys_tokens = self.estimate_tokens(sys_clean, tokenizer)
            truncated = True

        # 2. Budget User Query (Preserved with high priority)
        query_clean = user_query.strip()
        query_tokens = self.estimate_tokens(query_clean, tokenizer)
        if query_tokens > b.max_query_tokens:
            words = query_clean.split()
            keep_words = max(1, int(b.max_query_tokens / 1.3))
            query_clean = " ".join(words[:keep_words])
            query_tokens = self.estimate_tokens(query_clean, tokenizer)
            truncated = True

        # Estimate base delimiter overhead for query header
        base_query_overhead = self.estimate_tokens(f"{self.QUERY_HEADER}\n\n", tokenizer)
        used_essential = sys_tokens + query_tokens + base_query_overhead

        # Ensure essential components fit within max_prompt_tokens
        if used_essential > b.max_prompt_tokens:
            # Under extreme pressure, trim query further to fit prompt ceiling
            available_for_q = max(10, b.max_prompt_tokens - sys_tokens - base_query_overhead - 5)
            words = query_clean.split()
            keep_words = max(1, int(available_for_q / 1.3))
            query_clean = " ".join(words[:keep_words])
            query_tokens = self.estimate_tokens(query_clean, tokenizer)
            used_essential = sys_tokens + query_tokens + base_query_overhead
            truncated = True

        # Remaining auxiliary budget available for Memory, Knowledge, and History
        avail_aux = max(0, b.max_prompt_tokens - used_essential)

        # 3. Working Memory Allocation
        included_memories: List[MemoryItem] = []
        memory_str = ""
        mem_tokens_accum = 0

        if effective_memories and avail_aux > 10:
            avail_for_mem = min(avail_aux, b.max_memory_tokens)
            mem_delim_overhead = self.estimate_tokens(
                f"{self.MEMORY_START}\n{self.MEMORY_END}\n\n", tokenizer
            )
            avail_for_mem_items = max(0, avail_for_mem - mem_delim_overhead)

            mem_lines: List[str] = []
            for mem_entry in effective_memories:
                mem_item = mem_entry[0] if isinstance(mem_entry, tuple) else mem_entry
                line = f"[{mem_item.memory_type.value.upper()}] {mem_item.content.strip()}"
                line_tokens = self.estimate_tokens(line, tokenizer)
                if mem_tokens_accum + line_tokens <= avail_for_mem_items:
                    mem_lines.append(line)
                    included_memories.append(mem_item)
                    mem_tokens_accum += line_tokens
                elif not mem_lines and line_tokens <= avail_for_mem:
                    mem_lines.append(line)
                    included_memories.append(mem_item)
                    mem_tokens_accum += line_tokens
                    truncated = True
                    break
                else:
                    truncated = True
                    break

            if mem_lines:
                memory_str = f"{self.MEMORY_START}\n" + "\n".join(mem_lines) + f"\n{self.MEMORY_END}"
                total_mem_block_tokens = self.estimate_tokens(memory_str, tokenizer)
                avail_aux = max(0, avail_aux - total_mem_block_tokens)

        # 4. RAG Knowledge Allocation
        included_chunks: List[str] = []
        provenances: List[KnowledgeProvenance] = []
        knowledge_str = ""
        know_tokens_accum = 0

        if effective_chunks and avail_aux > 15:
            avail_for_know = min(avail_aux, b.max_knowledge_tokens)
            know_delim_overhead = self.estimate_tokens(
                f"{self.KNOWLEDGE_START}\n{self.KNOWLEDGE_END}\n\n", tokenizer
            )
            avail_for_know_items = max(0, avail_for_know - know_delim_overhead)

            for chunk, score in effective_chunks:
                chunk_header = f"[Source: {chunk.doc_id} | Chunk: {chunk.chunk_id} | Score: {score:.3f}]"
                chunk_block = f"{chunk_header}\n{chunk.text.strip()}"
                block_tokens = self.estimate_tokens(chunk_block, tokenizer)

                if know_tokens_accum + block_tokens <= avail_for_know_items:
                    included_chunks.append(chunk_block)
                    know_tokens_accum += block_tokens
                    provenances.append(
                        KnowledgeProvenance(
                            doc_id=chunk.doc_id,
                            chunk_id=chunk.chunk_id,
                            source_id=chunk.metadata.get("source_id", "unknown"),
                            content_hash=chunk.chunk_hash,
                            retrieval_score=score,
                            text_preview=chunk.text[:80] + ("..." if len(chunk.text) > 80 else ""),
                            metadata=dict(chunk.metadata),
                        )
                    )
                else:
                    truncated = True
                    break

            if included_chunks:
                knowledge_str = f"{self.KNOWLEDGE_START}\n" + "\n\n".join(included_chunks) + f"\n{self.KNOWLEDGE_END}"
                total_know_block_tokens = self.estimate_tokens(knowledge_str, tokenizer)
                avail_aux = max(0, avail_aux - total_know_block_tokens)

        # 5. Conversation History Allocation (Evicts oldest turns first)
        included_turns: List[ConversationTurn] = []
        history_str = ""
        hist_tokens_accum = 0

        if conversation_turns and avail_aux > 15:
            avail_for_hist = min(avail_aux, b.max_history_tokens)
            hist_delim_overhead = self.estimate_tokens(
                f"{self.HISTORY_START}\n{self.HISTORY_END}\n\n", tokenizer
            )
            avail_for_hist_items = max(0, avail_for_hist - hist_delim_overhead)

            # Traverse turns in reverse (newest to oldest) so newest turns are prioritized
            turns_to_include_rev: List[ConversationTurn] = []
            for turn in reversed(conversation_turns):
                turn_line = f"{turn.role.capitalize()}: {turn.text.strip()}"
                line_tokens = self.estimate_tokens(turn_line, tokenizer)
                if hist_tokens_accum + line_tokens <= avail_for_hist_items:
                    turns_to_include_rev.append(turn)
                    hist_tokens_accum += line_tokens
                else:
                    truncated = True
                    break

            # Reverse back to maintain chronological order
            if turns_to_include_rev:
                included_turns = list(reversed(turns_to_include_rev))
                history_lines = [f"{t.role.capitalize()}: {t.text.strip()}" for t in included_turns]
                history_str = f"{self.HISTORY_START}\n" + "\n".join(history_lines) + f"\n{self.HISTORY_END}"

        # 6. Assemble Full Prompt
        sections: List[str] = []
        if sys_clean:
            sections.append(sys_clean)
        if memory_str:
            sections.append(memory_str)
        if knowledge_str:
            sections.append(knowledge_str)
        if history_str:
            sections.append(history_str)

        query_str = f"{self.QUERY_HEADER}\n{query_clean}"
        sections.append(query_str)

        full_prompt = "\n\n".join(sections)
        total_prompt_tokens = self.estimate_tokens(full_prompt, tokenizer)

        # 7. Final Hard Invariant Safeguard
        # If full_prompt exceeds max_prompt_tokens due to whitespace or token estimation errors,
        # iteratively prune history, then knowledge, then memory.
        while total_prompt_tokens > b.max_prompt_tokens:
            truncated = True
            if included_turns:
                # Drop oldest remaining turn
                included_turns.pop(0)
                if included_turns:
                    history_lines = [f"{t.role.capitalize()}: {t.text.strip()}" for t in included_turns]
                    history_str = f"{self.HISTORY_START}\n" + "\n".join(history_lines) + f"\n{self.HISTORY_END}"
                else:
                    history_str = ""
            elif included_chunks:
                # Drop lowest scoring chunk
                included_chunks.pop()
                if provenances:
                    provenances.pop()
                if included_chunks:
                    knowledge_str = f"{self.KNOWLEDGE_START}\n" + "\n\n".join(included_chunks) + f"\n{self.KNOWLEDGE_END}"
                else:
                    knowledge_str = ""
            elif included_memories:
                # Drop lowest scoring memory
                included_memories.pop()
                if included_memories:
                    mem_lines = [f"[{m.memory_type.value.upper()}] {m.content.strip()}" for m in included_memories]
                    memory_str = f"{self.MEMORY_START}\n" + "\n".join(mem_lines) + f"\n{self.MEMORY_END}"
                else:
                    memory_str = ""
            else:
                # Prune user query to fit
                words = query_clean.split()
                if len(words) > 5:
                    query_clean = " ".join(words[:-5])
                    query_str = f"{self.QUERY_HEADER}\n{query_clean}"
                else:
                    break

            # Re-assemble
            sections = []
            if sys_clean:
                sections.append(sys_clean)
            if memory_str:
                sections.append(memory_str)
            if knowledge_str:
                sections.append(knowledge_str)
            if history_str:
                sections.append(history_str)
            sections.append(query_str)
            full_prompt = "\n\n".join(sections)
            total_prompt_tokens = self.estimate_tokens(full_prompt, tokenizer)

        return AssembledContext(
            full_prompt=full_prompt,
            system_prompt=sys_clean,
            knowledge_context=knowledge_str,
            user_query=query_clean,
            provenance=provenances,
            estimated_prompt_tokens=total_prompt_tokens,
            knowledge_used=len(included_chunks) > 0,
            truncated=truncated,
            memory_context=memory_str,
            conversation_history=history_str,
            memories_used=included_memories,
            turns_included=included_turns,
        )
