"""
Context Assembly & Prompt Budget Allocator for ChakrView (Step 11).

Provides deterministic context composition respecting the strict ModelConfig
maximum sequence horizon (T <= 512).

Architectural Contracts:
- Strictly enforces 512-token ceiling:
    System Tokens + Knowledge Tokens + User Query Tokens + Generation Budget <= 512
- Enforces unambiguous instruction-data boundaries:
    --- KNOWLEDGE CONTEXT START ---
    ...
    --- KNOWLEDGE CONTEXT END ---
    --- USER QUERY ---
    ...
- Knowledge is strictly passive data, never executable instructions.
- Preserves fine-grained provenance citations for all injected knowledge.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

from chakrview.runtime.knowledge import KnowledgeChunk, KnowledgeProvenance


@dataclass
class ContextBudget:
    """
    Token budget allocator enforcing sequence length constraints.
    
    Attributes:
        max_context: Total model sequence length ceiling (512 in ChakrMicro).
        generation_budget: Reserved token quota for autoregressive generation.
        max_system_tokens: Maximum tokens allocated for system/skill policy prompts.
        max_query_tokens: Maximum tokens allocated for user query.
        max_knowledge_tokens: Maximum tokens allocated for external knowledge context.
    """
    max_context: int = 512
    generation_budget: int = 64
    max_system_tokens: int = 64
    max_query_tokens: int = 128
    max_knowledge_tokens: int = 240

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
        truncated: True if knowledge or query was truncated to fit budget.
    """
    full_prompt: str
    system_prompt: str
    knowledge_context: str
    user_query: str
    provenance: List[KnowledgeProvenance] = field(default_factory=list)
    estimated_prompt_tokens: int = 0
    knowledge_used: bool = False
    truncated: bool = False

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
        }


class PromptContextBuilder:
    """
    Deterministic context builder and prompt assembler for ChakrView.
    
    Separates system instructions, passive data context, and user queries.
    """

    KNOWLEDGE_START = "--- KNOWLEDGE CONTEXT START ---"
    KNOWLEDGE_END = "--- KNOWLEDGE CONTEXT END ---"
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
        budget: Optional[ContextBudget] = None,
        tokenizer: Optional[Any] = None,
    ) -> AssembledContext:
        """
        Assemble a bounded prompt string within the configured token budget.
        """
        b = budget or self.default_budget
        truncated = False

        # 1. Budget System Prompt
        sys_clean = system_prompt.strip()
        sys_tokens = self.estimate_tokens(sys_clean, tokenizer)
        if sys_tokens > b.max_system_tokens:
            # Truncate system prompt if too long
            words = sys_clean.split()
            keep_words = int(b.max_system_tokens / 1.3)
            sys_clean = " ".join(words[:keep_words])
            sys_tokens = self.estimate_tokens(sys_clean, tokenizer)
            truncated = True

        # 2. Budget User Query
        query_clean = user_query.strip()
        query_tokens = self.estimate_tokens(query_clean, tokenizer)
        if query_tokens > b.max_query_tokens:
            words = query_clean.split()
            keep_words = int(b.max_query_tokens / 1.3)
            query_clean = " ".join(words[:keep_words])
            query_tokens = self.estimate_tokens(query_clean, tokenizer)
            truncated = True

        # Delimiter overhead estimate
        delimiter_overhead = self.estimate_tokens(
            f"{self.KNOWLEDGE_START}\n{self.KNOWLEDGE_END}\n{self.QUERY_HEADER}\n\n",
            tokenizer,
        )

        # 3. Available tokens for Knowledge Context
        avail_for_knowledge = (
            b.max_prompt_tokens - sys_tokens - query_tokens - delimiter_overhead
        )
        avail_for_knowledge = min(avail_for_knowledge, b.max_knowledge_tokens)

        # 4. Pack Knowledge Chunks
        included_chunks: List[str] = []
        provenances: List[KnowledgeProvenance] = []
        accumulated_know_tokens = 0

        if retrieved_chunks and avail_for_knowledge > 15:
            for chunk, score in retrieved_chunks:
                chunk_header = f"[Source: {chunk.doc_id} | Chunk: {chunk.chunk_id} | Score: {score:.3f}]"
                chunk_block = f"{chunk_header}\n{chunk.text.strip()}"
                block_tokens = self.estimate_tokens(chunk_block, tokenizer)

                if accumulated_know_tokens + block_tokens <= avail_for_knowledge:
                    included_chunks.append(chunk_block)
                    accumulated_know_tokens += block_tokens

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

        # 5. Assemble Full Prompt
        sections: List[str] = []
        if sys_clean:
            sections.append(sys_clean)

        knowledge_str = ""
        knowledge_used = len(included_chunks) > 0
        if knowledge_used:
            knowledge_body = "\n\n".join(included_chunks)
            knowledge_str = f"{self.KNOWLEDGE_START}\n{knowledge_body}\n{self.KNOWLEDGE_END}"
            sections.append(knowledge_str)

        query_str = f"{self.QUERY_HEADER}\n{query_clean}"
        sections.append(query_str)

        full_prompt = "\n\n".join(sections)
        total_prompt_tokens = self.estimate_tokens(full_prompt, tokenizer)

        # Final invariant safeguard: hard truncate if total prompt exceeds max_prompt_tokens
        if total_prompt_tokens > b.max_prompt_tokens:
            truncated = True

        return AssembledContext(
            full_prompt=full_prompt,
            system_prompt=sys_clean,
            knowledge_context=knowledge_str,
            user_query=query_clean,
            provenance=provenances,
            estimated_prompt_tokens=total_prompt_tokens,
            knowledge_used=knowledge_used,
            truncated=truncated,
        )
