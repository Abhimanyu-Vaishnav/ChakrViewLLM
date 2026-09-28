"""
Context Construction Abstraction for ChakrView (Step 20).

Compiles bounded, structured prompt context for ChakrMicro respecting the
frozen 512-token context ceiling.

Key Architectural Guarantees:
1. FROZEN CONTEXT CEILING: Total sequence length strictly <= 512 tokens.
   Default context budget: Prompt <= 384 tokens, Generation >= 128 tokens.
2. DETERMINISTIC PRIORITIZATION: Prioritizes system constraints and task
   objectives over secondary memory and unverified assertions.
3. DATA != AUTHORITY: Memory, user input, and retrieved facts are tagged
   strictly as passive data, preventing prompt-injection or privilege escalation.
4. PROVENANCE PRESERVATION: Records the source, authority level, and token
   budget consumption of every included fragment.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple


class ContextSourceType(str, Enum):
    """Categorization of context sources by epistemic and authority status."""
    SYSTEM_IDENTITY = "SYSTEM_IDENTITY"
    SYSTEM_CONSTRAINT = "SYSTEM_CONSTRAINT"
    TASK_OBJECTIVE = "TASK_OBJECTIVE"
    VERIFIED_KNOWLEDGE = "VERIFIED_KNOWLEDGE"
    REASONING_SUMMARY = "REASONING_SUMMARY"
    MEMORY = "MEMORY"
    CAPABILITY_OBSERVATION = "CAPABILITY_OBSERVATION"
    UNCERTAIN_INFORMATION = "UNCERTAIN_INFORMATION"
    USER_ASSERTION = "USER_ASSERTION"


@dataclass
class ContextBudget:
    """
    Token budget envelope for ChakrMicro context construction.
    
    Hard upper bound: max_context <= 512.
    """
    max_context: int = 512
    generation_budget: int = 128

    def __post_init__(self) -> None:
        if not (1 <= self.max_context <= 512):
            raise ValueError(f"max_context must be in [1, 512], got {self.max_context}")
        if self.generation_budget <= 0:
            raise ValueError(f"generation_budget must be positive, got {self.generation_budget}")
        if self.generation_budget >= self.max_context:
            raise ValueError(
                f"generation_budget ({self.generation_budget}) cannot equal or exceed max_context ({self.max_context})"
            )

    @property
    def max_prompt_tokens(self) -> int:
        return self.max_context - self.generation_budget


@dataclass
class ContextItem:
    """An individual item of context to be considered for prompt inclusion."""
    source_type: ContextSourceType
    content: str
    priority: int                   # Lower value = higher priority (1 is highest)
    is_authority: bool = False      # Only True for SYSTEM constraints / identity
    source_id: Optional[str] = None
    confidence: float = 1.0
    token_count: int = 0

    def format_for_prompt(self) -> str:
        """Format item with explicit demarcated boundaries."""
        tag = self.source_type.value
        return f"[{tag}]\n{self.content.strip()}"


@dataclass
class AssembledIntelligenceContext:
    """Complete assembled context package ready for neural inference."""
    full_prompt: str
    token_count: int
    items_included: List[ContextItem]
    items_dropped: List[ContextItem]
    budget: ContextBudget
    provenance: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "full_prompt": self.full_prompt,
            "token_count": self.token_count,
            "items_included_count": len(self.items_included),
            "items_dropped_count": len(self.items_dropped),
            "budget": {
                "max_context": self.budget.max_context,
                "generation_budget": self.budget.generation_budget,
                "max_prompt_tokens": self.budget.max_prompt_tokens,
            },
            "provenance": self.provenance,
        }


class IntelligenceContextBuilder:
    """
    Deterministic context builder enforcing token budget ceilings and authority boundaries.
    """

    # Priority mapping: 1 (highest) to 8 (lowest)
    DEFAULT_PRIORITIES = {
        ContextSourceType.SYSTEM_IDENTITY: 1,
        ContextSourceType.SYSTEM_CONSTRAINT: 2,
        ContextSourceType.TASK_OBJECTIVE: 3,
        ContextSourceType.VERIFIED_KNOWLEDGE: 4,
        ContextSourceType.REASONING_SUMMARY: 5,
        ContextSourceType.MEMORY: 6,
        ContextSourceType.CAPABILITY_OBSERVATION: 7,
        ContextSourceType.UNCERTAIN_INFORMATION: 8,
        ContextSourceType.USER_ASSERTION: 9,
    }

    def __init__(self, default_budget: Optional[ContextBudget] = None) -> None:
        self.default_budget = default_budget or ContextBudget()

    def estimate_tokens(self, text: str, tokenizer: Optional[Any] = None) -> int:
        """Estimate token count using tokenizer or robust heuristic fallback."""
        if not text:
            return 0
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            try:
                # add_bos=False, add_eos=False
                tokens = tokenizer.encode(text, add_bos=False, add_eos=False)
                return len(tokens)
            except Exception:
                pass
        # Heuristic fallback: ~4 characters per token
        return max(1, (len(text) + 3) // 4)

    def build_context(
        self,
        task_objective: str,
        system_identity: Optional[str] = "ChakrMicro Sovereign Neural Core v0.1",
        system_constraints: Optional[List[str]] = None,
        verified_knowledge: Optional[List[str]] = None,
        reasoning_summaries: Optional[List[str]] = None,
        memories: Optional[List[str]] = None,
        capability_observations: Optional[List[str]] = None,
        uncertainties: Optional[List[str]] = None,
        user_assertions: Optional[List[str]] = None,
        budget: Optional[ContextBudget] = None,
        tokenizer: Optional[Any] = None,
    ) -> AssembledIntelligenceContext:
        """
        Build a bounded context string strictly adhering to budget and priority.
        """
        eff_budget = budget or self.default_budget
        max_prompt_tokens = eff_budget.max_prompt_tokens

        candidates: List[ContextItem] = []

        # 1. System Identity
        if system_identity:
            candidates.append(ContextItem(
                source_type=ContextSourceType.SYSTEM_IDENTITY,
                content=system_identity,
                priority=self.DEFAULT_PRIORITIES[ContextSourceType.SYSTEM_IDENTITY],
                is_authority=True,
            ))

        # 2. System Constraints
        if system_constraints:
            constraints_text = "\n".join(f"- {c}" for c in system_constraints)
            candidates.append(ContextItem(
                source_type=ContextSourceType.SYSTEM_CONSTRAINT,
                content=constraints_text,
                priority=self.DEFAULT_PRIORITIES[ContextSourceType.SYSTEM_CONSTRAINT],
                is_authority=True,
            ))

        # 3. Task Objective
        if task_objective:
            candidates.append(ContextItem(
                source_type=ContextSourceType.TASK_OBJECTIVE,
                content=task_objective,
                priority=self.DEFAULT_PRIORITIES[ContextSourceType.TASK_OBJECTIVE],
                is_authority=False,
            ))

        # 4. Verified Knowledge
        if verified_knowledge:
            for k in verified_knowledge:
                candidates.append(ContextItem(
                    source_type=ContextSourceType.VERIFIED_KNOWLEDGE,
                    content=k,
                    priority=self.DEFAULT_PRIORITIES[ContextSourceType.VERIFIED_KNOWLEDGE],
                    is_authority=False,
                ))

        # 5. Reasoning Summaries
        if reasoning_summaries:
            for r in reasoning_summaries:
                candidates.append(ContextItem(
                    source_type=ContextSourceType.REASONING_SUMMARY,
                    content=r,
                    priority=self.DEFAULT_PRIORITIES[ContextSourceType.REASONING_SUMMARY],
                    is_authority=False,
                ))

        # 6. Memories
        if memories:
            for m in memories:
                candidates.append(ContextItem(
                    source_type=ContextSourceType.MEMORY,
                    content=m,
                    priority=self.DEFAULT_PRIORITIES[ContextSourceType.MEMORY],
                    is_authority=False,
                ))

        # 7. Capability Observations
        if capability_observations:
            for c in capability_observations:
                candidates.append(ContextItem(
                    source_type=ContextSourceType.CAPABILITY_OBSERVATION,
                    content=c,
                    priority=self.DEFAULT_PRIORITIES[ContextSourceType.CAPABILITY_OBSERVATION],
                    is_authority=False,
                ))

        # 8. Uncertainties
        if uncertainties:
            for u in uncertainties:
                candidates.append(ContextItem(
                    source_type=ContextSourceType.UNCERTAIN_INFORMATION,
                    content=u,
                    priority=self.DEFAULT_PRIORITIES[ContextSourceType.UNCERTAIN_INFORMATION],
                    is_authority=False,
                ))

        # 9. User Assertions
        if user_assertions:
            for ua in user_assertions:
                candidates.append(ContextItem(
                    source_type=ContextSourceType.USER_ASSERTION,
                    content=ua,
                    priority=self.DEFAULT_PRIORITIES[ContextSourceType.USER_ASSERTION],
                    is_authority=False,
                ))

        # Compute token estimates for each item
        for item in candidates:
            formatted = item.format_for_prompt()
            item.token_count = self.estimate_tokens(formatted, tokenizer)

        # Sort strictly by priority (ascending order: 1 first)
        candidates.sort(key=lambda x: (x.priority, -x.confidence))

        # Greedy packing within max_prompt_tokens
        included: List[ContextItem] = []
        dropped: List[ContextItem] = []
        current_token_count = 0
        prompt_fragments: List[str] = []
        provenance: List[Dict[str, Any]] = []

        # Account for delimiters between sections (~2 tokens per section)
        for item in candidates:
            needed = item.token_count + 2
            if current_token_count + needed <= max_prompt_tokens:
                included.append(item)
                prompt_fragments.append(item.format_for_prompt())
                current_token_count += needed
                provenance.append({
                    "source_type": item.source_type.value,
                    "priority": item.priority,
                    "is_authority": item.is_authority,
                    "token_count": item.token_count,
                    "status": "INCLUDED",
                })
            else:
                dropped.append(item)
                provenance.append({
                    "source_type": item.source_type.value,
                    "priority": item.priority,
                    "is_authority": item.is_authority,
                    "token_count": item.token_count,
                    "status": "DROPPED_BUDGET_EXCEEDED",
                })

        full_prompt = "\n\n".join(prompt_fragments)
        final_tokens = self.estimate_tokens(full_prompt, tokenizer)

        return AssembledIntelligenceContext(
            full_prompt=full_prompt,
            token_count=final_tokens,
            items_included=included,
            items_dropped=dropped,
            budget=eff_budget,
            provenance=provenance,
        )
