"""
Unit tests for Step 11 Context Assembly & Prompt Budget Allocator.
"""

import pytest

from chakrview.runtime.context import (
    ContextBudget,
    PromptContextBuilder,
    AssembledContext,
)
from chakrview.runtime.knowledge import KnowledgeChunk


def test_context_budget_invariants():
    budget = ContextBudget(max_context=512, generation_budget=64)
    assert budget.max_prompt_tokens == 448

    # Invalid configurations
    with pytest.raises(ValueError, match="cannot exceed model limit"):
        ContextBudget(max_context=1024)

    with pytest.raises(ValueError, match="must be strictly less"):
        ContextBudget(max_context=512, generation_budget=512)


def test_prompt_context_builder_delimiters():
    builder = PromptContextBuilder()
    chunks = [
        (KnowledgeChunk("c1", "doc_A", 0, "ChakrMicro is a decoder-only model."), 0.95),
    ]
    ctx = builder.build_prompt(
        user_query="What is ChakrMicro?",
        system_prompt="You are ChakrView.",
        retrieved_chunks=chunks,
    )

    assert "You are ChakrView." in ctx.full_prompt
    assert "--- KNOWLEDGE CONTEXT START ---" in ctx.full_prompt
    assert "--- KNOWLEDGE CONTEXT END ---" in ctx.full_prompt
    assert "--- USER QUERY ---" in ctx.full_prompt
    assert "What is ChakrMicro?" in ctx.full_prompt
    assert ctx.knowledge_used is True
    assert len(ctx.provenance) == 1
    assert ctx.provenance[0].doc_id == "doc_A"


def test_prompt_context_builder_no_knowledge():
    builder = PromptContextBuilder()
    ctx = builder.build_prompt(
        user_query="Hello world",
        system_prompt="Assistant prompt",
        retrieved_chunks=[],
    )

    assert "Assistant prompt" in ctx.full_prompt
    assert "--- KNOWLEDGE CONTEXT START ---" not in ctx.full_prompt
    assert "--- USER QUERY ---" in ctx.full_prompt
    assert "Hello world" in ctx.full_prompt
    assert ctx.knowledge_used is False
    assert len(ctx.provenance) == 0


def test_prompt_context_builder_strict_512_budget():
    budget = ContextBudget(
        max_context=512,
        generation_budget=64,
        max_system_tokens=32,
        max_query_tokens=64,
        max_knowledge_tokens=150,
    )
    builder = PromptContextBuilder(default_budget=budget)

    # Provide many large chunks that would exceed budget
    large_chunks = [
        (KnowledgeChunk(f"c{i}", "doc_A", i, " ".join(["word"] * 50)), 1.0 / (i + 1))
        for i in range(10)
    ]

    ctx = builder.build_prompt(
        user_query="Query " + " ".join(["test"] * 40),
        system_prompt="System " + " ".join(["rule"] * 20),
        retrieved_chunks=large_chunks,
        budget=budget,
    )

    # Estimated prompt tokens + generation budget must be <= 512
    assert ctx.estimated_prompt_tokens + budget.generation_budget <= 512
    # Not all 10 chunks should fit
    assert len(ctx.provenance) < 10
    assert ctx.truncated is True


def test_context_builder_query_truncation():
    budget = ContextBudget(
        max_context=512,
        generation_budget=64,
        max_query_tokens=20,
    )
    builder = PromptContextBuilder(default_budget=budget)
    huge_query = " ".join(["query_token"] * 100)

    ctx = builder.build_prompt(user_query=huge_query, budget=budget)
    assert ctx.truncated is True
    assert len(ctx.user_query.split()) < 50
