"""
Unit tests for Step 12 Conversational State & Working Memory Subsystem.

Tests data structures, deterministic memory ranking, tie-breaking, budget eviction,
session isolation, privacy guarantees, rule-based extraction, and summarization.
"""

import os
import pytest
from chakrview.runtime.memory import (
    MemoryType,
    ConversationTurn,
    MemoryItem,
    WorkingMemory,
    ConversationState,
    ConversationStore,
    MemoryExtractor,
    ConversationSummarizer,
)


def test_session_creation_and_retrieval():
    store = ConversationStore()
    assert store.list_sessions() == []

    session = store.create_session("sess_alpha")
    assert session.session_id == "sess_alpha"
    assert session.sequence_counter == 0
    assert len(session.turns) == 0
    assert len(session.working_memory) == 0

    retrieved = store.get_session("sess_alpha")
    assert retrieved is not None
    assert retrieved.session_id == "sess_alpha"

    # Cannot recreate existing session
    with pytest.raises(ValueError, match="already exists"):
        store.create_session("sess_alpha")


def test_session_isolation():
    store = ConversationStore()
    sess_a = store.create_session("session_A")
    sess_b = store.create_session("session_B")

    store.append_turn("session_A", role="user", text="Message in A")
    store.append_turn("session_B", role="user", text="Message in B")

    sess_a.working_memory.add(
        MemoryItem("m_a", "Secret fact of A", MemoryType.FACT, importance=0.9)
    )
    sess_b.working_memory.add(
        MemoryItem("m_b", "Secret fact of B", MemoryType.FACT, importance=0.8)
    )

    # Verify session A turns and memories
    turns_a = store.get_recent_turns("session_A")
    assert len(turns_a) == 1
    assert turns_a[0].text == "Message in A"
    assert sess_a.working_memory.get("m_a") is not None
    assert sess_a.working_memory.get("m_b") is None

    # Verify session B turns and memories
    turns_b = store.get_recent_turns("session_B")
    assert len(turns_b) == 1
    assert turns_b[0].text == "Message in B"
    assert sess_b.working_memory.get("m_b") is not None
    assert sess_b.working_memory.get("m_a") is None


def test_append_turns_chronological_ordering():
    store = ConversationStore()
    store.create_session("sess_seq")

    t0 = store.append_turn("sess_seq", role="system", text="System initialized.")
    t1 = store.append_turn("sess_seq", role="user", text="Hello assistant")
    t2 = store.append_turn("sess_seq", role="assistant", text="Hello user")
    t3 = store.append_turn("sess_seq", role="tool", text="Tool output 42")

    turns = store.get_recent_turns("sess_seq", n=10)
    assert len(turns) == 4
    assert [t.sequence_number for t in turns] == [0, 1, 2, 3]
    assert [t.role for t in turns] == ["system", "user", "assistant", "tool"]
    assert turns[0].turn_id == t0.turn_id
    assert turns[1].turn_id == t1.turn_id
    assert turns[2].turn_id == t2.turn_id
    assert turns[3].turn_id == t3.turn_id

    # Test slice recent
    recent_2 = store.get_recent_turns("sess_seq", n=2)
    assert len(recent_2) == 2
    assert [t.sequence_number for t in recent_2] == [2, 3]


def test_working_memory_insertion_and_ranking():
    wm = WorkingMemory(max_token_budget=500)
    item1 = MemoryItem(
        memory_id="mem_low",
        content="The sky is blue today.",
        memory_type=MemoryType.FACT,
        importance=0.2,
        created_sequence=1,
    )
    item2 = MemoryItem(
        memory_id="mem_high",
        content="User requires concise responses under 50 words.",
        memory_type=MemoryType.CONSTRAINT,
        importance=0.95,
        created_sequence=2,
    )
    wm.add(item1)
    wm.add(item2)

    assert len(wm) == 2

    # Query matching item2
    ranked = wm.retrieve_relevant(query="Please give me concise response", top_k=2)
    assert len(ranked) == 2
    assert ranked[0][0].memory_id == "mem_high"
    assert ranked[0][1] > ranked[1][1]


def test_memory_deterministic_tie_breaking():
    wm = WorkingMemory(max_token_budget=500)
    # Two items with identical importance, type, recency, and no query overlap
    m_beta = MemoryItem(
        memory_id="beta_item",
        content="Information fragment beta",
        memory_type=MemoryType.FACT,
        importance=0.5,
        created_sequence=1,
        last_access_sequence=1,
    )
    m_alpha = MemoryItem(
        memory_id="alpha_item",
        content="Information fragment alpha",
        memory_type=MemoryType.FACT,
        importance=0.5,
        created_sequence=1,
        last_access_sequence=1,
    )
    wm.add(m_beta)
    wm.add(m_alpha)

    # Retrieval should be strictly deterministic: tie-break alphabetical on memory_id
    ranked_1 = wm.retrieve_relevant(query="Unrelated query string", top_k=2)
    ranked_2 = wm.retrieve_relevant(query="Unrelated query string", top_k=2)

    assert [item.memory_id for item, _ in ranked_1] == ["alpha_item", "beta_item"]
    assert [item.memory_id for item, _ in ranked_2] == ["alpha_item", "beta_item"]


def test_working_memory_token_budget_eviction():
    # Budget capacity: 30 tokens
    wm = WorkingMemory(max_token_budget=30)

    # item1: estimate ~15 tokens, low importance
    item1 = MemoryItem(
        memory_id="m1_low",
        content="This is an older low importance casual remark.",
        memory_type=MemoryType.FACT,
        importance=0.1,
        token_estimate=15,
        created_sequence=0,
    )
    # item2: estimate ~15 tokens, medium importance
    item2 = MemoryItem(
        memory_id="m2_med",
        content="The user operates on Python 3.14 on Windows platform.",
        memory_type=MemoryType.FACT,
        importance=0.6,
        token_estimate=15,
        created_sequence=1,
    )
    # item3: estimate ~15 tokens, critical constraint
    item3 = MemoryItem(
        memory_id="m3_crit",
        content="CRITICAL: Never output markdown formatting or bullet points.",
        memory_type=MemoryType.CONSTRAINT,
        importance=0.99,
        token_estimate=15,
        created_sequence=2,
    )

    wm.add(item1)
    wm.add(item2)
    assert len(wm) == 2
    assert wm.total_tokens == 30

    # Adding item3 exceeds 30 tokens -> m1_low must be evicted
    wm.add(item3)
    assert len(wm) == 2
    assert wm.get("m1_low") is None  # Evicted
    assert wm.get("m2_med") is not None
    assert wm.get("m3_crit") is not None
    assert wm.total_tokens <= 30


def test_conversation_store_clear_and_delete():
    store = ConversationStore()
    store.create_session("sess_temp")
    store.append_turn("sess_temp", role="user", text="hello")
    store.append_turn("sess_temp", role="assistant", text="hi")
    state = store.get_session("sess_temp")
    state.working_memory.add(MemoryItem("m_t", "content", MemoryType.FACT))

    assert len(state.turns) == 2
    assert len(state.working_memory) == 1

    # Clear session preserves session ID but wipes turns and working memory
    cleared = store.clear_session("sess_temp")
    assert cleared is True
    assert len(state.turns) == 0
    assert len(state.working_memory) == 0
    assert state.sequence_counter == 0

    # Delete session removes session from registry
    deleted = store.delete_session("sess_temp")
    assert deleted is True
    assert store.get_session("sess_temp") is None
    assert store.list_sessions() == []


def test_conversation_snapshot():
    store = ConversationStore()
    store.create_session("sess_snap")
    store.append_turn("sess_snap", role="user", text="query 1")
    state = store.get_session("sess_snap")
    state.working_memory.add(MemoryItem("m1", "fact 1", MemoryType.FACT))
    state.active_domain = "coding"

    snap = store.snapshot("sess_snap")
    assert snap["session_id"] == "sess_snap"
    assert snap["active_domain"] == "coding"
    assert len(snap["turns"]) == 1
    assert len(snap["working_memory"]) == 1
    assert snap["turns"][0]["text"] == "query 1"


def test_privacy_in_memory_no_disk_persistence(tmp_path):
    """Verify that ConversationStore operations do NOT create disk files."""
    initial_cwd_files = set(os.listdir("."))
    
    store = ConversationStore()
    sess = store.create_session("sess_privacy")
    for i in range(10):
        store.append_turn("sess_privacy", role="user", text=f"Confidential data {i}")
        store.append_turn("sess_privacy", role="assistant", text=f"Response {i}")
        sess.working_memory.add(
            MemoryItem(f"mem_{i}", f"Confidential fact {i}", MemoryType.FACT)
        )

    store.snapshot("sess_privacy")
    store.clear_session("sess_privacy")
    store.delete_session("sess_privacy")

    final_cwd_files = set(os.listdir("."))
    # Zero new files should have been created in the workspace root
    assert initial_cwd_files == final_cwd_files


def test_memory_extractor_deterministic():
    text_constraint = "You must always respond in JSON format and do not use greetings."
    items_c = MemoryExtractor.extract_from_text(text_constraint, source_turn_id="t1", sequence_num=1)
    assert len(items_c) >= 1
    assert any(it.memory_type == MemoryType.CONSTRAINT for it in items_c)

    text_pref = "I prefer concise bullet points and I like short answers."
    items_p = MemoryExtractor.extract_from_text(text_pref, source_turn_id="t2", sequence_num=2)
    assert len(items_p) >= 1
    assert any(it.memory_type == MemoryType.PREFERENCE for it in items_p)

    text_task = "Our task is to refactor the database layer."
    items_t = MemoryExtractor.extract_from_text(text_task, source_turn_id="t3", sequence_num=3)
    assert len(items_t) >= 1
    assert any(it.memory_type == MemoryType.TASK_STATE for it in items_t)


def test_conversation_summarizer_deterministic():
    store = ConversationStore()
    session = store.create_session("sess_summarize")

    for i in range(10):
        store.append_turn("sess_summarize", role="user", text=f"Question step {i}")
        store.append_turn("sess_summarize", role="assistant", text=f"Answer step {i}")

    assert len(session.turns) == 20

    # Summarize with threshold 8, keep 4
    summary_item = ConversationSummarizer.maybe_summarize(
        state=session,
        max_turns_threshold=8,
        keep_recent=4,
    )

    assert summary_item is not None
    assert summary_item.memory_type == MemoryType.SUMMARY
    assert "Compressed dialogue" in summary_item.content
    assert session.working_memory.get(summary_item.memory_id) is not None
