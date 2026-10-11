"""
Tests for Batch 31: Conversational Runtime, Multi-Turn SFT, Generation, and Session State.
"""

import os
import tempfile
import pytest
import torch

from chakrview.brain.model import ChakrMicro, ModelConfig
from chakrview.runtime.conversation import ConversationTemplate, ChatMessage, Role
from chakrview.runtime.local_runtime import LocalModelRuntime, DEFAULT_TOKENIZER_DIR
from chakrview.runtime.session import LocalModelSession, ConversationRole, ConversationContextManager, ConversationTurn
from chakrview.runtime.inference import GenerationConfig, StopReason
from chakrview.runtime.sampling import SamplingConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def test_conversation_serialization_and_parsing():
    msgs = [
        ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=Role.USER, content="Hello!"),
        ChatMessage(role=Role.ASSISTANT, content="Hi, how can I help?"),
    ]
    formatted = ConversationTemplate.format_conversation(msgs)
    assert "<|im_start|>system" in formatted
    assert "<|im_start|>user" in formatted
    assert "<|im_start|>assistant" in formatted

    parsed = ConversationTemplate.parse_conversation(formatted)
    assert len(parsed) == 3
    assert parsed[0].role == Role.SYSTEM
    assert parsed[0].content == "You are a helpful assistant."
    assert parsed[1].role == Role.USER
    assert parsed[1].content == "Hello!"
    assert parsed[2].role == Role.ASSISTANT
    assert parsed[2].content == "Hi, how can I help?"


def test_multiturn_sft_masking():
    tok, _ = load_tokenizer_artifacts(DEFAULT_TOKENIZER_DIR)
    msgs = [
        ChatMessage(role=Role.USER, content="What is 2+2?"),
        ChatMessage(role=Role.ASSISTANT, content="It is 4."),
    ]
    inputs, targets, active_tokens = ConversationTemplate.encode_multiturn_sft(msgs, tok, max_seq_len=512)
    assert len(inputs) == len(targets)
    assert active_tokens > 0
    # First token is BOS
    assert inputs[0] == 0
    # User tokens must be masked (-100) in targets
    assert targets[0] == -100
    assert -100 in targets
    # Assistant response tokens are not masked
    active_target_count = sum(1 for t in targets if t != -100)
    assert active_target_count == active_tokens


def test_session_state_and_isolation():
    sess1 = LocalModelSession(session_id="session_A", tenant_id="tenant_1")
    sess2 = LocalModelSession(session_id="session_B", tenant_id="tenant_2")

    sess1.add_turn(ConversationRole.USER, "Hello from A", 5)
    sess2.add_turn(ConversationRole.USER, "Hello from B", 5)

    assert len(sess1.turns) == 1
    assert sess1.turns[0].content == "Hello from A"
    assert len(sess2.turns) == 1
    assert sess2.turns[0].content == "Hello from B"


def test_session_persistence_roundtrip():
    sess = LocalModelSession(session_id="persist_test", tenant_id="tenant_alpha")
    sess.add_system_prompt("System instructions")
    sess.add_turn(ConversationRole.USER, "Question 1", 4)
    sess.add_turn(ConversationRole.ASSISTANT, "Answer 1", 4)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name

    try:
        sess.save_to_file(tmp_path)
        loaded = LocalModelSession.load_from_file(tmp_path)

        assert loaded.session_id == "persist_test"
        assert loaded.tenant_id == "tenant_alpha"
        assert len(loaded.turns) == 3
        assert loaded.turns[0].role == ConversationRole.SYSTEM
        assert loaded.turns[1].role == ConversationRole.USER
        assert loaded.turns[2].role == ConversationRole.ASSISTANT
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_context_budget_compaction():
    cm = ConversationContextManager(max_context=512)
    turns = [
        ConversationTurn(role=ConversationRole.SYSTEM, content="System prompt", token_count=10, pinned=True),
        ConversationTurn(role=ConversationRole.USER, content="Old question", token_count=50, pinned=False),
        ConversationTurn(role=ConversationRole.ASSISTANT, content="Old answer", token_count=50, pinned=False),
        ConversationTurn(role=ConversationRole.USER, content="Recent question", token_count=30, pinned=False),
    ]

    # With budget = 50: system (10) pinned + recent (30) = 40 <= 50. Old turns evicted.
    compacted = cm.compact_turns(turns, budget=50)
    assert len(compacted) == 2
    assert compacted[0].role == ConversationRole.SYSTEM
    assert compacted[1].content == "Recent question"


def test_runtime_checkpoint_loading_and_immutability():
    ckpt_path = "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip("Step 10,000 checkpoint not found")

    rt = LocalModelRuntime.from_checkpoint(ckpt_path)
    assert rt.checkpoint_step == 10000
    assert rt.verify_runtime_integrity()

    cfg = GenerationConfig(max_new_tokens=8, sampling=SamplingConfig(temperature=0.0))
    res = rt.generate("ChakrView architecture", generation_config=cfg)
    assert res.output_token_count > 0

    # Ensure weights did not mutate after generation
    assert rt.verify_runtime_integrity()
