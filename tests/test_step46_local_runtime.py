"""
Step 46 Test Suite: Local Model Runtime, Interactive Streaming & Multi-Turn Context.

Covers:
1. Streaming token generation contract (StreamChunk, yielding, TTFT, stop_reason).
2. Streaming stop token masking under min_new_tokens.
3. Early generator abort and cleanup (KV cache and state invariant).
4. Multi-turn dialogue management (turns, pinned system prompt, history).
5. Sliding-window context compaction respecting the 512-token ceiling.
6. Cognitive memory integration into conversation sessions.
7. Tenant isolation across local model sessions.
8. Weight immutability (ΔW = 0) and canonical SHA-256 verification.
"""

from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.cognition.federation.cognitive.memory import PersistentCognitiveMemoryAdapter
from chakrview.cognition.federation.cognitive.models import (
    CognitiveContextEnvelope,
    SecretLeakageInContextError,
)
from chakrview.memory.models import MemoryVerificationState
from chakrview.runtime.inference import (
    GenerationConfig,
    StopReason,
)
from chakrview.runtime.local_runtime import (
    EXPECTED_PARAM_COUNT,
    EXPECTED_WEIGHT_HASH,
    LocalModelRuntime,
)
from chakrview.runtime.pipeline import (
    ContextOverflowError,
    InferenceEngine,
    InferenceRequest,
    NeuralWeightMutationError,
    StreamChunk,
)
from chakrview.runtime.sampling import SamplingConfig
from chakrview.runtime.session import (
    ConversationContextManager,
    ConversationRole,
    ConversationTurn,
    LocalModelSession,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

ROOT_DIR = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"


@pytest.fixture(scope="module")
def shared_tokenizer():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    return tok


@pytest.fixture
def clean_engine(shared_tokenizer):
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    model.eval()
    return InferenceEngine(model=model, tokenizer=shared_tokenizer)


@pytest.fixture
def clean_runtime():
    return LocalModelRuntime.from_default()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Streaming Inference Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamingInference:

    def test_stream_chunk_structure_and_yields(self, clean_engine: InferenceEngine):
        req = InferenceRequest(
            prompt="ChakrView is",
            generation_config=GenerationConfig(max_new_tokens=8, sampling=SamplingConfig(temperature=0.0)),
        )

        chunks = list(clean_engine.stream(req))
        assert len(chunks) == 8
        for i, c in enumerate(chunks):
            assert isinstance(c, StreamChunk)
            assert isinstance(c.token_id, int)
            assert isinstance(c.token_text, str)
            assert c.step_index == i
            assert c.latency_ms >= 0.0

        assert chunks[-1].is_final is True
        assert chunks[-1].stop_reason in (StopReason.MAX_TOKENS, StopReason.EOS)
        assert chunks[0].is_final is False

    def test_stream_matches_execute_tokens(self, clean_engine: InferenceEngine):
        gen_cfg = GenerationConfig(max_new_tokens=10, sampling=SamplingConfig(temperature=0.0))
        req = InferenceRequest(prompt="Verification of exact match", generation_config=gen_cfg)

        # Batch execution
        res_batch = clean_engine.execute(req)

        # Streaming execution
        chunks = list(clean_engine.stream(req))
        streamed_ids = [c.token_id for c in chunks]

        assert res_batch.token_ids == streamed_ids
        assert len(streamed_ids) == 10

    def test_stream_min_new_tokens_suppression(self, clean_engine: InferenceEngine):
        stop_tok = 42
        cfg = GenerationConfig(
            max_new_tokens=10,
            min_new_tokens=5,
            stop_token_ids=[stop_tok],
            sampling=SamplingConfig(temperature=0.0),
        )

        # Mock sampler to emit stop_tok at step 0
        call_count = 0
        orig_sample = clean_engine.sampler.sample

        def mock_sample(logits, generated_tokens, config, step, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return stop_tok
            return orig_sample(logits, generated_tokens, config, step, **kwargs)

        clean_engine.sampler.sample = mock_sample
        try:
            chunks = list(clean_engine.stream(InferenceRequest(prompt="Test min tokens", generation_config=cfg)))
            assert len(chunks) >= 5
        finally:
            clean_engine.sampler.sample = orig_sample

    def test_stream_early_break_generator_cleanup(self, clean_engine: InferenceEngine):
        req = InferenceRequest(
            prompt="Streaming break test",
            generation_config=GenerationConfig(max_new_tokens=16),
        )

        # Consume only 3 tokens then break early
        consumed = 0
        for chunk in clean_engine.stream(req):
            consumed += 1
            if consumed == 3:
                break

        assert consumed == 3
        # Assert KV cache was cleanly reset in finally block
        assert clean_engine.kv_cache.sequence_length == 0
        # Subsequent request should execute cleanly without error
        res = clean_engine.execute(InferenceRequest(prompt="Follow-up prompt", generation_config=GenerationConfig(max_new_tokens=4)))
        assert res.output_token_count == 4

    def test_stream_secret_leakage_fails_closed(self, clean_engine: InferenceEngine):
        req = InferenceRequest(prompt="My secret password is aws_secret_key=XYZ123")
        with pytest.raises(SecretLeakageInContextError):
            list(clean_engine.stream(req))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Turn Session & Context Management Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiTurnSessionAndContextManager:

    def test_conversation_turn_recording(self):
        sess = LocalModelSession(session_id="s1")
        t1 = sess.add_turn(ConversationRole.USER, "Hello", 2)
        t2 = sess.add_turn(ConversationRole.ASSISTANT, "Greetings", 2)

        assert len(sess.turns) == 2
        assert t1.role == ConversationRole.USER
        assert t2.role == ConversationRole.ASSISTANT
        assert t1.content == "Hello"

    def test_pinned_system_prompt_retention(self):
        sess = LocalModelSession(session_id="s1", system_prompt="You are a helpful assistant.")
        assert len(sess.turns) == 1
        assert sess.turns[0].role == ConversationRole.SYSTEM
        assert sess.turns[0].pinned is True

        sess.add_turn(ConversationRole.USER, "What is AI?", 5)
        sess.add_turn(ConversationRole.ASSISTANT, "Artificial Intelligence.", 4)
        assert len(sess.turns) == 3

        sess.clear()
        # System prompt preserved after clear()
        assert len(sess.turns) == 1
        assert sess.turns[0].role == ConversationRole.SYSTEM

    def test_sliding_window_compaction_under_budget(self):
        mgr = ConversationContextManager(max_context=512)

        sys_turn = ConversationTurn(ConversationRole.SYSTEM, "System instructions", token_count=10, pinned=True)
        t1 = ConversationTurn(ConversationRole.USER, "Question 1", token_count=20)
        t2 = ConversationTurn(ConversationRole.ASSISTANT, "Answer 1", token_count=20)
        t3 = ConversationTurn(ConversationRole.USER, "Question 2", token_count=20)
        t4 = ConversationTurn(ConversationRole.ASSISTANT, "Answer 2", token_count=20)

        turns = [sys_turn, t1, t2, t3, t4]

        # Budget of 60 tokens: sys (10) + t3 (20) + t4 (20) = 50 tokens (t1 and t2 evicted)
        compacted = mgr.compact_turns(turns, budget=60)
        assert len(compacted) == 3
        assert compacted[0] == sys_turn
        assert compacted[1] == t3
        assert compacted[2] == t4

    def test_context_overflow_without_truncation_raises(self, shared_tokenizer):
        sess = LocalModelSession(session_id="overflow_sess", max_context=64)
        # Create user input that consumes entire 64 token budget
        huge_input = "word " * 100
        with pytest.raises(ContextOverflowError):
            sess.build_prompt_package(
                user_input=huge_input,
                tokenizer=shared_tokenizer,
                max_new_tokens=32,
                truncate_if_overflow=False,
            )

    def test_session_tenant_isolation(self):
        sess_a = LocalModelSession(session_id="s1", tenant_id="tenant_alpha")
        sess_b = LocalModelSession(session_id="s1", tenant_id="tenant_beta")
        assert sess_a.tenant_id != sess_b.tenant_id


# ─────────────────────────────────────────────────────────────────────────────
# 3. Local Model Runtime Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalModelRuntimeIntegration:

    def test_runtime_from_default_invariants(self, clean_runtime: LocalModelRuntime):
        ident = clean_runtime.model_identity
        assert ident.parameter_count == EXPECTED_PARAM_COUNT == 3_443_136
        assert ident.vocab_size == 4096
        assert ident.max_context_len == 512
        assert ident.weight_hash == EXPECTED_WEIGHT_HASH

    def test_runtime_stateless_generate(self, clean_runtime: LocalModelRuntime):
        res = clean_runtime.generate(
            prompt="Stateless generation test",
            generation_config=GenerationConfig(max_new_tokens=8),
        )
        assert res.output_token_count == 8
        assert res.weight_hash_verified is True
        assert isinstance(res.text, str)

    def test_runtime_stateless_stream_generate(self, clean_runtime: LocalModelRuntime):
        chunks = list(clean_runtime.stream_generate(
            prompt="Stateless streaming test",
            generation_config=GenerationConfig(max_new_tokens=6),
        ))
        assert len(chunks) == 6
        assert chunks[-1].is_final is True

    def test_runtime_chat_multi_turn_history(self, clean_runtime: LocalModelRuntime):
        s_id = "test_chat_multi"
        clean_runtime.reset_session(s_id)

        # Turn 1
        res1 = clean_runtime.chat(session_id=s_id, prompt="Turn 1 prompt", generation_config=GenerationConfig(max_new_tokens=6))
        assert res1.output_token_count == 6

        # Turn 2
        res2 = clean_runtime.chat(session_id=s_id, prompt="Turn 2 prompt", generation_config=GenerationConfig(max_new_tokens=6))
        assert res2.output_token_count == 6

        sess = clean_runtime.get_or_create_session(s_id)
        assert len(sess.turns) == 4
        assert sess.turns[0].role == ConversationRole.USER
        assert sess.turns[1].role == ConversationRole.ASSISTANT
        assert sess.turns[2].role == ConversationRole.USER
        assert sess.turns[3].role == ConversationRole.ASSISTANT

    def test_runtime_stream_chat_multi_turn_history(self, clean_runtime: LocalModelRuntime):
        s_id = "test_stream_chat_multi"
        clean_runtime.reset_session(s_id)

        tokens = []
        for chunk in clean_runtime.stream_chat(
            session_id=s_id,
            prompt="Streaming multi turn prompt",
            generation_config=GenerationConfig(max_new_tokens=5),
        ):
            tokens.append(chunk)

        assert len(tokens) == 5
        sess = clean_runtime.get_or_create_session(s_id)
        assert len(sess.turns) == 2

    def test_runtime_with_cognitive_memory_retrieval(self, shared_tokenizer):
        from chakrview.memory.episodic import EpisodicMemoryStore
        from chakrview.memory.semantic import SemanticMemoryStore

        sem_store = SemanticMemoryStore()
        ep_store = EpisodicMemoryStore()
        adapter = PersistentCognitiveMemoryAdapter(
            semantic_store=sem_store,
            episodic_store=ep_store,
        )
        sem_store.add_memory(
            tenant_id="t_mem",
            subject="ChakrMicro",
            predicate="has_parameters",
            object_value="3443136",
            verification_status=MemoryVerificationState.VERIFIED,
            confidence=0.99,
        )

        torch.manual_seed(42)
        model = ChakrMicro(ModelConfig())
        model.eval()

        runtime = LocalModelRuntime(
            model=model,
            tokenizer=shared_tokenizer,
            memory_adapter=adapter,
        )

        res = runtime.chat(
            session_id="s_mem",
            prompt="How many parameters does ChakrMicro have?",
            tenant_id="t_mem",
            generation_config=GenerationConfig(max_new_tokens=8),
            retrieve_memory=True,
        )

        assert res.output_token_count == 8
        assert res.weight_hash_verified is True

    def test_runtime_weight_mutation_guard(self, clean_runtime: LocalModelRuntime):
        # Tamper with weight
        with torch.no_grad():
            clean_runtime.model.embedding.weight[0, 0] += 0.5

        with pytest.raises(NeuralWeightMutationError):
            clean_runtime.verify_runtime_integrity()

        with pytest.raises(NeuralWeightMutationError):
            clean_runtime.generate("Tampered execution test")

    def test_neural_core_immutability_step46(self, clean_runtime: LocalModelRuntime):
        # Restore canonical weights
        torch.manual_seed(42)
        clean_runtime.model = ChakrMicro(ModelConfig())
        clean_runtime.model.eval()
        clean_runtime.engine.model = clean_runtime.model

        # Run several generations
        for _ in range(5):
            clean_runtime.generate("Immutability verification pass", generation_config=GenerationConfig(max_new_tokens=4))

        final_hash = clean_runtime.engine.compute_weight_hash()
        assert final_hash == EXPECTED_WEIGHT_HASH
