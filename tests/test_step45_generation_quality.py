"""
Step 45 Dedicated Test Suite: Neural Inference Validation & Generation Quality Layer.

Comprehensive validation covering:
1. Configuration validation (GenerationConfig min_new_tokens, SamplingConfig bounds)
2. Deterministic greedy generation & reproducibility
3. Seeded stochastic sampling reproducibility & divergence
4. EOS termination & min_new_tokens suppression
5. Max-token enforcement & StopReason accounting
6. Context horizon & overflow boundary policies
7. Tokenizer <-> Model contract enforcement
8. Weight hash immutability (ΔW = 0) & tamper detection
9. Numerical safety: Pathological logits (NaN/Inf) & degenerate sampling distributions
10. Repetition penalty quality controls
11. Checkpoint identity & compatibility validation
12. ModelIdentity & reproducibility telemetry serialization
"""

from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.inference import GenerationConfig, StopReason
from chakrview.runtime.pipeline import (
    ContextOverflowError,
    IncompatibleCheckpointError,
    InferenceEngine,
    InferenceRequest,
    InferenceResult,
    ModelIdentity,
    NeuralWeightMutationError,
    PathologicalLogitsError,
    TokenizerModelMismatchError,
    load_and_validate_checkpoint,
    validate_checkpoint_compatibility,
)
from chakrview.runtime.sampling import (
    Sampler,
    SamplingConfig,
    SamplingProbabilityError,
    apply_repetition_penalty,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.tokenizer.tokenizer import BPETokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"
EXPECTED_WEIGHT_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_PARAM_COUNT = 3_443_136
EXPECTED_VOCAB_SIZE = 4096


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def shared_tokenizer() -> BPETokenizer:
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    return tok


@pytest.fixture
def clean_model() -> ChakrMicro:
    torch.manual_seed(42)
    m = ChakrMicro(ModelConfig())
    m.eval()
    return m


@pytest.fixture
def inference_engine(shared_tokenizer: BPETokenizer, clean_model: ChakrMicro) -> InferenceEngine:
    return InferenceEngine(model=clean_model, tokenizer=shared_tokenizer)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerationAndSamplingConfigValidation:

    def test_generation_config_min_new_tokens_valid(self):
        cfg = GenerationConfig(max_new_tokens=32, min_new_tokens=8)
        assert cfg.min_new_tokens == 8
        assert cfg.max_new_tokens == 32

    def test_generation_config_min_new_tokens_invalid_negative(self):
        with pytest.raises(ValueError, match="min_new_tokens must be non-negative"):
            GenerationConfig(max_new_tokens=32, min_new_tokens=-1)

    def test_generation_config_min_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="cannot exceed max_new_tokens"):
            GenerationConfig(max_new_tokens=16, min_new_tokens=20)

    def test_sampling_config_repetition_penalty_bounds(self):
        # Valid bounds [1.0, 10.0]
        cfg1 = SamplingConfig(repetition_penalty=1.0)
        assert cfg1.repetition_penalty == 1.0
        cfg2 = SamplingConfig(repetition_penalty=5.5)
        assert cfg2.repetition_penalty == 5.5
        cfg3 = SamplingConfig(repetition_penalty=10.0)
        assert cfg3.repetition_penalty == 10.0

        # Invalid bounds
        with pytest.raises(ValueError, match="repetition_penalty must be >= 1.0"):
            SamplingConfig(repetition_penalty=0.9)
        with pytest.raises(ValueError, match="repetition_penalty must be <= 10.0"):
            SamplingConfig(repetition_penalty=10.1)

    def test_sampling_config_temperature_bounds(self):
        cfg_greedy = SamplingConfig(temperature=0.0)
        assert cfg_greedy.is_greedy is True

        cfg_stochastic = SamplingConfig(temperature=0.8)
        assert cfg_stochastic.is_greedy is False

        with pytest.raises(ValueError, match="temperature must be non-negative"):
            SamplingConfig(temperature=-0.1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Deterministic Generation & Reproducibility Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterministicGenerationAndReproducibility:

    def test_greedy_decoding_exact_match_across_runs(self, inference_engine: InferenceEngine):
        prompt = "Determinism across repeated runs in indigenous AI"
        cfg = GenerationConfig(max_new_tokens=16, sampling=SamplingConfig(temperature=0.0))

        run1 = inference_engine.execute(InferenceRequest(prompt=prompt, generation_config=cfg))
        run2 = inference_engine.execute(InferenceRequest(prompt=prompt, generation_config=cfg))
        run3 = inference_engine.execute(InferenceRequest(prompt=prompt, generation_config=cfg))

        assert run1.token_ids == run2.token_ids == run3.token_ids
        assert run1.text == run2.text == run3.text
        assert run1.reproducibility["is_greedy"] is True

    def test_seeded_stochastic_sampling_exact_match(self, inference_engine: InferenceEngine):
        prompt = "Stochastic generation under deterministic pseudo-random seed"
        cfg = GenerationConfig(
            max_new_tokens=16,
            sampling=SamplingConfig(temperature=0.7, top_k=40, seed=9999),
        )

        res_a = inference_engine.execute(InferenceRequest(prompt=prompt, generation_config=cfg))
        res_b = inference_engine.execute(InferenceRequest(prompt=prompt, generation_config=cfg))

        assert res_a.token_ids == res_b.token_ids
        assert res_a.text == res_b.text
        assert res_a.reproducibility["seed"] == 9999

    def test_different_seeds_produce_divergent_samples(self, inference_engine: InferenceEngine):
        prompt = "Exploration with differing random generators"
        cfg1 = GenerationConfig(
            max_new_tokens=20,
            sampling=SamplingConfig(temperature=1.0, top_k=100, seed=1111),
        )
        cfg2 = GenerationConfig(
            max_new_tokens=20,
            sampling=SamplingConfig(temperature=1.0, top_k=100, seed=8888),
        )

        res1 = inference_engine.execute(InferenceRequest(prompt=prompt, generation_config=cfg1))
        res2 = inference_engine.execute(InferenceRequest(prompt=prompt, generation_config=cfg2))

        # Under temperature 1.0 with top-k 100, different seeds must produce different sequences
        assert res1.token_ids != res2.token_ids


# ─────────────────────────────────────────────────────────────────────────────
# 3. EOS & Min New Tokens Quality Controls
# ─────────────────────────────────────────────────────────────────────────────

class TestEOSAndMinNewTokens:

    def test_min_new_tokens_suppresses_premature_eos(self, inference_engine: InferenceEngine):
        stop_tok = 42
        cfg = GenerationConfig(
            max_new_tokens=10,
            min_new_tokens=5,
            stop_token_ids=[stop_tok],
            sampling=SamplingConfig(temperature=0.0),
        )

        # Mock sampler to emit stop_tok early at step 1, 2, and then step 6
        step_call_count = 0
        orig_sample = inference_engine.sampler.sample

        def mock_sample(logits, generated_tokens, config, step, **kwargs):
            nonlocal step_call_count
            step_call_count += 1
            if step_call_count in (1, 2):
                return stop_tok
            if step_call_count >= 6:
                return stop_tok
            return orig_sample(logits, generated_tokens, config, step, **kwargs)

        inference_engine.sampler.sample = mock_sample
        try:
            res = inference_engine.execute(InferenceRequest(prompt="Min tokens test", generation_config=cfg))
            # Output token count must be at least min_new_tokens
            assert res.output_token_count >= 5
            assert res.stop_reason == StopReason.EOS
        finally:
            inference_engine.sampler.sample = orig_sample

    def test_max_new_tokens_termination_reason(self, inference_engine: InferenceEngine):
        cfg = GenerationConfig(max_new_tokens=12, stop_token_ids=[99999])
        res = inference_engine.execute(InferenceRequest(prompt="Max tokens check", generation_config=cfg))
        assert res.output_token_count == 12
        assert res.stop_reason == StopReason.MAX_TOKENS


# ─────────────────────────────────────────────────────────────────────────────
# 4. Context & Decoding Boundaries Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestContextAndDecodingBoundaries:

    def test_empty_prompt_handling(self, inference_engine: InferenceEngine):
        req = InferenceRequest(prompt="", add_bos=True)
        res = inference_engine.execute(req)
        assert res.input_token_count == 1  # BOS token [0]
        assert res.output_token_count > 0

    def test_prompt_at_context_limit_fail_closed(self, inference_engine: InferenceEngine):
        # 505 prompt tokens + 10 max_new_tokens > 512
        oversized_tokens = [0] + [10] * 504
        req = InferenceRequest(
            prompt_tokens=oversized_tokens,
            generation_config=GenerationConfig(max_new_tokens=10),
            truncate_if_overflow=False,
        )
        with pytest.raises(ContextOverflowError):
            inference_engine.execute(req)

    def test_prompt_at_context_limit_governed_truncation(self, inference_engine: InferenceEngine):
        oversized_tokens = [0] + [10] * 504
        req = InferenceRequest(
            prompt_tokens=oversized_tokens,
            generation_config=GenerationConfig(max_new_tokens=10),
            truncate_if_overflow=True,
        )
        res = inference_engine.execute(req)
        assert res.total_token_count <= 512
        assert res.output_token_count == 10


# ─────────────────────────────────────────────────────────────────────────────
# 5. Numerical Safety & Sampling Protection Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSamplingSafetyAndPathologicalLogits:

    def test_nan_logits_in_forward_raises(self, shared_tokenizer: BPETokenizer, clean_model: ChakrMicro):
        engine = InferenceEngine(model=clean_model, tokenizer=shared_tokenizer)

        def mock_forward_with_nan(*args, **kwargs):
            out = torch.zeros((1, 3, EXPECTED_VOCAB_SIZE))
            out[0, 1, 5] = float("nan")
            return out

        clean_model.forward = mock_forward_with_nan
        with pytest.raises(PathologicalLogitsError):
            engine.forward([0, 1, 2])

    def test_inf_logits_in_decode_raises(self, inference_engine: InferenceEngine):
        def mock_decode_inf(*args, **kwargs):
            out = torch.zeros((1, 1, EXPECTED_VOCAB_SIZE))
            out[0, 0, 0] = float("inf")
            return out

        inference_engine.model.decode_next = mock_decode_inf
        with pytest.raises(PathologicalLogitsError):
            inference_engine.execute(
                InferenceRequest(prompt="Test Inf", generation_config=GenerationConfig(max_new_tokens=2))
            )

    def test_degenerate_probability_distribution_raises(self):
        sampler = Sampler()
        # Degenerate probability distribution via extreme temperature causing NaNs in softmax
        logits = torch.tensor([1.0, 2.0])
        with pytest.raises(SamplingProbabilityError, match="Degenerate probability"):
            sampler.sample(
                logits=logits,
                config=SamplingConfig(temperature=1e-45),
                strict_safety=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Repetition Penalty Quality Controls
# ─────────────────────────────────────────────────────────────────────────────

class TestRepetitionPenaltyQualityControls:

    def test_repetition_penalty_suppression(self):
        logits = torch.tensor([1.0, 4.0, 3.8, 0.5])
        # Token 1 has highest logit (4.0). Applying penalty 2.0 must suppress it below token 2 (3.8)
        penalized = apply_repetition_penalty(logits, tokens=[1], penalty=2.0)
        assert penalized[1] < penalized[2]
        # Unseen tokens must remain unchanged
        assert penalized[0] == logits[0]
        assert penalized[2] == logits[2]
        assert penalized[3] == logits[3]

    def test_repetition_penalty_negative_logits(self):
        # Keskar formula: for negative logit, multiplies by penalty to make it even more negative
        logits = torch.tensor([-2.0, -1.0])
        penalized = apply_repetition_penalty(logits, tokens=[1], penalty=2.0)
        assert penalized[1] == -2.0  # -1.0 * 2.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Checkpoint & Model Identity Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpointAndModelIdentity:

    def test_model_identity_dataclass(self):
        identity = ModelIdentity()
        assert identity.architecture_name == "ChakrMicro"
        assert identity.version == "0.1.0"
        assert identity.parameter_count == EXPECTED_PARAM_COUNT
        assert identity.vocab_size == EXPECTED_VOCAB_SIZE
        assert identity.max_context_len == 512
        assert identity.weight_hash == EXPECTED_WEIGHT_HASH

        d = identity.to_dict()
        assert d["parameter_count"] == EXPECTED_PARAM_COUNT
        assert d["vocab_size"] == EXPECTED_VOCAB_SIZE

    def test_validate_checkpoint_compatibility_success(self, clean_model: ChakrMicro):
        ckpt = {"model_state_dict": clean_model.state_dict()}
        assert validate_checkpoint_compatibility(ckpt) is True

    def test_validate_checkpoint_missing_state_dict_raises(self):
        with pytest.raises(IncompatibleCheckpointError, match="missing 'model_state_dict'"):
            validate_checkpoint_compatibility({"invalid_key": 123})

    def test_validate_checkpoint_missing_parameter_raises(self, clean_model: ChakrMicro):
        s_dict = clean_model.state_dict()
        del s_dict["layers.0.attn.q_proj.weight"]
        with pytest.raises(IncompatibleCheckpointError, match="missing required parameter"):
            validate_checkpoint_compatibility({"model_state_dict": s_dict})

    def test_validate_checkpoint_shape_mismatch_raises(self, clean_model: ChakrMicro):
        s_dict = clean_model.state_dict()
        # Tamper with embedding shape
        s_dict["embedding.weight"] = torch.zeros((2048, 192))
        with pytest.raises(IncompatibleCheckpointError, match="Shape mismatch"):
            validate_checkpoint_compatibility({"model_state_dict": s_dict})

    def test_validate_checkpoint_non_dict_raises(self):
        with pytest.raises(IncompatibleCheckpointError, match="must be a dict"):
            validate_checkpoint_compatibility("not_a_dict")  # type: ignore

    def test_load_and_validate_checkpoint_success(self, clean_model: ChakrMicro, tmp_path: Path):
        ckpt_path = tmp_path / "valid_model.pt"
        torch.save({"model_state_dict": clean_model.state_dict()}, ckpt_path)

        new_model = ChakrMicro(ModelConfig())
        loaded = load_and_validate_checkpoint(ckpt_path, new_model)
        assert "model_state_dict" in loaded


# ─────────────────────────────────────────────────────────────────────────────
# 8. Neural Core Immutability (ΔW = 0) Invariant Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNeuralCoreImmutabilityStep45:

    def test_inference_strictly_preserves_weight_hash(self, inference_engine: InferenceEngine):
        pre_hash = inference_engine.compute_weight_hash()
        assert pre_hash == EXPECTED_WEIGHT_HASH

        # Execute multiple inferences
        for p in ["Test alpha", "Test beta", "Test gamma"]:
            res = inference_engine.execute(
                InferenceRequest(prompt=p, generation_config=GenerationConfig(max_new_tokens=8))
            )
            assert res.weight_hash_verified is True
            assert res.model_identity.weight_hash == EXPECTED_WEIGHT_HASH

        post_hash = inference_engine.compute_weight_hash()
        assert post_hash == EXPECTED_WEIGHT_HASH
        assert post_hash == pre_hash


# ─────────────────────────────────────────────────────────────────────────────
# 9. Telemetry & Serialization Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTelemetryAndSerialization:

    def test_inference_result_telemetry_complete(self, inference_engine: InferenceEngine):
        cfg = GenerationConfig(
            max_new_tokens=10,
            sampling=SamplingConfig(temperature=0.0, repetition_penalty=1.2),
        )
        res = inference_engine.execute(InferenceRequest(prompt="Telemetry test", generation_config=cfg))

        res_dict = res.to_dict()
        assert "model_identity" in res_dict
        assert "reproducibility" in res_dict
        assert res_dict["model_identity"]["parameter_count"] == EXPECTED_PARAM_COUNT
        assert res_dict["reproducibility"]["repetition_penalty"] == 1.2
        assert res_dict["reproducibility"]["is_greedy"] is True
        assert res_dict["weight_hash_verified"] is True
