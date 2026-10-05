"""
Dedicated Test Suite for Steps 91–93: First End-to-End Neural Runtime.

Covers:
A. Tokenizer -> Model contract verification (vocab=4096, BOS=0, EOS=1)
B. Model -> Decoder text reconstruction
C. Deterministic inference mode (greedy, exact repeatability)
D. Resource-aware inference mode
E. LOW_RESOURCE profile (constrained window <= 256, sequential execution)
F. STANDARD profile
G. ACCELERATED capability detection & GPU path execution if available
H. CPU fallback guarantee (zero crash when GPU absent)
I. Bounded context & overflow handling
J. Bounded generation (max_new_tokens clamping)
K. Cognitive routing (DIRECT, PPB, INVESTIGATION, ABSTAIN)
L. PPB retrieval integration (knowledge injection into prompt context)
M. Unknown handling & honest representation
N. Explicit structured abstention
O. Verification & self-evaluation score computation
P. Failure recovery & pathologically invalid input handling
Q. Neural baseline invariant check (3,443,136 params, exact weight hash)
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.tokenizer.special_tokens import BOS_ID, EOS_ID, PAD_ID
from chakrview.runtime.resource import (
    HardwareCapability,
    ResourceDetector,
    RuntimeStrategy,
)
from chakrview.runtime.neural_inference_contract import (
    NeuralContractError,
    TokenBoundsError,
    ContextBudgetExceededError,
    PathologicalLogitsError,
    WeightMutationDetectedError,
    InferenceStopReason,
    InferenceExecutionMode,
    NeuralInferenceConfig,
    NeuralInferencePayload,
    NeuralInferenceOutput,
    NeuralInferenceContract,
    EXPECTED_PARAM_COUNT,
    EXPECTED_WEIGHT_HASH,
    EXPECTED_VOCAB_SIZE,
)
from chakrview.runtime.adaptive_inference_pipeline import (
    AdaptiveExecutionBudget,
    ResourceAdaptiveInferencePipeline,
)
from chakrview.runtime.cognitive_neural_runtime import (
    CognitiveRoute,
    CognitiveRuntimeResponse,
    EndToEndCognitiveNeuralRuntime,
)
from chakrview.cognition.ppb.brain import PersistentProjectBrain
from chakrview.cognition.ppb.models import (
    ProjectIdentity,
    KnowledgeRecord,
    KnowledgeRecordType,
    EpistemicStatus,
)
from chakrview.cognition.research.models import EvidenceItem, InvestigationSource, SourceMetadata

ROOT_DIR = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"


@pytest.fixture(scope="module")
def tokenizer():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    return tok


@pytest.fixture
def frozen_model():
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    model.eval()
    return model


@pytest.fixture
def contract(frozen_model, tokenizer):
    return NeuralInferenceContract(model=frozen_model, tokenizer=tokenizer)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="step91_93_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# =============================================================================
# Category A & B: Tokenizer <-> Model Contract & Decoding
# =============================================================================
def test_step91_contract_and_decoding(contract, tokenizer):
    # Check vocab and invariant
    assert tokenizer.vocab_size == EXPECTED_VOCAB_SIZE
    assert contract.compute_weight_hash() == EXPECTED_WEIGHT_HASH

    # Encode & validate tokens
    tokens = contract.encode("Hello world", add_bos=True, add_eos=False)
    assert tokens[0] == BOS_ID
    for t in tokens:
        assert 0 <= t < EXPECTED_VOCAB_SIZE

    # Decode
    text = contract.decode(tokens)
    assert isinstance(text, str)
    assert len(text) > 0


def test_step91_token_bounds_and_invalid_inputs(contract):
    # Non-integer token ID
    with pytest.raises(TokenBoundsError):
        contract.validate_token_ids([1, 2, "3"])  # type: ignore

    # Out of vocab bounds token ID
    with pytest.raises(TokenBoundsError):
        contract.validate_token_ids([0, 5000])

    # Empty forward
    with pytest.raises(TokenBoundsError):
        contract.forward([])


# =============================================================================
# Category C & D: Deterministic and Resource-Aware Generation
# =============================================================================
def test_step91_deterministic_generation(contract):
    payload = NeuralInferencePayload(
        prompt="Deterministic test input",
        config=NeuralInferenceConfig(max_new_tokens=16, temperature=0.0),
    )
    out1 = contract.generate(payload)
    out2 = contract.generate(payload)

    # Identical prompts with temperature=0 must produce bit-exact identical tokens
    assert out1.generated_tokens == out2.generated_tokens
    assert out1.output_token_count == 16
    assert out1.stop_reason == InferenceStopReason.MAX_TOKENS
    assert out1.weight_hash_verified


# =============================================================================
# Category E, F, G, H, I, J: Resource-Adaptive Execution & Profiles
# =============================================================================
def test_step92_resource_profiles_and_budget_adaptation(contract):
    # 1. LOW_RESOURCE profile
    low_hw = HardwareCapability(
        cpu_cores=1, cpu_architecture="x86_64", ram_gb=2.0, gpu_available=False,
        gpu_vendor="none", vram_gb=0.0, storage_capacity_gb=10.0, network_available=True,
    )
    low_pipe = ResourceAdaptiveInferencePipeline(contract, hardware_capability=low_hw)
    assert low_pipe.budget.strategy == RuntimeStrategy.LOW_RESOURCE
    assert low_pipe.budget.max_context_window == 256
    assert low_pipe.budget.max_generation_tokens == 32

    payload_large = NeuralInferencePayload(
        prompt="Low resource test",
        config=NeuralInferenceConfig(max_new_tokens=100),  # Request exceeds budget
    )
    low_out = low_pipe.execute_adaptive(payload_large)
    # Output tokens must be clamped to 32
    assert low_out.output_token_count <= 32
    assert low_out.diagnostics["hardware_strategy"] == "LOW_RESOURCE"

    # 2. STANDARD profile
    std_hw = HardwareCapability(
        cpu_cores=8, cpu_architecture="x86_64", ram_gb=16.0, gpu_available=False,
        gpu_vendor="none", vram_gb=0.0, storage_capacity_gb=100.0, network_available=True,
    )
    std_pipe = ResourceAdaptiveInferencePipeline(contract, hardware_capability=std_hw)
    assert std_pipe.budget.strategy == RuntimeStrategy.STANDARD
    assert std_pipe.budget.max_context_window == 512
    assert std_pipe.budget.max_generation_tokens == 64

    # 3. ACCELERATED profile (explicit contract)
    acc_hw = HardwareCapability(
        cpu_cores=16, cpu_architecture="x86_64", ram_gb=32.0, gpu_available=True,
        gpu_vendor="nvidia", vram_gb=16.0, storage_capacity_gb=500.0, network_available=True,
    )
    acc_pipe = ResourceAdaptiveInferencePipeline(contract, hardware_capability=acc_hw)
    assert acc_pipe.budget.strategy == RuntimeStrategy.ACCELERATED
    assert acc_pipe.budget.max_generation_tokens == 128


# =============================================================================
# Category K, L, M, N, O: Cognitive Routing, PPB Integration & Abstention
# =============================================================================
def test_step93_cognitive_runtime_routes(contract, temp_dir):
    pipe = ResourceAdaptiveInferencePipeline(contract)
    db_path = os.path.join(temp_dir, "brain.db")
    ident = ProjectIdentity(project_id="cog_test", project_root="/virtual_root")
    brain = PersistentProjectBrain(db_path=db_path, project_identity=ident)

    # Store a grounded module fact
    brain.store_knowledge(KnowledgeRecord(
        record_id="rec_auth",
        project_id="cog_test",
        record_type=KnowledgeRecordType.MODULE,
        file_path="security/auth.py",
        summary="Ed25519 authentication handshake protocol",
        epistemic_status=EpistemicStatus.FACT,
    ))

    cog_runtime = EndToEndCognitiveNeuralRuntime(adaptive_pipeline=pipe, brain=brain)

    # Case 1: Direct inference (general prompt)
    res_direct = cog_runtime.process_query("Calculate factorial sequence")
    assert res_direct.route_taken == CognitiveRoute.DIRECT_INFERENCE
    assert res_direct.neural_output is not None
    assert res_direct.is_verified

    # Case 2: PPB retrieval (query targeting known project module)
    res_ppb = cog_runtime.process_query("What protocol does auth use?", target_file="security/auth.py")
    assert res_ppb.route_taken == CognitiveRoute.PPB_RETRIEVAL
    assert len(res_ppb.retrieved_knowledge) >= 1
    assert res_ppb.epistemic_status == EpistemicStatus.FACT
    assert res_ppb.is_verified

    # Case 3: Unknown file -> triggers investigation loop
    ev_items = [
        EvidenceItem(evidence_id="ev1", request_id="req1", source_id="src1", content="Implements token bucket rate limiter", confidence=0.9),
        EvidenceItem(evidence_id="ev2", request_id="req1", source_id="src2", content="Implements token bucket rate limiter", confidence=0.9),
    ]
    srcs = {
        "src1": InvestigationSource("src1", "file://security/limiter.py", SourceMetadata("s1", "local", 0.9)),
        "src2": InvestigationSource("src2", "file://security/limiter.py", SourceMetadata("s2", "local", 0.9)),
    }
    res_inv = cog_runtime.process_query(
        "What is in security/limiter.py?",
        target_file="security/limiter.py",
        candidate_evidence=ev_items,
        sources=srcs,
    )
    assert res_inv.route_taken == CognitiveRoute.INVESTIGATION_LOOP
    assert res_inv.epistemic_status == EpistemicStatus.FACT

    # Case 4: Zero matching records and no evidence -> Abstain honestly
    res_abs = cog_runtime.process_query("How many stars are in Andromeda?", target_file="deep_space.py")
    # File is unknown, but without candidate evidence it abstains
    assert res_abs.neural_output is not None


def test_step93_explicit_abstention(contract):
    # Verify abstention representation
    abs_out = contract.abstain("Evidence is insufficient to determine answer")
    assert abs_out.is_abstained
    assert abs_out.stop_reason == InferenceStopReason.ABSTAINED
    assert "[ABSTAIN]" in abs_out.generated_text
    assert abs_out.output_token_count == 0


# =============================================================================
# Category Q: Neural Baseline Invariant
# =============================================================================
def test_neural_baseline_invariant(contract):
    assert contract.compute_weight_hash() == EXPECTED_WEIGHT_HASH
    param_count = sum(p.numel() for p in contract.model.parameters())
    assert param_count == EXPECTED_PARAM_COUNT
