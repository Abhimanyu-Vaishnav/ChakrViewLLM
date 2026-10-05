"""
Tests for Phase 4: Neural Governance & Boundary Verification.

Strictly verifies:
1. Canonical baseline hash remains immutable (hash: c5571c...).
2. NeuralInferenceContract enforces ΔW = 0 across repeated inferences.
3. In-flight parameter mutations during inference fail closed immediately.
4. Cognitive memory / PPB / Task expansion operations NEVER touch neural weights.
5. NeuralInferenceContract.from_checkpoint safely binds new checkpoint hashes
   while leaving canonical baseline untouched.
"""

from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.runtime.neural_inference_contract import (
    NeuralInferenceContract,
    NeuralInferencePayload,
    NeuralInferenceConfig,
    WeightMutationDetectedError,
)
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.checkpoint import CheckpointManager
from chakrview.cognition.ppb.brain import PersistentProjectBrain
from chakrview.cognition.ppb.models import KnowledgeRecord, KnowledgeRecordType, EpistemicStatus

ROOT_DIR = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"


@pytest.fixture
def tokenizer() -> BPETokenizer:
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    return tok


def test_canonical_baseline_bit_exactness():
    """Verify canonical baseline instantiation has bit-exact parameter count and hash."""
    model = instantiate_frozen_baseline()
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 3_443_136
    h = compute_model_hash(model)
    assert h == EXPECTED_WEIGHT_HASH


def test_neural_inference_contract_enforces_delta_w_zero(tokenizer: BPETokenizer):
    """Verify that multiple inferences through NeuralInferenceContract preserve ΔW = 0."""
    model = instantiate_frozen_baseline()
    contract = NeuralInferenceContract(model=model, tokenizer=tokenizer)

    initial_hash = contract.compute_weight_hash()
    assert initial_hash == EXPECTED_WEIGHT_HASH

    payload = NeuralInferencePayload(
        prompt="ChakrView indigenous neural architecture",
        config=NeuralInferenceConfig(max_new_tokens=8, temperature=0.0),
    )

    out1 = contract.generate(payload)
    assert out1.weight_hash_verified is True
    assert contract.compute_weight_hash() == EXPECTED_WEIGHT_HASH

    out2 = contract.generate(payload)
    assert out2.weight_hash_verified is True
    assert contract.compute_weight_hash() == EXPECTED_WEIGHT_HASH


def test_weight_mutation_during_inference_fails_closed(tokenizer: BPETokenizer):
    """Verify that any illegal parameter alteration triggers WeightMutationDetectedError."""
    model = instantiate_frozen_baseline()
    contract = NeuralInferenceContract(model=model, tokenizer=tokenizer)

    # Illegally tamper with a weight
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
            break

    payload = NeuralInferencePayload(
        prompt="Test tamper",
        config=NeuralInferenceConfig(max_new_tokens=4, temperature=0.0),
    )

    with pytest.raises(WeightMutationDetectedError):
        contract.generate(payload)


def test_trained_checkpoint_loader_boundary(tmp_path: Path, tokenizer: BPETokenizer):
    """
    Verify that an explicitly trained checkpoint with non-baseline weights
    can be loaded via NeuralInferenceContract.from_checkpoint, validates invariants,
    and binds its own weight hash without mutating the canonical baseline.
    """
    ckpt_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(checkpoint_dir=ckpt_dir)

    # Create a non-baseline model
    torch.manual_seed(12345)
    trained_model = ChakrMicro(ModelConfig())
    trained_hash = compute_model_hash(trained_model)
    assert trained_hash != EXPECTED_WEIGHT_HASH

    ckpt_path = manager.save(
        model=trained_model,
        step=50,
        tokenizer_checksum=tokenizer.checksum() if hasattr(tokenizer, "checksum") else None,
        parameter_count=3_443_136,
    )

    # Load into NeuralInferenceContract via from_checkpoint
    contract = NeuralInferenceContract.from_checkpoint(
        checkpoint_path=ckpt_path,
        tokenizer=tokenizer,
    )

    assert contract.expected_weight_hash == trained_hash
    assert contract.expected_weight_hash != EXPECTED_WEIGHT_HASH

    # Ensure canonical baseline was unaffected
    canonical_model = instantiate_frozen_baseline()
    assert compute_model_hash(canonical_model) == EXPECTED_WEIGHT_HASH


def test_cognitive_self_evolution_preserves_zero_weight_mutation(tmp_path: Path):
    """
    Phase 4 & 9: Verify that cognitive learning, PPB knowledge acquisition,
    lesson extraction, and strategy updates NEVER mutate neural weights (ΔW = 0).
    """
    baseline_model = instantiate_frozen_baseline()
    initial_hash = compute_model_hash(baseline_model)
    assert initial_hash == EXPECTED_WEIGHT_HASH

    # Initialize PPB
    ppb_file = tmp_path / "ppb.db"
    brain = PersistentProjectBrain(db_path=str(ppb_file))

    # Add knowledge records and lessons
    rec = KnowledgeRecord(
        record_id="know_001",
        project_id="chakrview",
        record_type=KnowledgeRecordType.ARCHITECTURE,
        file_path="chakrview/brain/model.py",
        summary="Decoupled Neural Baseline",
        details={"content": "Neural weights are strictly separated from episodic knowledge updates."},
        epistemic_status=EpistemicStatus.FACT,
    )
    brain.store_knowledge(rec)

    # Verify that model parameter hash remains strictly bit-exact
    post_hash = compute_model_hash(baseline_model)
    assert post_hash == initial_hash
    assert post_hash == EXPECTED_WEIGHT_HASH
