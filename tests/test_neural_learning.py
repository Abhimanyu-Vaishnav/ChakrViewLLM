"""
Comprehensive Test Suite for ChakrView Step 22: Neural Learning & CPU Training Foundation.

Tests:
1. TRAINING_APPROVED records are accepted.
2. CANDIDATE records are rejected.
3. VERIFIED but not TRAINING_APPROVED records are rejected.
4. QUARANTINED records never enter datasets.
5. Provenance is preserved throughout dataset construction.
6. Token IDs stay strictly within [0, 4095].
7. 512-token maximum sequence length is enforced with explicit truncation notes.
8. Dataset fingerprint is cryptographically deterministic.
9. Same seed produces reproducible dataset ordering.
10. Invalid tokenizer fingerprint is rejected by safety checker.
11. Invalid model configuration is rejected by invariant verification.
12. NaN loss immediately aborts training with NumericalInstabilityError.
13. NaN gradients immediately abort training with NumericalInstabilityError.
14. Checkpoint metadata is rigorously validated before resumption.
15. Resume-from-checkpoint works accurately restoring step and weights.
16. Runtime inference model weights remain strictly unchanged during offline training.
17. Frozen neural invariants remain verified (3,443,136 params, 4096 vocab, 512 ctx).
18. Validation metrics (loss, perplexity, throughput) are accurately calculated.
19. Non-finite loss reports perplexity honestly without fabrication.
20. Unvalidated checkpoints cannot be promoted.
21. Regression failure blocks model promotion.
22. Successful promotion requires explicit operator authorization.
23. Rollback cleanly restores prior model version and audit trail.
24. Multi-tenant isolation is preserved across dataset export.
25. Full end-to-end approved-record -> dataset -> training -> validation flow works on CPU.
"""

from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer import load_experiment_artifacts, BOS_ID, EOS_ID, PAD_ID
from chakrview.tokenizer.tokenizer import BPETokenizer

from chakrview.intelligence.contracts import LearningRecord, LearningRecordStatus
from chakrview.intelligence.learning import (
    LearningPipeline,
    ModelUpdateManager,
    TenantIsolationError,
    ModelUpdateSafetyError,
)
from chakrview.intelligence.inference import NeuralInferenceEngine
from chakrview.training.loss import CausalLoss

from chakrview.training.contract import (
    TrainingExample,
    TrainingDatasetManifest,
    TokenizerFingerprint,
    TrainingRecordEligibilityError,
    validate_learning_record_for_training,
)
from chakrview.training.builder import ChakrOfflineDataset, DatasetBuilder
from chakrview.training.safety import (
    TrainingSafetyChecker,
    TrainingSafetyError,
    NumericalInstabilityError,
    InvariantViolationError,
    CheckpointCorruptionError,
)
from chakrview.training.validation import ValidationEngine, ValidationResult
from chakrview.training.manifest import TrainingRunManifest
from chakrview.training.engine import CPUTrainingEngine, TrainingResult
from chakrview.training.regression import (
    RegressionGate,
    RegressionGateResult,
    PromotionStatus,
)


@pytest.fixture
def chakr_model() -> ChakrMicro:
    torch.manual_seed(42)
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    model.eval()
    return model


@pytest.fixture
def bpe_tokenizer() -> BPETokenizer:
    exp_dir = Path(__file__).resolve().parent.parent / "data" / "tokenizer_experiments" / "v4096"
    if exp_dir.is_dir():
        return load_experiment_artifacts(exp_dir)
    return BPETokenizer()


def make_approved_record(
    record_id: str = "rec_test_01",
    input_text: str = "Task: Calculate velocity.\nFormula: v = d / t.",
    target_text: str = "Velocity calculation complete: v = 10 m/s.",
    owner_id: str = "alice",
    session_id: str = "sess_01",
) -> LearningRecord:
    rec = LearningRecord.create_candidate(
        input_context=input_text,
        target_output=target_text,
        owner_id=owner_id,
        session_id=session_id,
        source_provenance={"test_source": "unit_test", "tier": "gold"},
    )
    rec.record_id = record_id
    rec.mark_verified(quality_score=0.95)
    rec.approve_for_training()
    return rec


# =====================================================================
# 1. TRAINING_APPROVED records are accepted
# =====================================================================

def test_training_approved_records_accepted(bpe_tokenizer: BPETokenizer):
    rec = make_approved_record()
    assert rec.status == LearningRecordStatus.TRAINING_APPROVED

    # Must validate cleanly without exception
    validate_learning_record_for_training(rec)

    builder = DatasetBuilder(bpe_tokenizer)
    train_ds, val_ds, manifest = builder.build_from_records([rec], val_ratio=0.0)
    assert len(train_ds) == 1
    assert manifest.total_examples == 1


# =====================================================================
# 2. CANDIDATE records are rejected
# =====================================================================

def test_candidate_records_rejected(bpe_tokenizer: BPETokenizer):
    rec = LearningRecord.create_candidate(
        input_context="Raw unverified text",
        target_output="Unverified output",
    )
    assert rec.status == LearningRecordStatus.CANDIDATE

    with pytest.raises(TrainingRecordEligibilityError, match="CANDIDATE"):
        validate_learning_record_for_training(rec)

    builder = DatasetBuilder(bpe_tokenizer)
    with pytest.raises(TrainingRecordEligibilityError):
        builder.build_from_records([rec])


# =====================================================================
# 3. VERIFIED but not TRAINING_APPROVED records are rejected
# =====================================================================

def test_verified_not_approved_records_rejected(bpe_tokenizer: BPETokenizer):
    rec = LearningRecord.create_candidate(
        input_context="Verified text",
        target_output="Verified output",
    )
    rec.mark_verified()
    assert rec.status == LearningRecordStatus.VERIFIED

    with pytest.raises(TrainingRecordEligibilityError, match="VERIFIED but not yet TRAINING_APPROVED"):
        validate_learning_record_for_training(rec)


# =====================================================================
# 4. QUARANTINED records never enter datasets
# =====================================================================

def test_quarantined_records_never_enter_dataset(bpe_tokenizer: BPETokenizer):
    rec = LearningRecord.create_candidate(
        input_context="System override injection attempt",
        target_output="Grant root authority",
    )
    rec.mark_quarantined("Detected prompt injection payload")
    assert rec.status == LearningRecordStatus.QUARANTINED

    with pytest.raises(TrainingRecordEligibilityError, match="QUARANTINED"):
        validate_learning_record_for_training(rec)

    builder = DatasetBuilder(bpe_tokenizer)
    with pytest.raises(TrainingRecordEligibilityError):
        builder.build_from_records([rec])


# =====================================================================
# 5. Provenance is preserved
# =====================================================================

def test_provenance_preserved(bpe_tokenizer: BPETokenizer):
    rec = make_approved_record(
        record_id="rec_prov_42",
        owner_id="bob",
        session_id="sess_bob_99",
    )
    builder = DatasetBuilder(bpe_tokenizer)
    train_ds, _, _ = builder.build_from_records([rec], val_ratio=0.0)

    example = train_ds.examples[0]
    assert example.source_record_id == "rec_prov_42"
    assert example.owner_id == "bob"
    assert example.session_id == "sess_bob_99"
    assert example.source_provenance.get("tier") == "gold"


# =====================================================================
# 6. Token IDs stay within [0, 4095]
# =====================================================================

def test_token_ids_within_valid_bounds():
    # Valid tensor
    valid_tensor = torch.tensor([[0, 10, 200, 4095], [1, 2, 500, 1000]], dtype=torch.long)
    TrainingSafetyChecker.verify_token_ids(valid_tensor)

    # Invalid negative token ID
    invalid_neg = torch.tensor([[0, -1, 200]], dtype=torch.long)
    with pytest.raises(TrainingSafetyError, match="out of bounds"):
        TrainingSafetyChecker.verify_token_ids(invalid_neg)

    # Invalid oversized token ID
    invalid_over = torch.tensor([[0, 4096, 200]], dtype=torch.long)
    with pytest.raises(TrainingSafetyError, match="out of bounds"):
        TrainingSafetyChecker.verify_token_ids(invalid_over)


# =====================================================================
# 7. 512-token maximum is enforced
# =====================================================================

def test_512_token_maximum_enforced(bpe_tokenizer: BPETokenizer):
    long_input = "word " * 600  # Will exceed 512 tokens
    rec = make_approved_record(input_text=long_input, target_text="conclusion")

    builder = DatasetBuilder(bpe_tokenizer)
    train_ds, _, manifest = builder.build_from_records([rec], val_ratio=0.0)

    example = train_ds.examples[0]
    assert example.is_truncated is True
    assert example.total_token_count == 512
    assert "truncated" in example.truncation_notes.lower()
    assert manifest.provenance_summary.get("truncated_examples_count") == 1


# =====================================================================
# 8. Dataset fingerprint is deterministic
# =====================================================================

def test_dataset_fingerprint_is_deterministic(bpe_tokenizer: BPETokenizer):
    rec1 = make_approved_record(record_id="rec_fp_1", input_text="Sample input 1")
    rec2 = make_approved_record(record_id="rec_fp_2", input_text="Sample input 2")

    builder = DatasetBuilder(bpe_tokenizer)
    _, _, m1 = builder.build_from_records([rec1, rec2], seed=42)
    _, _, m2 = builder.build_from_records([rec2, rec1], seed=42)  # different initial order

    # Due to deterministic sorting, fingerprint must be identical
    assert m1.dataset_fingerprint == m2.dataset_fingerprint
    assert len(m1.dataset_fingerprint) == 16


# =====================================================================
# 9. Same seed produces reproducible dataset ordering
# =====================================================================

def test_same_seed_reproducible_dataset_ordering(bpe_tokenizer: BPETokenizer):
    records = [make_approved_record(record_id=f"rec_{i}", input_text=f"Task {i}") for i in range(10)]

    builder = DatasetBuilder(bpe_tokenizer)
    train_ds1, val_ds1, _ = builder.build_from_records(records, val_ratio=0.3, seed=123)
    train_ds2, val_ds2, _ = builder.build_from_records(records, val_ratio=0.3, seed=123)

    assert [e.example_id for e in train_ds1.examples] == [e.example_id for e in train_ds2.examples]
    assert [e.example_id for e in val_ds1.examples] == [e.example_id for e in val_ds2.examples]


# =====================================================================
# 10. Invalid tokenizer fingerprint is rejected
# =====================================================================

def test_invalid_tokenizer_fingerprint_rejected():
    with pytest.raises(TrainingSafetyError, match="Tokenizer incompatibility"):
        TrainingSafetyChecker.verify_tokenizer_compatibility("expected_fp_123", "different_fp_456")


# =====================================================================
# 11. Invalid model configuration is rejected
# =====================================================================

def test_invalid_model_config_rejected():
    cfg = ModelConfig(vocab_size=2048)  # Invariant violation: must be 4096
    invalid_model = ChakrMicro(cfg)

    with pytest.raises(InvariantViolationError, match="Vocabulary invariant violation"):
        TrainingSafetyChecker.enforce_model_invariants(invalid_model)


# =====================================================================
# 12. NaN loss aborts training
# =====================================================================

def test_nan_loss_aborts_training():
    with pytest.raises(NumericalInstabilityError, match="Loss is NaN"):
        TrainingSafetyChecker.check_loss(float("nan"), step=5)


# =====================================================================
# 13. NaN gradients abort training
# =====================================================================

def test_nan_gradients_abort_training(chakr_model: ChakrMicro):
    # Simulate a backward pass that produced NaN
    for p in chakr_model.parameters():
        p.grad = torch.zeros_like(p)
    chakr_model.embedding.weight.grad[0, 0] = float("nan")

    with pytest.raises(NumericalInstabilityError, match="contains NaN"):
        TrainingSafetyChecker.check_gradients(chakr_model, step=10)


# =====================================================================
# 14. Checkpoint metadata is validated
# =====================================================================

def test_checkpoint_metadata_validated():
    # Corrupt payload missing required keys
    corrupt_payload = {"step": 10}
    with pytest.raises(CheckpointCorruptionError, match="missing required field"):
        TrainingSafetyChecker.verify_checkpoint_metadata(corrupt_payload)


# =====================================================================
# 15. Resume-from-checkpoint works
# =====================================================================

def test_resume_from_checkpoint_works(
    chakr_model: ChakrMicro,
    bpe_tokenizer: BPETokenizer,
    tmp_path: Path,
):
    rec = make_approved_record()
    builder = DatasetBuilder(bpe_tokenizer)
    train_ds, val_ds, _ = builder.build_from_records([rec], val_ratio=0.0)

    engine1 = CPUTrainingEngine(
        model=chakr_model,
        tokenizer=bpe_tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        checkpoint_dir=tmp_path / "ckpts",
        seed=42,
    )
    res1 = engine1.train(max_steps=2, save_interval=2)
    assert res1.steps_completed == 2
    ckpt_path = res1.final_checkpoint_path

    # Fresh engine resuming from checkpoint
    fresh_model = ChakrMicro(ModelConfig())
    engine2 = CPUTrainingEngine(
        model=fresh_model,
        tokenizer=bpe_tokenizer,
        train_dataset=train_ds,
        checkpoint_dir=tmp_path / "ckpts",
    )
    resumed_step = engine2.resume(ckpt_path)
    assert resumed_step == 2

    # Weights in engine2 should now match engine1
    diff = (fresh_model.embedding.weight - chakr_model.embedding.weight).abs().max().item()
    assert diff < 1e-6


# =====================================================================
# 16. Runtime inference model weights remain unchanged
# =====================================================================

def test_runtime_inference_model_weights_unchanged(
    chakr_model: ChakrMicro,
    bpe_tokenizer: BPETokenizer,
    tmp_path: Path,
):
    inf_engine = NeuralInferenceEngine(chakr_model, bpe_tokenizer)
    initial_weights = chakr_model.embedding.weight.clone()

    # Create separate training model instance
    training_model = ChakrMicro(ModelConfig())
    rec = make_approved_record()
    builder = DatasetBuilder(bpe_tokenizer)
    train_ds, _, _ = builder.build_from_records([rec], val_ratio=0.0)

    trainer = CPUTrainingEngine(
        model=training_model,
        train_dataset=train_ds,
        checkpoint_dir=tmp_path / "ckpts",
    )
    trainer.train(max_steps=2)

    # Runtime inference engine must remain 100% untouched
    assert inf_engine.verify_weights_unmodified() is True
    assert torch.equal(chakr_model.embedding.weight, initial_weights)


# =====================================================================
# 17. Frozen neural invariants remain unchanged
# =====================================================================

def test_frozen_neural_invariants_preserved(chakr_model: ChakrMicro):
    param_count = sum(p.numel() for p in chakr_model.parameters())
    assert param_count == 3_443_136
    assert chakr_model.config.vocab_size == 4096
    assert chakr_model.config.max_seq_len == 512
    assert BOS_ID == 0
    assert EOS_ID == 1
    assert PAD_ID == 2


# =====================================================================
# 18. Validation metrics are correctly calculated
# =====================================================================

def test_validation_metrics_calculated(
    chakr_model: ChakrMicro,
    bpe_tokenizer: BPETokenizer,
):
    rec = make_approved_record()
    builder = DatasetBuilder(bpe_tokenizer)
    train_ds, val_ds, _ = builder.build_from_records([rec], val_ratio=0.0)

    val_loader = torch.utils.data.DataLoader(train_ds, batch_size=1)
    loss_fn = CausalLoss(ignore_index=2)

    result = ValidationEngine.evaluate(chakr_model, val_loader, loss_fn)
    assert isinstance(result, ValidationResult)
    assert result.val_loss > 0.0
    assert result.perplexity_valid is True
    assert result.perplexity is not None
    assert result.evaluated_examples == 1


# =====================================================================
# 19. Non-finite perplexity is reported honestly
# =====================================================================

def test_non_finite_perplexity_reported_honestly(chakr_model: ChakrMicro):
    class NanLoss(torch.nn.Module):
        def forward(self, x, y):
            return torch.tensor(float("nan"))

    dummy_loader = [{"input_ids": torch.tensor([[0, 1]]), "target_ids": torch.tensor([[1, 2]])}]
    result = ValidationEngine.evaluate(chakr_model, dummy_loader, NanLoss())

    assert result.perplexity is None
    assert result.perplexity_valid is False
    assert "Non-finite" in result.status_note


# =====================================================================
# 20. Unvalidated checkpoints cannot be promoted
# =====================================================================

def test_unvalidated_checkpoints_cannot_be_promoted(
    chakr_model: ChakrMicro,
    bpe_tokenizer: BPETokenizer,
):
    gate = RegressionGate()
    manifest = TrainingDatasetManifest(
        manifest_id="m1",
        dataset_version="v1",
        dataset_fingerprint="fp1",
        tokenizer_fingerprint="non_matching_fp",
        total_examples=1,
        total_tokens=10,
        train_examples_count=1,
        val_examples_count=0,
        test_examples_count=0,
    )
    val_res = ValidationResult(
        val_loss=float("nan"),
        perplexity=None,
        perplexity_valid=False,
        total_tokens=0,
        evaluated_examples=0,
        batches_evaluated=0,
        throughput_tokens_per_sec=0.0,
        duration_ms=1.0,
        status_note="Failed",
    )

    gate_res = gate.evaluate_candidate(
        candidate_model=chakr_model,
        candidate_version_id="candidate_unvalidated",
        checkpoint_path="dummy.pt",
        val_result=val_res,
        dataset_manifest=manifest,
        active_tokenizer=bpe_tokenizer,
    )
    assert gate_res.is_promotable is False
    assert gate_res.status == PromotionStatus.REJECTED


# =====================================================================
# 21. Regression failure blocks promotion
# =====================================================================

def test_regression_failure_blocks_promotion(
    chakr_model: ChakrMicro,
    bpe_tokenizer: BPETokenizer,
):
    def failing_regression_fn(model):
        return False, ["Failed critical domain capability regression test."]

    tok_fp = TokenizerFingerprint.from_tokenizer(bpe_tokenizer)
    manifest = TrainingDatasetManifest(
        manifest_id="m1",
        dataset_version="v1",
        dataset_fingerprint="fp1",
        tokenizer_fingerprint=tok_fp.fingerprint_hash,
        total_examples=1,
        total_tokens=10,
        train_examples_count=1,
        val_examples_count=0,
        test_examples_count=0,
    )
    val_res = ValidationResult(
        val_loss=2.5,
        perplexity=12.18,
        perplexity_valid=True,
        total_tokens=100,
        evaluated_examples=1,
        batches_evaluated=1,
        throughput_tokens_per_sec=500.0,
        duration_ms=10.0,
        status_note="Passed",
    )

    gate = RegressionGate(regression_test_fn=failing_regression_fn)
    gate_res = gate.evaluate_candidate(
        candidate_model=chakr_model,
        candidate_version_id="candidate_reg_fail",
        checkpoint_path="dummy.pt",
        val_result=val_res,
        dataset_manifest=manifest,
        active_tokenizer=bpe_tokenizer,
    )

    assert gate_res.is_promotable is False
    assert gate_res.regression_passed is False
    assert any("Failed critical domain" in v for v in gate_res.violations)


# =====================================================================
# 22. Successful promotion requires explicit action
# =====================================================================

def test_successful_promotion_requires_explicit_action(
    chakr_model: ChakrMicro,
    bpe_tokenizer: BPETokenizer,
):
    mgr = ModelUpdateManager()
    gate = RegressionGate(update_manager=mgr)
    tok_fp = TokenizerFingerprint.from_tokenizer(bpe_tokenizer)

    manifest = TrainingDatasetManifest(
        manifest_id="m1",
        dataset_version="v1",
        dataset_fingerprint="fp1",
        tokenizer_fingerprint=tok_fp.fingerprint_hash,
        total_examples=1,
        total_tokens=10,
        train_examples_count=1,
        val_examples_count=0,
        test_examples_count=0,
    )
    val_res = ValidationResult(
        val_loss=1.5,
        perplexity=4.48,
        perplexity_valid=True,
        total_tokens=100,
        evaluated_examples=1,
        batches_evaluated=1,
        throughput_tokens_per_sec=500.0,
        duration_ms=10.0,
        status_note="Passed",
    )

    gate_res = gate.evaluate_candidate(
        candidate_model=chakr_model,
        candidate_version_id="v0.2-candidate",
        checkpoint_path="ckpts/v0.2.pt",
        val_result=val_res,
        dataset_manifest=manifest,
        active_tokenizer=bpe_tokenizer,
    )

    assert gate_res.is_promotable is True
    assert mgr.active_version.version_id == "chakrmicro-v0.1"  # NOT yet promoted!

    # Explicit promotion operation required
    gate.promote_candidate("v0.2-candidate", authorized_by="OPERATOR_ALICE")
    assert mgr.active_version.version_id == "v0.2-candidate"


# =====================================================================
# 23. Rollback restores previous model version
# =====================================================================

def test_rollback_restores_previous_model_version(
    chakr_model: ChakrMicro,
    bpe_tokenizer: BPETokenizer,
):
    mgr = ModelUpdateManager()
    gate = RegressionGate(update_manager=mgr)
    tok_fp = TokenizerFingerprint.from_tokenizer(bpe_tokenizer)

    manifest = TrainingDatasetManifest(
        manifest_id="m1",
        dataset_version="v1",
        dataset_fingerprint="fp1",
        tokenizer_fingerprint=tok_fp.fingerprint_hash,
        total_examples=1,
        total_tokens=10,
        train_examples_count=1,
        val_examples_count=0,
        test_examples_count=0,
    )
    val_res = ValidationResult(
        val_loss=1.2,
        perplexity=3.32,
        perplexity_valid=True,
        total_tokens=100,
        evaluated_examples=1,
        batches_evaluated=1,
        throughput_tokens_per_sec=500.0,
        duration_ms=10.0,
        status_note="Passed",
    )

    gate.evaluate_candidate(
        candidate_model=chakr_model,
        candidate_version_id="v0.2",
        checkpoint_path="ckpts/v0.2.pt",
        val_result=val_res,
        dataset_manifest=manifest,
        active_tokenizer=bpe_tokenizer,
    )
    gate.promote_candidate("v0.2", authorized_by="OPERATOR_ALICE")
    assert mgr.active_version.version_id == "v0.2"

    # Execute rollback
    gate.rollback("chakrmicro-v0.1", reason="Anomalous behavior in production")
    assert mgr.active_version.version_id == "chakrmicro-v0.1"
    assert mgr.active_version.metadata["status"] == "ACTIVE_PRODUCTION"


# =====================================================================
# 24. Tenant isolation is preserved
# =====================================================================

def test_tenant_isolation_preserved(tmp_path: Path):
    pipeline = LearningPipeline()
    r1 = make_approved_record(record_id="rec_alice_1", owner_id="alice")
    r2 = make_approved_record(record_id="rec_bob_1", owner_id="bob")

    pipeline.add_record(r1)
    pipeline.add_record(r2)

    # Bob cannot approve Alice's record
    with pytest.raises(TenantIsolationError):
        pipeline.approve_record("rec_alice_1", owner_id="bob")

    # Export for Alice only includes Alice's data
    alice_file = tmp_path / "alice.jsonl"
    count = pipeline.export_training_dataset("alice", alice_file)
    assert count == 1


# =====================================================================
# 25. Full end-to-end approved-record -> dataset -> training -> validation flow works on CPU
# =====================================================================

def test_full_end_to_end_offline_learning_flow(
    chakr_model: ChakrMicro,
    bpe_tokenizer: BPETokenizer,
    tmp_path: Path,
):
    # 1. Experience ingestion
    r1 = make_approved_record(record_id="rec_e2e_1", input_text="Calculate 2 + 2", target_text="Result: 4")
    r2 = make_approved_record(record_id="rec_e2e_2", input_text="Calculate 3 + 5", target_text="Result: 8")

    # 2. Dataset compilation
    builder = DatasetBuilder(bpe_tokenizer)
    train_ds, val_ds, manifest = builder.build_from_records([r1, r2], val_ratio=0.5, seed=42)
    assert len(train_ds) >= 1
    assert len(val_ds) >= 1

    # 3. Offline CPU Training
    training_model = ChakrMicro(ModelConfig())
    engine = CPUTrainingEngine(
        model=training_model,
        tokenizer=bpe_tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=1,
        learning_rate=1e-4,
        checkpoint_dir=tmp_path / "e2e_ckpts",
        seed=42,
    )
    result = engine.train(max_steps=2, eval_interval=1, save_interval=1, dataset_manifest=manifest)
    assert result.success is True
    assert result.steps_completed == 2
    assert Path(result.final_checkpoint_path).is_file()

    # 4. Validation Engine
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1)
    val_res = ValidationEngine.evaluate(training_model, val_loader, engine.loss_fn)
    assert val_res.perplexity_valid is True

    # 5. Regression Gate & Promotion
    mgr = ModelUpdateManager()
    gate = RegressionGate(update_manager=mgr)
    gate_res = gate.evaluate_candidate(
        candidate_model=training_model,
        candidate_version_id="e2e_v0.2",
        checkpoint_path=result.final_checkpoint_path,
        val_result=val_res,
        dataset_manifest=manifest,
        active_tokenizer=bpe_tokenizer,
    )
    assert gate_res.is_promotable is True

    gate.promote_candidate("e2e_v0.2", authorized_by="E2E_TEST_RUNNER")
    assert mgr.active_version.version_id == "e2e_v0.2"
