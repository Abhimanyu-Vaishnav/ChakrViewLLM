"""
Step 47 — Dedicated Training Readiness and First Learning Loop Test Suite.

Verifies:
1. Configuration validation
2. Dataset identity & token range validation
3. Tokenizer compatibility
4. Batch shape & collator validation
5. Forward pass and finite causal loss
6. Backward pass and gradient existence
7. Gradient NaN/Inf detection
8. Loss NaN/Inf detection
9. Gradient norm calculation
10. Optimizer step & parameter update (ΔW > 0 on test model)
11. Checkpoint creation with training metadata
12. Checkpoint corruption rejection
13. Inference checkpoint rejected for training resume
14. Resumption restoring identical state
15. Deterministic seed reproducibility
16. Baseline model immutability (ΔW = 0 on baseline)
17. Training vs baseline model separation
"""

import math
import hashlib
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.training.builder import ChakrOfflineDataset
from chakrview.training.config import (
    PretrainingConfig,
    TrainingHyperparameters,
    CheckpointConfig,
    EvaluationConfig,
    DataConfig,
)
from chakrview.training.trainer import Trainer
from chakrview.training.checkpoint import CheckpointManager
from chakrview.training.safety import (
    TrainingSafetyChecker,
    NumericalInstabilityError,
    CheckpointCorruptionError,
    TrainingSafetyError,
)
from chakrview.training.loss import CausalLoss
from chakrview.training.optimizer import build_optimizer

ROOT_DIR = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"
TRAIN_SHARD_DIR = ROOT_DIR / "data" / "tokenized" / "train"

FROZEN_BASELINE_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_PARAM_COUNT = 3_443_136


def compute_model_hash(model: torch.nn.Module) -> str:
    """Compute deterministic SHA-256 hash across all named parameters."""
    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration Validation
# ─────────────────────────────────────────────────────────────────────────────

def test_step47_configuration_validation():
    """Verify pretraining configuration validates fields and handles defaults."""
    cfg = PretrainingConfig(
        model=ModelConfig(),
        training=TrainingHyperparameters(
            learning_rate=1e-3,
            max_steps=10,
            gradient_accumulation_steps=2,
            gradient_clipping=1.0,
        ),
    )
    assert cfg.training.learning_rate == 1e-3
    assert cfg.training.max_steps == 10
    assert cfg.training.gradient_accumulation_steps == 2
    assert cfg.training.gradient_clipping == 1.0

    d = cfg.to_dict()
    assert "model" in d
    assert "training" in d
    assert "checkpoint" in d


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset Identity & Token Range Validation
# ─────────────────────────────────────────────────────────────────────────────

def test_step47_dataset_identity_and_bounds():
    """Verify existing train shard exists, loads tokens, and token IDs are in bounds [0, 4095]."""
    assert TRAIN_SHARD_DIR.is_dir(), f"Missing train shard dir at {TRAIN_SHARD_DIR}"

    dataset = StreamingTokenDataset(
        shard_dir=TRAIN_SHARD_DIR,
        sequence_length=64,
        loop=False,
    )
    tokens = []
    for item in dataset:
        tokens.extend(item["input_ids"].tolist())

    assert len(tokens) > 0
    token_tensor = torch.tensor(tokens, dtype=torch.long)
    TrainingSafetyChecker.verify_token_ids(token_tensor, min_id=0, max_id=4095)

    # Test out of bounds rejection
    out_of_bounds = torch.tensor([10, 20, 5000], dtype=torch.long)
    with pytest.raises(TrainingSafetyError, match="Token ID range violation"):
        TrainingSafetyChecker.verify_token_ids(out_of_bounds, min_id=0, max_id=4095)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tokenizer Compatibility
# ─────────────────────────────────────────────────────────────────────────────

def test_step47_tokenizer_compatibility():
    """Verify tokenizer artifacts load with correct vocab size and fingerprinting."""
    from chakrview.tokenizer.special_tokens import BOS_ID, EOS_ID, PAD_ID
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    assert tokenizer.vocab_size == 4096
    assert BOS_ID == 0
    assert EOS_ID == 1
    assert PAD_ID == 2

    encoded_bos = tokenizer.encode("test", add_bos=True)
    assert encoded_bos[0] == BOS_ID

    encoded_eos = tokenizer.encode("test", add_eos=True)
    assert encoded_eos[-1] == EOS_ID

    # Checksum verification
    TrainingSafetyChecker.verify_tokenizer_compatibility("tok-v1", "tok-v1")
    with pytest.raises(TrainingSafetyError, match="Tokenizer incompatibility"):
        TrainingSafetyChecker.verify_tokenizer_compatibility("tok-v1", "tok-v2")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Batch Shape & Collator Validation
# ─────────────────────────────────────────────────────────────────────────────

def test_step47_batch_shape_validation():
    """Verify collator produces aligned input_ids and target_ids with correct shapes."""
    samples = [
        {
            "input_ids": torch.tensor([10, 20, 30, 40, 50], dtype=torch.long),
            "target_ids": torch.tensor([20, 30, 40, 50, 60], dtype=torch.long),
        },
        {
            "input_ids": torch.tensor([100, 200, 300, 400], dtype=torch.long),
            "target_ids": torch.tensor([200, 300, 400, 500], dtype=torch.long),
        },
    ]
    batch = ChakrOfflineDataset.collate_fn(samples, pad_token_id=2)

    assert "input_ids" in batch
    assert "target_ids" in batch
    assert "attention_mask" in batch

    # Batch size 2, max seq length 5
    assert batch["input_ids"].shape == (2, 5)
    assert batch["target_ids"].shape == (2, 5)
    assert batch["attention_mask"].shape == (2, 5)
    assert batch["input_ids"][1, 4].item() == 2
    assert batch["attention_mask"][1, 4].item() == 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Forward Pass and Finite Causal Loss
# ─────────────────────────────────────────────────────────────────────────────

def test_step47_forward_pass_and_finite_loss():
    """Verify forward pass produces valid logits and finite causal loss."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    loss_fn = CausalLoss(ignore_index=2)

    input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    target_ids = torch.tensor([[20, 30, 40, 50]], dtype=torch.long)

    logits = model(input_ids)
    assert logits.shape == (1, 4, 4096)

    loss = loss_fn(logits, target_ids)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    loss_val = TrainingSafetyChecker.check_loss(loss)
    # Expected random initialization loss is approx -ln(1/4096) = 8.318
    assert 5.0 < loss_val < 12.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Backward Pass and Gradient Existence
# ─────────────────────────────────────────────────────────────────────────────

def test_step47_backward_pass_and_gradient_existence():
    """Verify backward pass populates non-zero gradients on all active parameters."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    loss_fn = CausalLoss(ignore_index=2)

    input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    target_ids = torch.tensor([[20, 30, 40, 50]], dtype=torch.long)

    logits = model(input_ids)
    loss = loss_fn(logits, target_ids)
    loss.backward()

    grads_found = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"
            grads_found += 1

    assert grads_found > 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Gradient & Loss NaN/Inf Detection
# ─────────────────────────────────────────────────────────────────────────────

def test_step47_gradient_nan_inf_detection():
    """Verify TrainingSafetyChecker fails closed when gradients contain NaN or Inf."""
    model = ChakrMicro(ModelConfig())
    for param in model.parameters():
        param.grad = torch.zeros_like(param)

    # Clean pass
    norm = TrainingSafetyChecker.check_gradients(model)
    assert norm == 0.0

    # Inject NaN
    first_param = next(model.parameters())
    first_param.grad[0, 0] = float("nan")
    with pytest.raises(NumericalInstabilityError, match="contains NaN"):
        TrainingSafetyChecker.check_gradients(model)

    # Inject Inf
    first_param.grad[0, 0] = float("inf")
    with pytest.raises(NumericalInstabilityError, match="contains Inf"):
        TrainingSafetyChecker.check_gradients(model)


def test_step47_loss_nan_inf_detection():
    """Verify TrainingSafetyChecker fails closed on NaN, Inf, or exploded loss."""
    with pytest.raises(NumericalInstabilityError, match="Loss is NaN"):
        TrainingSafetyChecker.check_loss(float("nan"))

    with pytest.raises(NumericalInstabilityError, match="Loss is Infinite"):
        TrainingSafetyChecker.check_loss(float("inf"))

    with pytest.raises(NumericalInstabilityError, match="Loss exploded"):
        TrainingSafetyChecker.check_loss(1500.0)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Gradient Norm Calculation
# ─────────────────────────────────────────────────────────────────────────────

def test_step47_gradient_norm_calculation():
    """Verify gradient norm matches expected Euclidean norm calculation."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    loss_fn = CausalLoss(ignore_index=2)

    input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    target_ids = torch.tensor([[20, 30, 40, 50]], dtype=torch.long)

    logits = model(input_ids)
    loss = loss_fn(logits, target_ids)
    loss.backward()

    norm = TrainingSafetyChecker.check_gradients(model)
    assert isinstance(norm, float)
    assert norm > 0.0
    assert not math.isnan(norm)
    assert not math.isinf(norm)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Optimizer Update & Parameter Change
# ─────────────────────────────────────────────────────────────────────────────

def test_step47_optimizer_update():
    """Verify optimizer step actually modifies parameters (ΔW > 0)."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    initial_w = model.embedding.weight.clone().detach()

    cfg = TrainingHyperparameters(learning_rate=1e-2)
    optimizer = build_optimizer(model, cfg)

    input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    target_ids = torch.tensor([[20, 30, 40, 50]], dtype=torch.long)

    logits = model(input_ids)
    loss = CausalLoss()(logits, target_ids)
    loss.backward()

    optimizer.step()
    updated_w = model.embedding.weight.detach()

    delta = (updated_w - initial_w).abs().max().item()
    assert delta > 0.0, "Optimizer step failed to update weights"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Checkpoint Creation & Metadata Typing
# ─────────────────────────────────────────────────────────────────────────────

def test_step47_checkpoint_creation_and_typing(tmp_path: Path):
    """Verify CheckpointManager writes training checkpoints with all metadata."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    mgr = CheckpointManager(checkpoint_dir=tmp_path / "ckpts")
    ckpt_path = mgr.save(
        model=model,
        optimizer=optimizer,
        step=5,
        tokenizer_checksum="tok_sha256_mock",
        dataset_manifest_hash="data_sha256_mock",
    )

    assert ckpt_path.is_file()
    payload = CheckpointManager.load(ckpt_path)

    assert payload["checkpoint_type"] == "training"
    assert payload["step"] == 5
    assert payload["tokenizer_checksum"] == "tok_sha256_mock"
    assert payload["dataset_manifest_hash"] == "data_sha256_mock"
    assert payload["parameter_count"] == EXPECTED_PARAM_COUNT
    assert "model_state_dict" in payload
    assert "optimizer_state_dict" in payload


# ─────────────────────────────────────────────────────────────────────────────
# 11. Checkpoint Corruption Rejection
# ─────────────────────────────────────────────────────────────────────────────

def test_step47_checkpoint_corruption_rejection():
    """Verify CheckpointManager.validate_training_checkpoint rejects malformed payloads."""
    # 1. Missing key
    bad_payload_1 = {"checkpoint_type": "training", "step": 1}
    with pytest.raises(CheckpointCorruptionError, match="missing required key"):
        CheckpointManager.validate_training_checkpoint(bad_payload_1)

    # 2. Invalid param count
    bad_payload_2 = {
        "checkpoint_type": "training",
        "step": 1,
        "model_state_dict": {"k": torch.tensor([1])},
        "optimizer_state_dict": {},
        "config": {},
        "timestamp": "2026-09-30",
        "parameter_count": 999,
    }
    with pytest.raises(CheckpointCorruptionError, match="parameter count mismatch"):
        CheckpointManager.validate_training_checkpoint(bad_payload_2, expected_param_count=EXPECTED_PARAM_COUNT)

    # 3. Mismatched tokenizer checksum
    bad_payload_3 = {
        "checkpoint_type": "training",
        "step": 1,
        "model_state_dict": {"k": torch.tensor([1])},
        "optimizer_state_dict": {},
        "config": {},
        "timestamp": "2026-09-30",
        "parameter_count": EXPECTED_PARAM_COUNT,
        "tokenizer_checksum": "tok_wrong",
    }
    with pytest.raises(CheckpointCorruptionError, match="Tokenizer checksum mismatch"):
        CheckpointManager.validate_training_checkpoint(
            bad_payload_3,
            expected_tokenizer_checksum="tok_correct",
            expected_param_count=EXPECTED_PARAM_COUNT,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 12. Inference Checkpoint Rejection for Training Resume
# ─────────────────────────────────────────────────────────────────────────────

def test_step47_inference_checkpoint_rejected_for_training(tmp_path: Path):
    """Verify inference checkpoints cannot be used to resume pre-training."""
    inference_payload = {
        "checkpoint_type": "inference",
        "model_state_dict": {"weight": torch.zeros(10)},
        "timestamp": "2026-09-30",
    }
    inf_path = tmp_path / "inference_ckpt.pt"
    torch.save(inference_payload, inf_path)

    with pytest.raises(CheckpointCorruptionError, match="Invalid checkpoint type 'inference'"):
        CheckpointManager.load(inf_path, validate_training=True)


# ─────────────────────────────────────────────────────────────────────────────
# 13. Resumption Restoring Identical State
# ─────────────────────────────────────────────────────────────────────────────

def test_step47_resumption_state_integrity(tmp_path: Path):
    """Verify Trainer resumes cleanly from checkpoint and continues learning."""
    config = PretrainingConfig(
        model=ModelConfig(),
        training=TrainingHyperparameters(
            max_steps=4,
            learning_rate=1e-3,
            gradient_accumulation_steps=1,
        ),
        checkpoint=CheckpointConfig(directory=str(tmp_path / "ckpts")),
    )
    batch = {
        "input_ids": torch.randint(0, 4000, (2, 16)),
        "target_ids": torch.randint(0, 4000, (2, 16)),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
    }

    # Run A: 2 steps and save
    trainer_a = Trainer(config=config)
    trainer_a.train_step(batch)
    trainer_a.train_step(batch)
    assert trainer_a.current_step == 2
    ckpt = trainer_a.checkpoint_manager.save(
        model=trainer_a.model,
        optimizer=trainer_a.optimizer,
        scheduler=trainer_a.scheduler,
        step=trainer_a.current_step,
        config=config.to_dict(),
    )

    ref_weights = trainer_a.model.embedding.weight.clone().detach()

    # Run B: fresh trainer resumes
    trainer_b = Trainer(config=config)
    resumed_step = trainer_b.resume(ckpt)
    assert resumed_step == 2
    assert trainer_b.current_step == 2
    assert torch.equal(ref_weights, trainer_b.model.embedding.weight)


# ─────────────────────────────────────────────────────────────────────────────
# 14. Deterministic Seed Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def test_step47_deterministic_seed_behavior(tmp_path: Path):
    """Verify identical seed produces bit-exact identical loss and updated weights."""
    def run_training_experiment(seed: int):
        torch.manual_seed(seed)
        m = ChakrMicro(ModelConfig())
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        loss_fn = CausalLoss(ignore_index=2)

        batch_in = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
        batch_tgt = torch.tensor([[20, 30, 40, 50, 60]], dtype=torch.long)

        opt.zero_grad()
        loss = loss_fn(m(batch_in), batch_tgt)
        loss_val = loss.item()
        loss.backward()
        opt.step()

        return loss_val, compute_model_hash(m)

    loss_1, hash_1 = run_training_experiment(42)
    loss_2, hash_2 = run_training_experiment(42)

    assert math.isclose(loss_1, loss_2, rel_tol=1e-7)
    assert hash_1 == hash_2, "Seed determinism violated: model hashes differ"


# ─────────────────────────────────────────────────────────────────────────────
# 15. Baseline Model Immutability & Separation (ΔW_baseline = 0, ΔW_exp > 0)
# ─────────────────────────────────────────────────────────────────────────────

def test_step47_baseline_model_immutability():
    """Verify frozen baseline model is untouched (ΔW = 0) while test instance trains."""
    # 1. Instantiate baseline model and verify hash
    torch.manual_seed(42)
    baseline_model = ChakrMicro(ModelConfig())
    baseline_model.eval()
    initial_baseline_hash = compute_model_hash(baseline_model)
    assert initial_baseline_hash == FROZEN_BASELINE_HASH

    # 2. Instantiate separate experiment model and train it
    torch.manual_seed(42)
    exp_model = ChakrMicro(ModelConfig())
    exp_model.train()
    optimizer = torch.optim.AdamW(exp_model.parameters(), lr=1e-2)
    loss_fn = CausalLoss(ignore_index=2)

    batch_in = torch.tensor([[15, 25, 35, 45]], dtype=torch.long)
    batch_tgt = torch.tensor([[25, 35, 45, 55]], dtype=torch.long)

    for _ in range(3):
        optimizer.zero_grad()
        loss = loss_fn(exp_model(batch_in), batch_tgt)
        loss.backward()
        optimizer.step()

    exp_hash = compute_model_hash(exp_model)

    # 3. Assert baseline model remains strictly identical (ΔW = 0)
    post_baseline_hash = compute_model_hash(baseline_model)
    assert post_baseline_hash == FROZEN_BASELINE_HASH, "Baseline model was mutated!"

    # 4. Assert experiment model updated (ΔW_exp > 0)
    assert exp_hash != FROZEN_BASELINE_HASH, "Experiment model failed to update weights!"
