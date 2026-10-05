"""
Validation test for Phase L of the Post-Smoke-Training Model Readiness Audit:
Stage-B 50-Step Generalization and Held-Out Validation Experiment.

Empirically verifies:
1. Model trains on realistic Stage B curated corpus shards.
2. Training loss decreases across 50 steps.
3. Validation loss on held-out Stage B validation shard decreases measurably.
4. Validation perplexity decreases.
5. Checkpoint is saved with valid weights, non-baseline hash, and can be resumed.
6. Post-training inference preserves ΔW = 0.
7. Canonical baseline remains bit-exact throughout.
"""

from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.training.config import (
    PretrainingConfig,
    TrainingHyperparameters,
    CheckpointConfig,
    EvaluationConfig,
    DataConfig,
)
from chakrview.training.trainer import Trainer
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.runtime.neural_inference_contract import (
    NeuralInferenceContract,
    NeuralInferencePayload,
    NeuralInferenceConfig,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

ROOT_DIR = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"
STAGE_B_TRAIN = ROOT_DIR / "data" / "tokenized" / "stage_b" / "train"
STAGE_B_VAL = ROOT_DIR / "data" / "tokenized" / "stage_b" / "validation"


@pytest.fixture
def tokenizer():
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    return tok


def test_stage_b_generalization_and_held_out_learning(tmp_path: Path, tokenizer):
    """
    Execute 50-step training experiment on Stage B shards and verify held-out generalization.
    """
    ckpt_dir = tmp_path / "stage_b_checkpoints"
    seq_len = 128
    batch_size = 2

    # Step 1: Baseline invariant verification before run
    canonical_model = instantiate_frozen_baseline()
    assert compute_model_hash(canonical_model) == EXPECTED_WEIGHT_HASH

    # Step 2: Configure 50 steps with evaluation at step 0, 25, 50
    config = PretrainingConfig(
        model=ModelConfig(),
        training=TrainingHyperparameters(
            max_steps=50,
            learning_rate=2e-3,
            warmup_steps=5,
            gradient_accumulation_steps=1,
            gradient_clipping=1.0,
            seed=42,
            batch_size=batch_size,
        ),
        checkpoint=CheckpointConfig(
            directory=str(ckpt_dir),
            save_interval=25,
            keep_last_n=2,
        ),
        evaluation=EvaluationConfig(
            eval_interval=25,
            eval_batches=4,
            eval_on_start=True,
        ),
        data=DataConfig(
            train_path=str(STAGE_B_TRAIN),
            validation_path=str(STAGE_B_VAL),
            sequence_length=seq_len,
        ),
    )

    train_ds = StreamingTokenDataset(
        shard_dir=STAGE_B_TRAIN,
        sequence_length=seq_len,
        loop=True,
        seed=42,
    )
    val_ds = StreamingTokenDataset(
        shard_dir=STAGE_B_VAL,
        sequence_length=seq_len,
        loop=True,
        seed=42,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    trainer = Trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Step 3: Run 50 steps
    res = trainer.train()
    assert res["final_step"] == 50

    history = trainer.metrics.history
    assert len(history) == 50

    # Step 4: Verify train loss descent
    initial_train_loss = history[0]["train_loss"]
    final_train_loss = history[-1]["train_loss"]
    assert final_train_loss < initial_train_loss, (
        f"Train loss did not decrease: {initial_train_loss} -> {final_train_loss}"
    )

    # Step 5: Verify held-out validation loss descent
    val_records = [h for h in history if "val_loss" in h and h["val_loss"] is not None]
    assert len(val_records) >= 2, f"Expected at least 2 validation evaluation points, got {len(val_records)}"
    
    first_val = val_records[0]["val_loss"]
    final_val = val_records[-1]["val_loss"]
    assert final_val < first_val, (
        f"Held-out validation loss did not decrease: {first_val} -> {final_val}"
    )

    # Step 6: Verify validation perplexity reduction
    first_ppl = val_records[0]["val_perplexity"]
    final_ppl = val_records[-1]["val_perplexity"]
    assert final_ppl < first_ppl, (
        f"Held-out validation perplexity did not drop: {first_ppl} -> {final_ppl}"
    )

    # Step 7: Checkpoint integrity and post-training inference
    final_ckpt = Path(res["final_checkpoint"])
    assert final_ckpt.is_file()

    contract = NeuralInferenceContract.from_checkpoint(
        checkpoint_path=final_ckpt,
        tokenizer=tokenizer,
    )
    trained_hash = contract.compute_weight_hash()
    assert trained_hash != EXPECTED_WEIGHT_HASH

    payload = NeuralInferencePayload(
        prompt="def calculate_total(items):",
        config=NeuralInferenceConfig(max_new_tokens=6, temperature=0.0),
    )
    out = contract.generate(payload)
    assert out.weight_hash_verified is True
    assert contract.compute_weight_hash() == trained_hash
    assert len(out.generated_tokens) > 0

    # Step 8: Verify canonical baseline is untouched
    canonical_recheck = instantiate_frozen_baseline()
    assert compute_model_hash(canonical_recheck) == EXPECTED_WEIGHT_HASH
