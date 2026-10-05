"""
Step 110 Controlled Stage-C Pre-Training Curriculum Runner.

Executes:
1. Warm start from the Step 102 candidate checkpoint (artifacts/step102_stage_b_curriculum/checkpoints/checkpoint_0000500.pt).
2. Verifies tokenized Stage C splits, manifest hash, and tokenizer checksum.
3. Pre-trains ChakrMicro on Stage C training shards under controlled, reproducible conditions.
4. Periodic validation on Stage C validation shards and checkpointing.
5. Post-training evaluation on Stage C test split and the frozen 40-probe syntactic benchmark.
6. Saves model record in ModelRegistry, strictly preserving the immutable canonical baseline.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
    load_trained_checkpoint,
)
from chakrview.runtime.neural_inference_contract import (
    NeuralInferenceContract,
    NeuralInferencePayload,
    NeuralInferenceConfig,
)
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.config import (
    PretrainingConfig,
    TrainingHyperparameters,
    CheckpointConfig,
    EvaluationConfig,
    DataConfig,
)
from chakrview.training.dataset import StreamingTokenDataset
from chakrview.training.loss import CausalLoss
from chakrview.training.trainer import Trainer
from chakrview.training.syntactic_probe import SyntacticProbeEvaluator, SyntacticProbeMetrics
from chakrview.training.model_registry import ModelRegistry, ModelRecord, ModelStatus
from chakrview.training.curriculum_runner import evaluate_split_full


STEP102_CANDIDATE_HASH = "d5886e02e62df29fc88379ed31b37239c85fcda9648ffaf0cf92191730df9a01"


def run_stage_c_pretraining_wave(
    artifacts_dir: Path,
    warm_start_checkpoint: Optional[Path] = None,
    max_steps: int = 100,
    sequence_length: int = 256,
    batch_size: int = 2,
    accum_steps: int = 2,
    eval_interval: int = 25,
    save_interval: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes a controlled Stage-C training wave on CPU.
    """
    t_start = time.perf_counter()
    root_dir = Path(__file__).resolve().parents[2]
    tok_dir = root_dir / "data" / "experiments" / "vocab_4096"
    train_dir = root_dir / "data" / "tokenized" / "stage_c" / "train"
    val_dir = root_dir / "data" / "tokenized" / "stage_c" / "validation"
    test_dir = root_dir / "data" / "tokenized" / "stage_c" / "test"
    probes_file = root_dir / "tests" / "fixtures" / "step102_syntactic_probes.json"

    tokenizer, _ = load_tokenizer_artifacts(tok_dir)
    probe_evaluator = SyntacticProbeEvaluator(tokenizer=tokenizer, probes_path=probes_file)

    # 1. Verify Canonical Baseline Immutability
    canonical_model = instantiate_frozen_baseline()
    base_hash = compute_model_hash(canonical_model)
    assert base_hash == EXPECTED_WEIGHT_HASH, "Canonical baseline was mutated!"

    # Evaluate canonical baseline on Stage C test split for comparative rigor
    base_test_loss, base_test_ppl, _ = evaluate_split_full(canonical_model, test_dir, max_batches=20)
    base_probes = probe_evaluator.evaluate(canonical_model)

    # 2. Setup Model for Warm Start
    model = ChakrMicro(ModelConfig())
    if warm_start_checkpoint is None:
        warm_start_checkpoint = root_dir / "artifacts" / "step102_stage_b_curriculum" / "checkpoints" / "checkpoint_0000500.pt"

    assert warm_start_checkpoint.is_file(), f"Warm start checkpoint missing at {warm_start_checkpoint}"
    loaded_model, _, _ = load_trained_checkpoint(warm_start_checkpoint)
    initial_candidate_hash = compute_model_hash(loaded_model)
    assert initial_candidate_hash == STEP102_CANDIDATE_HASH, f"Warm start hash mismatch: {initial_candidate_hash}"
    model.load_state_dict(loaded_model.state_dict())

    # Pre-training Stage C evaluation of candidate before further training
    pre_val_loss, pre_val_ppl, _ = evaluate_split_full(model, val_dir, max_batches=20)
    pre_test_loss, pre_test_ppl, _ = evaluate_split_full(model, test_dir, max_batches=20)

    # 3. Training Configuration
    ckpt_dir = artifacts_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve Stage C metadata checksums
    val_meta = json.loads((val_dir / "metadata.json").read_text("utf-8"))
    tok_checksum = val_meta["tokenizer_checksum"]
    dataset_manifest_hash = val_meta["source_manifest_hash"]

    config = PretrainingConfig(
        model=ModelConfig(),
        training=TrainingHyperparameters(
            max_steps=max_steps,
            learning_rate=3e-4,
            min_learning_rate=3e-5,
            warmup_steps=20,
            gradient_accumulation_steps=accum_steps,
            gradient_clipping=1.0,
            seed=seed,
            batch_size=batch_size,
        ),
        checkpoint=CheckpointConfig(
            directory=str(ckpt_dir),
            save_interval=save_interval,
            keep_last_n=3,
        ),
        evaluation=EvaluationConfig(
            eval_interval=eval_interval,
            eval_batches=5,
            eval_on_start=True,
        ),
        data=DataConfig(
            train_path=str(train_dir),
            validation_path=str(val_dir),
            sequence_length=sequence_length,
        ),
    )

    # 4. Data Loaders
    train_ds = StreamingTokenDataset(train_dir, sequence_length=sequence_length, loop=True, seed=seed)
    val_ds = StreamingTokenDataset(val_dir, sequence_length=sequence_length, loop=True, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 5. Trainer with explicit identity verification
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer_checksum=tok_checksum,
        dataset_manifest_hash=dataset_manifest_hash,
    )

    train_result = trainer.train()
    final_step = train_result["final_step"]
    final_ckpt_path = Path(train_result["final_checkpoint"])
    final_candidate_hash = compute_model_hash(trainer.model)

    # 6. Evaluation of Trained Checkpoint
    cand_val_loss, cand_val_ppl, _ = evaluate_split_full(trainer.model, val_dir, max_batches=20)
    cand_test_loss, cand_test_ppl, _ = evaluate_split_full(trainer.model, test_dir, max_batches=20)
    cand_probes = probe_evaluator.evaluate(trainer.model)

    # 7. Model Registry Recording
    registry_db = artifacts_dir / "model_registry.db"
    registry = ModelRegistry(registry_db)
    # Register canonical baseline
    registry.register_model(ModelRecord(
        model_id="canonical_baseline",
        model_hash=EXPECTED_WEIGHT_HASH,
        architecture="ChakrMicro-v0.1",
        parameter_count=3_443_136,
        tokenizer_checksum=tok_checksum,
        dataset_identity="stage_a_smoke",
        status=ModelStatus.CANONICAL_BASELINE,
        checkpoint_path="canonical_frozen",
        validation_loss=base_test_loss,
        validation_ppl=base_test_ppl,
        top5_syntactic_acc=base_probes.top5_accuracy,
        metadata={"test_loss": base_test_loss, "test_ppl": base_test_ppl},
    ))

    # Register new Stage C candidate
    cand_record = ModelRecord(
        model_id=f"stage_c_candidate_step_{final_step:07d}",
        model_hash=final_candidate_hash,
        architecture="ChakrMicro-v0.1",
        parameter_count=3_443_136,
        tokenizer_checksum=tok_checksum,
        dataset_identity="stage_c",
        status=ModelStatus.EXPERIMENTAL_CANDIDATE,
        checkpoint_path=str(final_ckpt_path),
        parent_model_id="step102_candidate",
        validation_loss=cand_val_loss,
        validation_ppl=cand_val_ppl,
        top5_syntactic_acc=cand_probes.top5_accuracy,
        experiment_id="step110_stage_c_foundation_wave",
        metadata={
            "steps": final_step,
            "seed": seed,
            "test_loss": cand_test_loss,
            "test_ppl": cand_test_ppl,
            "probe_top1_accuracy": cand_probes.top1_accuracy,
            "probe_top5_accuracy": cand_probes.top5_accuracy,
        },
    )
    registry.register_model(cand_record)

    # 8. Re-verify Canonical Baseline bit-exactness post-training
    assert compute_model_hash(instantiate_frozen_baseline()) == EXPECTED_WEIGHT_HASH, "Baseline mutated post-training!"

    elapsed_total = time.perf_counter() - t_start

    summary = {
        "milestone": "Step 110",
        "experiment_name": "Stage C Foundation Pre-Training Wave",
        "elapsed_sec": round(elapsed_total, 2),
        "steps_completed": final_step,
        "seed": seed,
        "warm_start_hash": initial_candidate_hash,
        "final_candidate_hash": final_candidate_hash,
        "pre_training_stage_c": {
            "validation_loss": pre_val_loss,
            "validation_perplexity": pre_val_ppl,
            "test_loss": pre_test_loss,
            "test_perplexity": pre_test_ppl,
        },
        "post_training_stage_c": {
            "final_train_loss": trainer.metrics.history[-1]["train_loss"],
            "validation_loss": cand_val_loss,
            "validation_perplexity": cand_val_ppl,
            "test_loss": cand_test_loss,
            "test_perplexity": cand_test_ppl,
            "probe_top1_accuracy": cand_probes.top1_accuracy,
            "probe_top5_accuracy": cand_probes.top5_accuracy,
            "probe_mean_log_prob": cand_probes.mean_target_log_prob,
        },
        "canonical_baseline": {
            "weight_hash": EXPECTED_WEIGHT_HASH,
            "test_loss": base_test_loss,
            "test_perplexity": base_test_ppl,
            "probe_top5_accuracy": base_probes.top5_accuracy,
        },
    }

    summary_path = artifacts_dir / "step110_stage_c_training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
