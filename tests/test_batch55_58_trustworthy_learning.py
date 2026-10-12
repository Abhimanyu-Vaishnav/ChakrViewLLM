"""
Integration and Regression Tests for Master Sprint Batches 55-58:
- Batch 55: Impossible-to-Bypass Training Trust Gate & Role Contamination Prevention
- Batch 56: Verified Micro-Benchmark Integrity, Multi-Domain Answer Checking, and Family Isolation
- Batch 57: Generation Failure Diagnostics, Wikipedia Table Attractor Analysis, and Reproducible Scoring
- Batch 58: Hardware-Adaptive Runtime Observability, Memory Budgeting, and Safe Execution Fallback
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.hardware import (
    ComputeDevice,
    HardwareCapabilityDetector,
    HardwareProfile,
    PrecisionType,
)
from chakrview.training.sft_engine import SFTConfig, SFTDataset, SFTExperimentEngine
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    DeterministicEvaluator,
    compute_file_sha256,
    load_tokenizer_artifacts,
)
from experiments.batch43_46_architecture_capability_sprint.hardware_readiness.scheduler import (
    HardwareAdaptiveScheduler,
    WorkloadType,
)
from experiments.batch51_54_correctness_sprint.data.dataset_eligibility_gate import (
    extract_source_family,
)
from experiments.batch55_58_trustworthy_learning.benchmark.micro_benchmark import (
    VERIFIED_MICROBENCHMARK_ITEMS,
    DataRole,
    evaluate_benchmark_item,
)

EXPECTED_CANONICAL_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_CANONICAL_PARAMS = 3_443_136
EXPECTED_STEP10000_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"


# ==============================================================================
# BATCH 55: ENFORCED TRAINING TRUST GATE & CONTAMINATION REFUSAL
# ==============================================================================


def test_batch55_unreviewed_record_refusal_at_training_boundary(tmp_path):
    """
    Proves that SFTDataset and SFTExperimentEngine refuse records with
    PENDING_HUMAN_REVIEW unless explicit allow_exploratory_opt_in=True is supplied.
    """
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    unreviewed_file = tmp_path / "unreviewed_data.jsonl"
    with open(unreviewed_file, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "id": "UNREVIEWED-RECORD-001",
                    "prompt": "<|im_start|>user\nHello?<|im_end|>\n<|im_start|>assistant\n",
                    "response": "Hi there.<|im_end|>\n",
                    "review_status": "PENDING_HUMAN_REVIEW",
                    "status": "SOURCE_SUPPORTED",
                }
            )
            + "\n"
        )

    # 1. Normal verified mode MUST refuse pending review data with PermissionError
    with pytest.raises(PermissionError, match="Trust Gate Refusal: Record UNREVIEWED-RECORD-001 is PENDING_HUMAN_REVIEW"):
        SFTDataset(
            jsonl_path=unreviewed_file,
            tokenizer=tokenizer,
            allow_exploratory_opt_in=False,
            enforce_trust_gate=True,
            is_training=True,
        )

    # 2. Explicit exploratory opt-in permits loading but keeps status unchanged
    ds_optin = SFTDataset(
        jsonl_path=unreviewed_file,
        tokenizer=tokenizer,
        allow_exploratory_opt_in=True,
        enforce_trust_gate=True,
        is_training=True,
    )
    assert len(ds_optin) == 1
    assert ds_optin.records[0]["review_status"] == "PENDING_HUMAN_REVIEW"


def test_batch55_held_out_benchmark_contamination_refusal(tmp_path):
    """
    Proves that SFTDataset strictly rejects records tagged with HELD_OUT_BENCHMARK
    from entering the gradient update training pipeline.
    """
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    contaminated_file = tmp_path / "contaminated_train.jsonl"
    with open(contaminated_file, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "id": "BENCHMARK-ITEM-LEAKED",
                    "prompt": "<|im_start|>user\nWhat is 2 + 3?<|im_end|>\n<|im_start|>assistant\n",
                    "response": "5<|im_end|>\n",
                    "data_role": DataRole.HELD_OUT_BENCHMARK.value,
                    "review_status": "ANSWER_VERIFIED",
                }
            )
            + "\n"
        )

    with pytest.raises(ValueError, match="Contamination error: Record BENCHMARK-ITEM-LEAKED with role HELD_OUT_BENCHMARK cannot enter training"):
        SFTDataset(
            jsonl_path=contaminated_file,
            tokenizer=tokenizer,
            is_training=True,
        )


def test_batch55_checkpoint_metadata_records_dataset_mode(tmp_path):
    """
    Proves that SFT checkpoints record dataset mode and trust gate flags.
    """
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    clean_rec_file = tmp_path / "verified_sample.jsonl"
    with open(clean_rec_file, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "id": "VERIFIED-REC-01",
                    "prompt": "<|im_start|>user\nHello?<|im_end|>\n<|im_start|>assistant\n",
                    "response": "World.<|im_end|>\n",
                    "review_status": "ANSWER_VERIFIED",
                    "data_role": "TRAIN",
                }
            )
            + "\n"
        )

    cfg = SFTConfig(
        repo_root=Path("."),
        train_file=clean_rec_file,
        val_file=clean_rec_file,
        output_dir=tmp_path / "out",
        base_checkpoint=Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"),
        steps=1,
        batch_size=1,
        allow_exploratory_opt_in=False,
    )

    engine = SFTExperimentEngine(cfg)
    engine.save_checkpoint(step=1)

    saved_ckpt = torch.load(tmp_path / "out" / "sft_checkpoint_step_1.pt", map_location="cpu", weights_only=False)
    assert saved_ckpt["dataset_mode"] == "VERIFIED_TRAINING"
    assert saved_ckpt["allow_exploratory_opt_in"] is False
    assert saved_ckpt["enforce_trust_gate"] is True


# ==============================================================================
# BATCH 56: VERIFIED MICRO-BENCHMARK INTEGRITY
# ==============================================================================


def test_batch56_micro_benchmark_items_and_deterministic_checks():
    """
    Verifies that the micro-benchmark items cover Math, Coding, and Science
    and possess deterministic scoring without heuristic keyword matching.
    """
    assert len(VERIFIED_MICROBENCHMARK_ITEMS) == 9

    domains = {item.domain for item in VERIFIED_MICROBENCHMARK_ITEMS}
    assert domains == {"mathematics", "code", "science"}

    for item in VERIFIED_MICROBENCHMARK_ITEMS:
        assert item.data_role == DataRole.HELD_OUT_BENCHMARK.value
        assert item.verification_status in ("ANSWER_VERIFIED", "EXECUTION_VERIFIED")

        # Self-evaluation on ground truth must pass 100%
        res = evaluate_benchmark_item(item, item.expected_answer, eos_reached=True)
        assert res["task_success"] is True, f"Benchmark item {item.id} failed self-test"


def test_batch56_benchmark_zero_contamination_against_training_set():
    """
    Proves that benchmark families have zero overlap with training families.
    """
    train_path = Path("experiments/batch51_54_correctness_sprint/data/train_clean.jsonl")
    assert train_path.exists()

    train_families = set()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_families.add(extract_source_family(json.loads(line)))

    benchmark_families = {item.family for item in VERIFIED_MICROBENCHMARK_ITEMS}
    overlap = train_families.intersection(benchmark_families)
    assert len(overlap) == 0, f"Contamination detected between benchmark and training: {overlap}"


# ==============================================================================
# BATCH 57: GENERATION FAILURE DIAGNOSTICS & REPETITION DETECTION
# ==============================================================================


def test_batch57_scoring_repetition_detector():
    """
    Verifies that repetitive Wikipedia table outputs are correctly identified as failures.
    """
    item = VERIFIED_MICROBENCHMARK_ITEMS[0]  # What is 2 + 3?
    
    table_rep_output = "|-\n|200\n|-\n|2005\n|-\n|2005\n|2005\n"
    res = evaluate_benchmark_item(item, table_rep_output, eos_reached=False)
    assert res["repetition_detected"] is True
    assert res["task_success"] is False


# ==============================================================================
# BATCH 58: HARDWARE-ADAPTIVE RUNTIME & SCHEDULING LOGIC
# ==============================================================================


def test_batch58_real_hardware_discovery_and_scheduling():
    """
    Tests real hardware discovery and ensures scheduler decisions reflect host realities.
    """
    profile = HardwareCapabilityDetector.detect()
    assert profile.physical_cores >= 1
    assert profile.available_ram_bytes > 0

    scheduler = HardwareAdaptiveScheduler(profile)
    plan = scheduler.plan_workload(WorkloadType.TRAINING_SFT, total_items=20, allow_gpu=True)

    # On CPU host, target must be CPU with explicit fallback note
    if not torch.cuda.is_available():
        assert plan.target_device == ComputeDevice.CPU
        assert "No CUDA" in plan.fallback_reason
    assert plan.memory_budget_mb > 0
    assert plan.safety_reserve_mb > 0
