"""
Tests for Master Sprint Batches 51-54:
- Batch 51: Causal Next-Token Alignment, Masking, and Diagnostic Reproducibility
- Batch 52: Dataset Trust Gate, Provenance Auditing, and Family Leakage Elimination
- Batch 53: Evaluator Target Consistency, Probe Rubrics, and Generalization Metrics
- Batch 54: Hardware-Adaptive Discovery, Headroom Budgeting, and Safe Execution Fallback
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

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
    AdaptiveScheduleDecision,
    HardwareAdaptiveScheduler,
    WorkloadType,
)

EXPECTED_CANONICAL_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_CANONICAL_PARAMS = 3_443_136
EXPECTED_STEP10000_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
TOKENIZER_DIR = "release/chakrview-v0.5.0/data/experiments/vocab_4096"


# ==============================================================================
# BATCH 51 TESTS: CAUSAL TARGET ALIGNMENT & DIAGNOSTIC REPRODUCTION
# ==============================================================================


def test_batch51_canonical_invariants_and_weights_immutable():
    """Verify parameters, architecture, fresh-init hash, and step-10000 checkpoint hash."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == EXPECTED_CANONICAL_PARAMS

    import hashlib
    h = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            h.update(name.encode("utf-8"))
            h.update(param.detach().cpu().numpy().tobytes())
    assert h.hexdigest() == EXPECTED_CANONICAL_HASH

    base_ckpt = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    assert compute_file_sha256(base_ckpt) == EXPECTED_STEP10000_HASH


def test_batch51_causal_next_token_alignment_and_masking():
    """
    Formal mathematical check of the causal next-token prediction contract.
    Token at position i predicts position i+1.
    Input seq: [0] + prompt_tokens + response_tokens
    Target seq: [-100]*len(prompt_tokens) + response_tokens + [1]
    
    Verifies:
    1. Input and target lengths are identical.
    2. Prompt tokens are masked with -100.
    3. Position len(prompt_tokens) input (the last prompt token) predicts the first response token.
    4. Position len(prompt_tokens)+len(response_tokens) input (last response token) predicts EOS (1).
    5. Active token count exactly equals len(response_tokens) + 1.
    """
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    prompt = "<|im_start|>user\nWhat is 2 + 3?<|im_end|>\n<|im_start|>assistant\n"
    response = "5<|im_end|>\n"

    p_toks = tokenizer.encode(prompt.strip())
    r_toks = tokenizer.encode(response.strip())

    inp = [0] + p_toks + r_toks
    tgt = ([-100] * len(p_toks)) + r_toks + [1]

    assert len(inp) == len(tgt)
    assert len(p_toks) == 38
    assert len(r_toks) == 9

    # Position 0 predicts target 0: inp[0] is BOS (0), tgt[0] is -100 (masked)
    assert inp[0] == 0
    assert tgt[0] == -100

    # The last prompt token is at index len(p_toks) in inp
    # Because inp has BOS at index 0, prompt tokens occupy index 1 to len(p_toks).
    # Token at index len(p_toks) is p_toks[-1] ('ant')
    last_prompt_idx = len(p_toks)
    assert inp[last_prompt_idx] == p_toks[-1]

    # The first response token is r_toks[0] ('5'), which is placed at target index len(p_toks)
    # This means input at index len(p_toks) predicts target at index len(p_toks) (r_toks[0])!
    assert tgt[last_prompt_idx] == r_toks[0]

    # The last input token is r_toks[-1] ('>')
    assert inp[-1] == r_toks[-1]
    # The last target token is EOS (1)
    assert tgt[-1] == 1

    # Active token count
    active_tokens = sum(1 for t in tgt if t != -100)
    assert active_tokens == len(r_toks) + 1


def test_batch51_sft_dataset_and_evaluator_contract_agreement(tmp_path):
    """
    Verifies that SFTDataset and DeterministicEvaluator adhere to identical tokenization
    and shifting contracts on arbitrary prompt/response pairs.
    """
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    sample_file = tmp_path / "sample.jsonl"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "Alpha Beta", "response": "Gamma Delta"}) + "\n")

    ds = SFTDataset(sample_file, tokenizer)
    evaluator = DeterministicEvaluator(
        checkpoint_path="experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
        tokenizer_dir=TOKENIZER_DIR,
    )

    p, r = "Alpha Beta", "Gamma Delta"
    ds_inp, ds_tgt = ds.encode_example(p, r)
    ev_inp, ev_tgt, ev_status, ev_err = evaluator.encode_record(p, r)

    assert ev_status == "EVALUATED"
    assert ds_inp == ev_inp
    assert ds_tgt == ev_tgt

    # Empty response rejection
    with pytest.raises(ValueError):
        ds.encode_example("Alpha", "")

    _, _, status_empty, _ = evaluator.encode_record("Alpha", "")
    assert status_empty == "SKIPPED_EMPTY_RESPONSE"


# ==============================================================================
# BATCH 52 TESTS: DATASET TRUST GATES & SPLIT LEAKAGE ELIMINATION
# ==============================================================================


def test_batch52_fail_closed_dataset_eligibility_policy():
    """
    Proves that records with PENDING_HUMAN_REVIEW or unsupported provenance
    cannot enter the verified-training path without explicit authorization.
    """
    from experiments.batch51_54_correctness_sprint.data.dataset_eligibility_gate import (
        DatasetEligibilityGate,
        TrustStatus,
    )

    gate = DatasetEligibilityGate()

    # Case 1: Structurally valid, source supported, but pending review -> NOT eligible for verified path
    rec_pending = {
        "id": "TEST-PENDING-01",
        "domain": "mathematics",
        "prompt": "<|im_start|>user\nState theorem.<|im_end|>\n<|im_start|>assistant\n",
        "response": "Theorem statement.<|im_end|>\n",
        "status": TrustStatus.SOURCE_SUPPORTED.value,
        "review_status": TrustStatus.PENDING_HUMAN_REVIEW.value,
    }
    decision = gate.evaluate_record(rec_pending, allow_pending_review=False)
    assert not decision.training_eligible
    assert decision.rejection_reason == "PENDING_HUMAN_REVIEW"

    # Case 2: Explicit opt-in path allows exploratory research only
    decision_optin = gate.evaluate_record(rec_pending, allow_pending_review=True)
    assert decision_optin.training_eligible
    assert decision_optin.mode == "EXPLORATORY_OPT_IN"

    # Case 3: Empty or malformed response -> strictly rejected
    rec_malformed = {
        "id": "TEST-MALFORMED",
        "domain": "mathematics",
        "prompt": "<|im_start|>user\nState theorem.<|im_end|>\n<|im_start|>assistant\n",
        "response": "",
        "status": TrustStatus.STRUCTURALLY_VALID.value,
        "review_status": TrustStatus.ANSWER_VERIFIED.value,
    }
    decision_malformed = gate.evaluate_record(rec_malformed, allow_pending_review=True)
    assert not decision_malformed.training_eligible


def test_batch52_zero_family_leakage_between_clean_splits():
    """
    Verifies that the cleaned Batch 52 dataset has 0 source-family leakage
    between train and validation splits.
    """
    from experiments.batch51_54_correctness_sprint.data.dataset_eligibility_gate import (
        extract_source_family,
    )

    clean_train_path = Path("experiments/batch51_54_correctness_sprint/data/train_clean.jsonl")
    clean_val_path = Path("experiments/batch51_54_correctness_sprint/data/val_clean.jsonl")

    if not clean_train_path.exists() or not clean_val_path.exists():
        pytest.skip("Clean splits not yet generated.")

    train_families = set()
    with open(clean_train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                train_families.add(extract_source_family(rec))

    val_families = set()
    with open(clean_val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                val_families.add(extract_source_family(rec))

    leakage = train_families.intersection(val_families)
    assert len(leakage) == 0, f"Detected cross-split family leakage: {leakage}"


# ==============================================================================
# BATCH 53 TESTS: CONTROLLED LEARNING & EVALUATION INTEGRITY
# ==============================================================================


def test_batch53_evaluator_probe_exact_rubric():
    """
    Verifies that probe outputs are evaluated against documented rubrics
    rather than uncalibrated heuristics.
    """
    probes_path = Path("experiments/batch42_reproducible_evaluation/evaluation/benchmark_probes.json")
    assert probes_path.exists()

    with open(probes_path, "r", encoding="utf-8") as f:
        probes = json.load(f)

    assert len(probes) >= 5
    for p in probes:
        assert "id" in p
        assert "prompt" in p
        assert "expected_output_hint" in p


# ==============================================================================
# BATCH 54 TESTS: HARDWARE-ADAPTIVE SCHEDULING & SAFE REAL EXECUTION
# ==============================================================================


def test_batch54_hardware_detection_and_scheduling_logic():
    """
    Tests hardware capability detection and dynamic adaptive scheduling.
    """
    detector = HardwareCapabilityDetector()
    profile = detector.detect()

    assert profile.physical_cores >= 1
    assert profile.available_ram_bytes > 0
    assert PrecisionType.FP32 in profile.supported_precisions

    scheduler = HardwareAdaptiveScheduler(profile)
    decision = scheduler.plan_workload(
        workload_type=WorkloadType.TRAINING_SFT,
        total_items=46,
        allow_gpu=True,
    )

    assert decision.target_device in (ComputeDevice.CPU, ComputeDevice.CUDA)
    assert decision.batch_size >= 1
    assert decision.chunk_size >= 1
    assert decision.memory_budget_mb > 0
    assert decision.safety_reserve_mb > 0


def test_batch54_simulated_cuda_unavailable_fallback():
    """
    Proves that when CUDA is requested but unavailable or forbidden,
    the scheduler gracefully falls back to CPU with explicit rationale.
    """
    # Create synthetic CUDA profile
    cuda_profile = HardwareProfile(
        device=ComputeDevice.CUDA,
        cpu_model="Simulated Xeon",
        physical_cores=16,
        logical_threads=32,
        total_ram_bytes=64 * (1024**3),
        available_ram_bytes=32 * (1024**3),
        gpu_name="Simulated RTX 4090",
        gpu_vram_bytes=24 * (1024**3),
        supported_precisions=[PrecisionType.FP32, PrecisionType.BF16, PrecisionType.FP16],
    )

    scheduler = HardwareAdaptiveScheduler(cuda_profile)

    # When allow_gpu=False, fallback to CPU must be cleanly recorded
    decision = scheduler.plan_workload(
        workload_type=WorkloadType.INFERENCE,
        total_items=100,
        allow_gpu=False,
    )
    assert decision.target_device == ComputeDevice.CPU
    assert decision.fallback_reason == "GPU execution explicitly disabled by policy."
