"""
Comprehensive regression test suite for Batches 43-46.
Covers:
- Batch 43: Evaluation integrity, boundary continuity, and token-weighted aggregation.
- Batch 44: Generation diagnosis, memorization overfit test, and EOS emission verification.
- Batch 45: Grounded multidomain extraction pipeline, schema validity, and provenance digests.
- Batch 46: Hardware-adaptive scheduler, adaptive chunking, memory reserves, and simulated CUDA fallback.
"""

import hashlib
import json
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.hardware import ComputeDevice, HardwareProfile, PrecisionType
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import (
    DeterministicEvaluator,
    compute_file_sha256,
)
from experiments.batch43_46_architecture_capability_sprint.data_pipeline.domain_pipeline import (
    CodingExample,
    MathematicsExample,
    MultidomainCorpusExtractor,
    ScienceReasoningExample,
)
from experiments.batch43_46_architecture_capability_sprint.hardware_readiness.scheduler import (
    AdaptiveScheduleDecision,
    HardwareAdaptiveScheduler,
    WorkloadType,
)

EXPECTED_CANONICAL_HASH = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
EXPECTED_CANONICAL_PARAMS = 3_443_136
EXPECTED_STEP10000_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"


# --- BATCH 43 TESTS: EVALUATION INTEGRITY ---


def test_batch43_canonical_invariants():
    """Confirms canonical baseline initialization and step 10k checkpoint SHA-256."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig())
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == EXPECTED_CANONICAL_PARAMS

    hasher = hashlib.sha256()
    with torch.no_grad():
        for name, param in sorted(model.named_parameters()):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
    assert hasher.hexdigest() == EXPECTED_CANONICAL_HASH

    base_ckpt = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    assert base_ckpt.exists()
    assert compute_file_sha256(base_ckpt) == EXPECTED_STEP10000_HASH


def test_batch43_evaluator_unstripped_boundary_match():
    """Confirms that encode_record maintains proper token continuity without boundary truncation."""
    base_ckpt = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    if not base_ckpt.exists():
        pytest.skip("Base checkpoint missing")

    evaluator = DeterministicEvaluator(base_ckpt)
    prompt = "<|im_start|>user\nWhat is 2 + 3?<|im_end|>\n<|im_start|>assistant\n"
    response = "5<|im_end|>\n"

    inp, tgt, status, err = evaluator.encode_record(prompt, response)
    assert status == "EVALUATED"
    assert err is None
    # Check BOS
    assert inp[0] == 0
    # Check target ending with EOS
    assert tgt[-1] == 1


# --- BATCH 44 TESTS: GENERATION RECOVERY & EOS EMISSION ---


def test_batch44_targeted_eos_emission():
    """Verifies that the ChakrMicro architecture can emit EOS deterministically after targeted gradient steps."""
    torch.manual_seed(42)
    model = ChakrMicro(ModelConfig(max_seq_len=512))

    # In causal LM:
    # input:  [BOS] + p_toks + r_toks
    # target: [-100]*len(p_toks) + r_toks + [EOS]
    # Length of inp must match length of tgt!
    p_toks = [63, 127, 1012, 98, 427, 13]  # mock prompt without BOS
    r_toks = [56, 63, 127, 1012, 98, 275, 13]  # mock response
    inp = [0] + p_toks + r_toks
    tgt = ([-100] * len(p_toks)) + r_toks + [1]
    assert len(inp) == len(tgt)

    x = torch.tensor([inp], dtype=torch.long)
    y = torch.tensor([tgt], dtype=torch.long)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for _ in range(25):
        optimizer.zero_grad()
        loss = loss_fn(model(x).view(-1, 4096), y.view(-1))
        loss.backward()
        optimizer.step()

    model.eval()
    generated = [0] + p_toks
    with torch.no_grad():
        for _ in range(15):
            logits = model(torch.tensor([generated], dtype=torch.long))
            next_t = torch.argmax(logits[0, -1, :]).item()
            generated.append(next_t)
            if next_t == 1:
                break

    assert generated[-1] == 1, "Model failed to emit EOS token under targeted supervision"


# --- BATCH 45 TESTS: MULTIDOMAIN DATA PIPELINE ---


def test_batch45_multidomain_extraction_and_provenance():
    """Verifies that MultidomainCorpusExtractor extracts mathematics, coding, and reasoning records with SHA-256."""
    extractor = MultidomainCorpusExtractor(".")
    math_recs = extractor.extract_mathematics_theorems()
    code_recs = extractor.extract_coding_algorithms()
    rsn_recs = extractor.extract_science_reasoning_protocols()

    assert len(math_recs) == 15
    assert len(code_recs) == 12
    assert len(rsn_recs) == 5

    # Check schema and provenance
    for r in math_recs:
        assert isinstance(r, MathematicsExample)
        assert len(r.source_sha256) == 64
        assert r.domain == "mathematics"
        assert r.review_status == "PENDING_HUMAN_REVIEW"
        chat = r.to_chat_format()
        assert "<|im_start|>user" in chat["prompt"]
        assert "<|im_end|>\n" in chat["response"]

    for c in code_recs:
        assert isinstance(c, CodingExample)
        assert len(c.source_sha256) == 64
        assert c.domain == "code"

    for s in rsn_recs:
        assert isinstance(s, ScienceReasoningExample)
        assert len(s.source_sha256) == 64
        assert s.domain == "reasoning"


# --- BATCH 46 TESTS: HARDWARE-ADAPTIVE SCHEDULING ---


def test_batch46_adaptive_scheduler_cpu_and_simulated_gpu():
    """Verifies that HardwareAdaptiveScheduler plans appropriate memory reserves, batch sizes, and chunk sizes."""
    # 1. Test CPU Host Profile
    cpu_profile = HardwareProfile(
        device=ComputeDevice.CPU,
        cpu_model="Mock CPU",
        physical_cores=8,
        logical_threads=16,
        total_ram_bytes=32 * (1024**3),
        available_ram_bytes=16 * (1024**3),
        supported_precisions=[PrecisionType.FP32],
    )
    scheduler_cpu = HardwareAdaptiveScheduler(profile=cpu_profile)
    decision_cpu = scheduler_cpu.plan_workload(WorkloadType.TRAINING_SFT, total_items=50)

    assert decision_cpu.target_device == ComputeDevice.CPU
    assert decision_cpu.batch_size == 2
    assert decision_cpu.chunk_size == 50
    assert decision_cpu.safety_reserve_mb > 1000

    # 2. Test Simulated CUDA Host Profile
    cuda_profile = HardwareProfile(
        device=ComputeDevice.CUDA,
        cpu_model="Mock CPU",
        physical_cores=16,
        logical_threads=32,
        total_ram_bytes=64 * (1024**3),
        available_ram_bytes=32 * (1024**3),
        gpu_name="Mock NVIDIA A100",
        gpu_vram_bytes=40 * (1024**3),
        supported_precisions=[PrecisionType.FP32, PrecisionType.BF16, PrecisionType.FP16],
    )
    scheduler_cuda = HardwareAdaptiveScheduler(profile=cuda_profile)
    decision_cuda = scheduler_cuda.plan_workload(WorkloadType.TRAINING_SFT, total_items=200)

    assert decision_cuda.target_device == ComputeDevice.CUDA
    assert decision_cuda.precision in [PrecisionType.BF16, PrecisionType.FP16]
    assert decision_cuda.batch_size == 8
    assert decision_cuda.fallback_reason is None

    # 3. Test Disallowed GPU Policy Fallback
    decision_fallback = scheduler_cuda.plan_workload(WorkloadType.TRAINING_SFT, total_items=100, allow_gpu=False)
    assert decision_fallback.target_device == ComputeDevice.CPU
    assert "GPU execution explicitly disabled" in decision_fallback.fallback_reason
