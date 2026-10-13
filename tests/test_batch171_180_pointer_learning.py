"""
Tests for Batches 171-180: Pointer Learning, Diagnostics, and Cognitive Execution.
Validates:
1. Canonical model invariants (parameters, architecture, checkpoint SHA-256).
2. ChakrPointerHead probability normalization, causal masking, gate boundary constraints.
3. Safe CognitiveRouter routing with POINTER_COPY route toggle.
4. Specialized deterministic task engine workers (Retrieval, Verification, Math).
5. Multi-source task execution and process restart state recovery.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import torch
import pytest

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pointer_mechanism import ChakrPointerHead
from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType, ComponentOrigin
from chakrview.runtime.task_engine import (
    CognitiveTaskEngine,
    TaskStatus,
    RetrievalWorker,
    EvidenceVerificationWorker,
    MathVerifierWorker
)
from chakrview.runtime.persistent_memory import PersistentSessionMemory
from chakrview.runtime.retrieval_adapter import RetrievalAdapter

CANONICAL_CHECKPOINT_PATH = "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"
EXPECTED_CHECKPOINT_SHA256 = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"


def test_canonical_invariants():
    """Verify that canonical model architecture, checkpoint, and parameter count are pristine."""
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 3443136, f"Expected 3,443,136 parameters, got {total_params}"

    # Verify checkpoint integrity
    assert os.path.exists(CANONICAL_CHECKPOINT_PATH), "Canonical checkpoint must exist"
    with open(CANONICAL_CHECKPOINT_PATH, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    assert file_hash == EXPECTED_CHECKPOINT_SHA256, "Canonical checkpoint SHA-256 mismatch!"


def test_pointer_head_normalization_and_masking():
    """Verify that ChakrPointerHead normalizes probabilities and strictly masks disallowed positions."""
    B, T, D = 2, 8, 192
    V = 4096
    head = ChakrPointerHead(d_model=D, vocab_size=V, d_proj=64)

    h = torch.randn(B, T, D)
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    # Context mask: only positions 0, 1, 2 are valid context
    context_mask = torch.zeros(B, T, dtype=torch.bool)
    context_mask[:, :3] = True

    final_probs, attn, gate = head(h, input_ids, logits, context_mask)

    # 1. Output shapes
    assert final_probs.shape == (B, T, V)
    assert attn.shape == (B, T, T)
    assert gate.shape == (B, T, 1)

    # 2. Probability normalization across vocabulary
    sum_probs = torch.sum(final_probs, dim=-1)
    assert torch.allclose(sum_probs, torch.ones_like(sum_probs), atol=1e-5), "Probabilities must sum to 1"

    # 3. Gate boundaries: gate must be strictly in [0, 1]
    assert (gate >= 0.0).all() and (gate <= 1.0).all(), "Gate values must be within [0, 1]"

    # 4. Context masking: attention to non-context positions (>2) must be zero
    assert (attn[:, :, 3:] == 0.0).all(), "Attention weights on non-context tokens must be zero"


def test_cognitive_router_pointer_flag():
    """Verify that POINTER_COPY route only activates when explicitly enabled."""
    # Disabled by default
    router_default = CognitiveRouter(enable_pointer_copy=False)
    dec1 = router_default.classify_route("What is the atomic number of Gold?", supplied_context="Gold has atomic number 79.")
    assert dec1.route == RouteType.SLOT_EXTRACTION, "Default router must use SLOT_EXTRACTION"

    # Enabled
    router_pointer = CognitiveRouter(enable_pointer_copy=True)
    dec2 = router_pointer.classify_route("What is the atomic number of Gold?", supplied_context="Gold has atomic number 79.")
    assert dec2.route == RouteType.POINTER_COPY, "Pointer router must use POINTER_COPY when enabled"


def test_specialized_workers_verification():
    """Verify deterministic behavior of specialized workers."""
    # MathVerifierWorker
    assert MathVerifierWorker.verify_numeric_fact("42", "integer") is True
    assert MathVerifierWorker.verify_numeric_fact("abc", "integer") is False
    assert MathVerifierWorker.verify_numeric_fact("3.1415", "float") is True
    assert MathVerifierWorker.verify_numeric_fact("xyz", "float") is False

    # EvidenceVerificationWorker
    sources = [
        "Lithium is an alkali metal with atomic number 3.",
        "The element with atomic number 3 is Lithium, a soft reactive metal."
    ]
    res_cons = EvidenceVerificationWorker.verify_consistency(sources, "3")
    assert res_cons["verified"] is True
    assert res_cons["consensus_ratio"] == 1.0

    res_fail = EvidenceVerificationWorker.verify_consistency(sources, "999")
    assert res_fail["verified"] is False


def test_task_engine_restart_recovery():
    """Verify task state durability and idempotent execution across simulated restarts."""
    temp_dir = Path(tempfile.mkdtemp(prefix="chakr_test_task_"))
    try:
        mem = PersistentSessionMemory(temp_dir / "sessions")
        engine1 = CognitiveTaskEngine(storage_dir=temp_dir / "tasks", memory=mem)

        task_id = "test_persistence_1"
        sources = [
            "Hydrogen is the lightest element with atomic number 1.",
            "Atomic number of Hydrogen is 1."
        ]
        state1 = engine1.execute_multi_source_verified_workflow(
            task_id=task_id,
            session_id="test_session",
            query="What is the atomic number of Hydrogen?",
            sources=sources,
        )
        assert state1.status == TaskStatus.COMPLETED
        assert state1.final_result == "1"

        # Simulate restart: New engine instance from same storage dir
        engine2 = CognitiveTaskEngine(storage_dir=temp_dir / "tasks", memory=mem)
        loaded_state = engine2.load_task(task_id)
        assert loaded_state is not None
        assert loaded_state.status == TaskStatus.COMPLETED
        assert loaded_state.final_result == "1"

        # Idempotent re-execution returns completed state without re-running
        state2 = engine2.execute_multi_source_verified_workflow(
            task_id=task_id,
            session_id="test_session",
            query="What is the atomic number of Hydrogen?",
            sources=sources,
        )
        assert state2.status == TaskStatus.COMPLETED
        assert state2.final_result == "1"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
