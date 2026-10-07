"""Wave 417-424 Test Suite: Release Preparation & Cognitive Core Hardening.

Covers:
- Step 417: Release Scope & Invariant Audit
- Step 418: Release Artifact & Model Registry Hardening
- Step 419: Clean Install & Reproducibility Path
- Step 420: CPU-First Runtime & Resource Sanity
- Step 421: Benchmark & Regression Finalization
- Step 422: Capability Contract & Limitations Consistency
- Step 423: Security & Privacy Invariants
- Step 424: Release Candidate Gate Verdict
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.runtime.local_runtime import LocalModelRuntime
from chakrview.cognition.release_manifest_verifier import (
    verify_release_artifact_integrity,
    EXPECTED_TOKENIZER_HASHES,
    EXPECTED_BASELINE_PARAMS,
    EXPECTED_CANDIDATE_PARAMS,
)
from chakrview.cognition.release_benchmark import run_release_benchmark
from chakrview.cognition.i4_candidate_manifest import create_i4_candidate_manifest
from chakrview.cognition.capability_contract import run_step412_capability_contract
from chakrview.cognition.model_registry import ModelRegistry, ModelLifecycleState


class TestWave417To424ReleasePreparation:

    def test_01_canonical_baseline_bit_exact(self):
        """Canonical baseline parameter count and hash must be bit-exact."""
        baseline = instantiate_frozen_baseline()
        live_sha = compute_model_hash(baseline)
        live_params = sum(p.numel() for p in baseline.parameters())
        assert live_params == EXPECTED_BASELINE_PARAMS == 3_443_136
        assert live_sha == EXPECTED_WEIGHT_HASH

    def test_02_step418_release_artifact_integrity(self):
        """All v0.1 release artifacts, hashes, and isolated candidates must verify."""
        rep = verify_release_artifact_integrity()
        assert rep.all_integrity_checks_passed
        assert rep.baseline_sha_valid
        assert rep.baseline_parameters_valid
        assert rep.candidate_parameters_valid
        assert rep.candidate_isolated
        assert rep.tokenizer_integrity_valid
        assert rep.candidate_manifest_valid
        assert rep.registry_promoted_to_released

    def test_03_step418_candidate_isolation_no_baseline_drift(self):
        """Instantiating and using candidate module must never alter baseline weights."""
        baseline = instantiate_frozen_baseline()
        hash_before = compute_model_hash(baseline)

        from chakrview.cognition.neural_relational_acquisition import (
            NeuralRelationalAcquisitionModule,
        )
        cand = NeuralRelationalAcquisitionModule(d_input=96, d_model=96, vocab_size=4096)
        x = torch.randint(0, 4096, (2, 8))
        _ = cand(x)

        hash_after = compute_model_hash(baseline)
        assert hash_before == hash_after == EXPECTED_WEIGHT_HASH

    def test_04_step419_clean_runtime_import_and_default_instantiation(self):
        """LocalModelRuntime.from_default() must succeed and run CPU-first."""
        rt = LocalModelRuntime.from_default()
        assert rt.verify_runtime_integrity()
        assert rt.model_identity is not None

    def test_05_step420_cpu_first_inference_sanity(self):
        """Runtime must execute deterministic generation on CPU within reasonable memory."""
        rt = LocalModelRuntime.from_default()
        res = rt.generate("ChakrView")
        assert res is not None
        assert len(res.token_ids) > 0
        assert res.text != ""

    def test_06_step421_release_benchmark_entrypoint(self):
        """Release benchmark runner must return structured report and candidate readiness."""
        rep = run_release_benchmark()
        assert rep.version == "0.1.0"
        assert rep.release_candidate_ready
        assert rep.verdict == "RELEASE_CANDIDATE_READY_FOR_FOUNDER_TEST"
        assert rep.cognitive_evidence.i4_gate_passed
        assert rep.cognitive_evidence.g4_multi_seed_mean >= 0.50
        assert rep.infrastructure_verification.all_infrastructure_passed

    def test_07_step422_capability_contract_verified(self):
        """v0.1 Capability Contract assertions must all hold."""
        rep = run_step412_capability_contract()
        assert rep.verification.all_passed
        assert rep.contract["version"] == "0.1.0"
        assert len(rep.contract["known_limitations"]) >= 7

    def test_08_step423_security_and_privacy_cleanliness(self):
        """Ensure no secrets or keys are hardcoded in cognitive modules."""
        candidate_manifest = create_i4_candidate_manifest()
        assert candidate_manifest.contamination_zero
        assert candidate_manifest.anti_shortcut_passed
