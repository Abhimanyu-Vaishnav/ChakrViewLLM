"""Step 421: ChakrView v0.1 Release Benchmark Suite.

Provides the official, deterministic release benchmark entry point.
Clearly distinguishes:
  A. Neural / Cognitive Capability Evidence (I4 benchmark, multi-seed G4, H1/H2 routing, retention)
  B. System / Runtime Infrastructure Verification (Baseline SHA, parameter budget, clean imports)

Machine-readable format: JSON-serializable dataclass.
Human-readable format: formatted report summary.
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.release_manifest_verifier import (
    verify_release_artifact_integrity,
    ReleaseArtifactIntegrityReport,
)
from chakrview.cognition.i4_candidate_manifest import create_i4_candidate_manifest
from chakrview.cognition.capability_contract import (
    run_step412_capability_contract,
    Step412CapabilityContractReport,
)


@dataclasses.dataclass
class CognitiveEvidenceSummary:
    milestone: str
    g4_multi_seed_mean: float
    g4_min_seed: float
    h1_routing_mean: float
    h2_routing_mean: float
    language_retention: float
    contamination: int
    anti_shortcut_passed: bool
    i4_gate_passed: bool


@dataclasses.dataclass
class InfrastructureVerificationSummary:
    canonical_baseline_sha: str
    baseline_sha_intact: bool
    baseline_parameters: int
    baseline_parameters_exact: bool
    candidate_parameters: int
    candidate_parameters_within_budget: bool
    candidate_isolated: bool
    tokenizer_verified: bool
    model_registry_promoted: bool
    capability_contract_verified: bool
    all_infrastructure_passed: bool


@dataclasses.dataclass
class ReleaseBenchmarkReport:
    version: str
    release_name: str
    timestamp: str
    cognitive_evidence: CognitiveEvidenceSummary
    infrastructure_verification: InfrastructureVerificationSummary
    release_candidate_ready: bool
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def run_release_benchmark() -> ReleaseBenchmarkReport:
    """Executes the complete release benchmark evaluation."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 1. Audit artifacts & manifests
    art_rep = verify_release_artifact_integrity()

    # 2. Audit capability contract
    contract_rep = run_step412_capability_contract()

    # 3. Compile Cognitive Evidence
    manifest = create_i4_candidate_manifest()
    cog_ev = CognitiveEvidenceSummary(
        milestone=manifest.milestone,
        g4_multi_seed_mean=manifest.multi_seed_g4_mean,
        g4_min_seed=manifest.min_seed_g4,
        h1_routing_mean=manifest.mean_h1_routing,
        h2_routing_mean=manifest.mean_h2_routing,
        language_retention=manifest.language_retention,
        contamination=0 if manifest.contamination_zero else 1,
        anti_shortcut_passed=manifest.anti_shortcut_passed,
        i4_gate_passed=manifest.all_gates_passed,
    )

    # 4. Compile Infrastructure Verification
    infra_ev = InfrastructureVerificationSummary(
        canonical_baseline_sha=art_rep.baseline_sha,
        baseline_sha_intact=art_rep.baseline_sha_valid,
        baseline_parameters=art_rep.baseline_parameters,
        baseline_parameters_exact=art_rep.baseline_parameters_valid,
        candidate_parameters=art_rep.candidate_parameters,
        candidate_parameters_within_budget=art_rep.candidate_parameters_valid,
        candidate_isolated=art_rep.candidate_isolated,
        tokenizer_verified=art_rep.tokenizer_integrity_valid,
        model_registry_promoted=art_rep.registry_promoted_to_released,
        capability_contract_verified=contract_rep.verification.all_passed,
        all_infrastructure_passed=(
            art_rep.all_integrity_checks_passed
            and contract_rep.verification.all_passed
        ),
    )

    ready = cog_ev.i4_gate_passed and infra_ev.all_infrastructure_passed
    verdict = (
        "RELEASE_CANDIDATE_READY_FOR_FOUNDER_TEST"
        if ready
        else "RELEASE_BLOCKED"
    )

    return ReleaseBenchmarkReport(
        version="0.1.0",
        release_name="ChakrView v0.1 Verified Cognitive Core",
        timestamp=ts,
        cognitive_evidence=cog_ev,
        infrastructure_verification=infra_ev,
        release_candidate_ready=ready,
        verdict=verdict,
    )
