"""ChakrView v0.1 Authoritative Release Manifest.

Wave 434: Single source of truth for the v0.1 release candidate:
"ChakrView v0.1 — Verified Cognitive Core & Relational Acquisition Prototype"

Clearly separates:
1. Canonical Baseline: ChakrMicro (3,443,136 parameters, SHA c5571c...00a282da)
2. I4 Candidate: NeuralRelationalAcquisitionModule (69,809 parameters)
3. Tokenizer: BPE 4,096 vocab (hashes verified)
4. Empirical Evidence: I4 benchmark metrics (G4=77.78%, H1=100%, H2=88.89%)
5. Scope & Limitations: Open-ended conversational chat is NOT claimed or supported.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.cognition.neural_relational_acquisition import (
    inspect_relational_acquisition_module,
    NeuralRelationalAcquisitionModule,
)
from chakrview.cognition.release_manifest_verifier import (
    verify_release_artifact_integrity,
    EXPECTED_TOKENIZER_HASHES,
    EXPECTED_BASELINE_PARAMS,
    EXPECTED_CANDIDATE_PARAMS,
)


@dataclasses.dataclass(frozen=True)
class ChakrViewReleaseManifest:
    release_version: str
    release_title: str
    release_codename: str
    release_status: str

    # Canonical Baseline
    baseline_model_name: str
    baseline_parameter_count: int
    baseline_weight_sha256: str
    baseline_architecture: str

    # I4 Candidate
    candidate_id: str
    candidate_module_name: str
    candidate_trainable_parameters: int
    candidate_isolated: bool

    # Tokenizer
    tokenizer_type: str
    tokenizer_vocab_size: int
    tokenizer_artifact_hashes: Dict[str, str]

    # Benchmark Evidence
    benchmark_name: str
    benchmark_evidence: Dict[str, Any]

    # System & Runtime Guarantees
    execution_target: str
    max_memory_mb: int
    python_minimum_version: str

    # Scope & Boundary
    verified_capabilities: List[str]
    unverified_capabilities: List[str]
    explicit_limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def get_v0_1_release_manifest() -> ChakrViewReleaseManifest:
    """Returns the immutable, authoritative release manifest for ChakrView v0.1."""
    return ChakrViewReleaseManifest(
        release_version="0.1.0",
        release_title="ChakrView v0.1 — Verified Cognitive Core & Relational Acquisition Prototype",
        release_codename="ChakrView-I4-Core",
        release_status="RELEASE_CANDIDATE_PREPARED",
        baseline_model_name="ChakrMicro",
        baseline_parameter_count=EXPECTED_BASELINE_PARAMS,
        baseline_weight_sha256=EXPECTED_WEIGHT_HASH,
        baseline_architecture="Autoregressive Transformer (d_model=96, n_layer=6, n_head=6, vocab=4096)",
        candidate_id="chakrview-i4-wave408-v0.1",
        candidate_module_name="NeuralRelationalAcquisitionModule",
        candidate_trainable_parameters=EXPECTED_CANDIDATE_PARAMS,
        candidate_isolated=True,
        tokenizer_type="Byte-Pair Encoding (BPE)",
        tokenizer_vocab_size=4096,
        tokenizer_artifact_hashes=dict(EXPECTED_TOKENIZER_HASHES),
        benchmark_name="I4 Compositional Relational Binding Benchmark",
        benchmark_evidence={
            "milestone": "I4_COMPOSITIONAL_BINDING",
            "g4_multi_seed_mean": 0.7778,
            "g4_min_seed": 0.6667,
            "h1_routing_mean": 1.0000,
            "h2_routing_mean": 0.8889,
            "language_retention": 0.9500,
            "contamination": 0,
            "anti_shortcut_passed": True,
            "evaluation_seeds": [42, 101, 2026],
        },
        execution_target="CPU-first (no GPU or CUDA required)",
        max_memory_mb=512,
        python_minimum_version="3.10+",
        verified_capabilities=[
            "CPU-first local runtime and deterministic model loading",
            "Canonical baseline parameter count (3,443,136) and weight SHA bit-exactness",
            "BPE tokenizer encoding/decoding consistency with 4,096 vocabulary",
            "Candidate isolation (zero baseline weight drift: delta W = 0)",
            "I4 two-hop compositional relational binding on unseen, disjoint token pools",
            "Multi-seed Hop-1 (100%) and Hop-2 (88.89%) key routing",
            "Contamination-free held-out evaluation with anti-shortcut adversarial resilience",
        ],
        unverified_capabilities=[
            "Artificial General Intelligence (AGI)",
            "Open-ended conversational instruction following or chatbot capabilities",
            "Broad open-domain reasoning across general knowledge bases",
            "Autonomous self-improvement in the broad sense",
            "Proven long-horizon continual learning across catastrophic domain shifts",
            "Three-hop or deeper compositional reasoning (I5 remains in research)",
            "Universal hardware adaptation or distributed cluster orchestration",
        ],
        explicit_limitations=[
            "The default ChakrMicro baseline is a 3.44M-parameter research prototype; under default greedy decoding, unconstrained prompts trigger autoregressive suffix repetition loops terminating at MAX_TOKENS.",
            "Open-ended conversational chat is NOT a supported or claimed capability of v0.1.",
            "I4 relational capability operates through structured episode binding, not unconstrained open-domain dialogue.",
            "Relational module sequence length is currently bounded at 128 tokens.",
            "Language retention is evaluated at the 0.9500 floor; further unregularized fine-tuning may degrade baseline language modeling.",
        ],
    )
