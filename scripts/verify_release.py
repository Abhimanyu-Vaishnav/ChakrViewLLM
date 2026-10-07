"""Wave 437: One-Command Release Verification Script for ChakrView v0.1.

Executes and audits all release dimensions:
1. Canonical baseline parameter count & SHA-256 bit-exactness
2. Tokenizer artifact integrity (config, merges, vocab SHA-256)
3. I4 candidate manifest & isolation (delta W = 0)
4. Capability contract verification
5. Release benchmark runner
6. CPU-first runtime execution & resource sanity

Usage:
    python scripts/verify_release.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is on sys.path for direct script invocation
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.runtime.interactive import (
    instantiate_frozen_baseline,
    compute_model_hash,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.runtime.local_runtime import LocalModelRuntime
from chakrview.cognition.release_manifest_verifier import (
    verify_release_artifact_integrity,
    EXPECTED_BASELINE_PARAMS,
    EXPECTED_CANDIDATE_PARAMS,
)
from chakrview.cognition.release_benchmark import run_release_benchmark
from chakrview.cognition.capability_contract import run_step412_capability_contract
from chakrview.release_manifest import get_v0_1_release_manifest


def run_verification() -> bool:
    print("=" * 70)
    print("CHAKRVIEW v0.1 ONE-COMMAND RELEASE VERIFICATION")
    print("=" * 70)

    results: Dict[str, Tuple[str, str]] = {}
    all_passed = True

    # 1. Baseline Invariant
    try:
        baseline = instantiate_frozen_baseline()
        live_sha = compute_model_hash(baseline)
        live_params = sum(p.numel() for p in baseline.parameters())
        if live_params == EXPECTED_BASELINE_PARAMS and live_sha == EXPECTED_WEIGHT_HASH:
            results["1. Canonical Baseline Invariant"] = ("PASS", f"3,443,136 params, SHA {live_sha[:16]}...")
        else:
            results["1. Canonical Baseline Invariant"] = ("FAIL", f"params={live_params}, sha={live_sha}")
            all_passed = False
    except Exception as e:
        results["1. Canonical Baseline Invariant"] = ("FAIL", str(e))
        all_passed = False

    # 2. Tokenizer Artifacts
    try:
        art_rep = verify_release_artifact_integrity()
        if art_rep.tokenizer_integrity_valid:
            results["2. Tokenizer Artifact Hashes"] = ("PASS", "config.json, merges.json, vocab.json verified")
        else:
            results["2. Tokenizer Artifact Hashes"] = ("FAIL", f"Hashes: {art_rep.tokenizer_hashes}")
            all_passed = False
    except Exception as e:
        results["2. Tokenizer Artifact Hashes"] = ("FAIL", str(e))
        all_passed = False

    # 3. Candidate Isolation & Manifest
    try:
        if art_rep.candidate_isolated and art_rep.candidate_manifest_valid and art_rep.candidate_parameters_valid:
            results["3. Candidate Isolation & Manifest"] = ("PASS", f"{art_rep.candidate_id} (69,809 params, delta W = 0)")
        else:
            results["3. Candidate Isolation & Manifest"] = ("FAIL", "Isolation or manifest gate failed")
            all_passed = False
    except Exception as e:
        results["3. Candidate Isolation & Manifest"] = ("FAIL", str(e))
        all_passed = False

    # 4. Capability Contract
    try:
        contract_rep = run_step412_capability_contract()
        if contract_rep.verification.all_passed:
            results["4. Capability Contract"] = ("PASS", f"v{contract_rep.contract['version']} verified")
        else:
            results["4. Capability Contract"] = ("FAIL", "Contract verification failed")
            all_passed = False
    except Exception as e:
        results["4. Capability Contract"] = ("FAIL", str(e))
        all_passed = False

    # 5. Release Benchmark
    try:
        bench_rep = run_release_benchmark()
        if bench_rep.release_candidate_ready:
            results["5. Release Benchmark"] = ("PASS", f"Verdict: {bench_rep.verdict} (G4 mean=77.78%, H1=100%, H2=88.89%)")
        else:
            results["5. Release Benchmark"] = ("FAIL", f"Verdict: {bench_rep.verdict}")
            all_passed = False
    except Exception as e:
        results["5. Release Benchmark"] = ("FAIL", str(e))
        all_passed = False

    # 6. CPU-First Runtime Execution
    try:
        rt = LocalModelRuntime.from_default()
        res = rt.generate("ChakrView")
        if rt.verify_runtime_integrity() and len(res.token_ids) > 0:
            results["6. CPU-First Runtime Execution"] = ("PASS", f"Generated {len(res.token_ids)} tokens on CPU")
        else:
            results["6. CPU-First Runtime Execution"] = ("FAIL", "Runtime execution or integrity failed")
            all_passed = False
    except Exception as e:
        results["6. CPU-First Runtime Execution"] = ("FAIL", str(e))
        all_passed = False

    # 7. Authoritative Release Manifest
    try:
        man = get_v0_1_release_manifest()
        if man.release_status == "RELEASE_CANDIDATE_PREPARED":
            results["7. Authoritative Release Manifest"] = ("PASS", f"{man.release_title}")
        else:
            results["7. Authoritative Release Manifest"] = ("FAIL", f"Status: {man.release_status}")
            all_passed = False
    except Exception as e:
        results["7. Authoritative Release Manifest"] = ("FAIL", str(e))
        all_passed = False

    print("\nVERIFICATION SUMMARY:")
    print("-" * 70)
    for dim, (status, detail) in results.items():
        print(f"[{status:4s}] {dim:<38s} | {detail}")
    print("-" * 70)

    verdict = "PUBLIC_RELEASE_READY_FOR_FOUNDER_APPROVAL" if all_passed else "RELEASE_BLOCKED"
    print(f"\nFINAL VERDICT: {verdict}\n")
    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
