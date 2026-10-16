"""
Unit & Regression Tests for Batches 1501–1540:
Neuro-Symbolic Tool Routing, Verified Arithmetic & Structured Refutation.
Location: tests/test_batch1501_1540_neuro_symbolic_tools.py
"""
import hashlib
import json
from pathlib import Path
import pytest

from chakrview.runtime.safe_calculator import SafeCalculator
from chakrview.runtime.structured_verifier import (
    StructuredVerifier,
    StructuredClaim,
    VerificationVerdict,
)
from chakrview.runtime.governed_tool_router import (
    GovernedToolRouter,
    CanonicalRoute,
)
from chakrverify.core import DocumentChunk

REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH_DIR = REPO_ROOT / "experiments" / "batch1501_1540"
DATA_DIR = BATCH_DIR / "data"
RESULTS_DIR = BATCH_DIR / "results"

CANONICAL_CHECKPOINT = (
    REPO_ROOT
    / "experiments"
    / "batch16_21_combined"
    / "main_training_10000"
    / "training_checkpoint_step_10000.pt"
)
EXPECTED_CANONICAL_HASH = (
    "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
)

ADAPTED_CHECKPOINT = (
    REPO_ROOT
    / "experiments"
    / "batch1381_1420"
    / "checkpoints"
    / "chakrmicro_domain_adapted_experimental.pt"
)
EXPECTED_ADAPTED_HASH = (
    "15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139"
)


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_checkpoint_immutability():
    """Verify that both canonical and experimental checkpoint hashes remain unaltered."""
    assert CANONICAL_CHECKPOINT.exists()
    assert ADAPTED_CHECKPOINT.exists()

    canon_hash = _compute_sha256(CANONICAL_CHECKPOINT)
    adapt_hash = _compute_sha256(ADAPTED_CHECKPOINT)

    assert canon_hash == EXPECTED_CANONICAL_HASH
    assert adapt_hash == EXPECTED_ADAPTED_HASH


def test_safe_calculator_core_and_security():
    """Test SafeCalculator operations and strict security sandboxing."""
    calc = SafeCalculator()

    # Elementary operations
    assert calc.evaluate("147 + 258").result == 405
    assert calc.evaluate("1024 - 512").result == 512
    assert calc.evaluate("50 * 20").result == 1000
    assert calc.evaluate("100 / 4").result == 25
    assert calc.evaluate("17 % 5").result == 2
    assert calc.evaluate("2 ** 10").result == 1024
    assert calc.evaluate("(50 + 50) / 10").result == 10

    # Domain violations
    res_div0 = calc.evaluate("10 / 0")
    assert not res_div0.success
    assert res_div0.error_code == "DIVISION_BY_ZERO"

    # Security attacks
    assert not calc.evaluate("__import__('os').system('dir')").success
    assert not calc.evaluate("eval('2 + 2')").success
    assert not calc.evaluate("open('secret.txt')").success
    assert not calc.evaluate("2 ** 500").success
    assert not calc.evaluate("True + 1").success


def test_structured_verifier_verdict_separation():
    """Verify that StructuredVerifier distinguishes SUPPORTED, CONTRADICTED, and INSUFFICIENT."""
    chunks = [
        DocumentChunk(
            doc_id="storage_bm25_index.md",
            chunk_index=0,
            text="# BM25 Hyperparameters\n- Term saturation parameter k1: 1.5.\n- Document length normalization parameter b: 0.75.",
            start_char=0,
            end_char=100,
        )
    ]
    verifier = StructuredVerifier(chunks=chunks)

    # 1. Supported
    rec_supp = verifier.verify_claim(
        StructuredClaim(
            claim_id="c1",
            subject_entity="BM25 Hyperparameters",
            attribute_name="k1 parameter",
            claimed_value="1.5",
        )
    )
    assert rec_supp.verdict == VerificationVerdict.SUPPORTED

    # 2. Contradicted
    rec_contra = verifier.verify_claim(
        StructuredClaim(
            claim_id="c2",
            subject_entity="BM25 Hyperparameters",
            attribute_name="k1 parameter",
            claimed_value="2.5",
        )
    )
    assert rec_contra.verdict == VerificationVerdict.CONTRADICTED

    # 3. Insufficient Evidence (absence of evidence != contradiction)
    rec_insuff = verifier.verify_claim(
        StructuredClaim(
            claim_id="c3",
            subject_entity="Quantum Cryptography Core",
            attribute_name="key length",
            claimed_value="4096",
        )
    )
    assert rec_insuff.verdict == VerificationVerdict.INSUFFICIENT_EVIDENCE

    # 4. Invalid Claim
    rec_invalid = verifier.verify_claim(
        StructuredClaim(
            claim_id="c4",
            subject_entity="",
            attribute_name="port",
            claimed_value="80",
        )
    )
    assert rec_invalid.verdict == VerificationVerdict.INVALID_CLAIM


def test_governed_tool_router_dispatch():
    """Test router request classification across canonical routes."""
    calc = SafeCalculator()
    router = GovernedToolRouter(calculator=calc)

    # Greeting -> DIRECT_RESPONSE
    r_chat = router.route_request("Hello assistant")
    assert r_chat.route == CanonicalRoute.DIRECT_RESPONSE
    assert r_chat.component_origin == "neural_core"

    # Arithmetic -> CALCULATOR
    r_math = router.route_request("Calculate: 147 + 258")
    assert r_math.route == CanonicalRoute.CALCULATOR
    assert r_math.component_origin == "calculator_tool"
    assert r_math.final_response == "405"

    # Out-of-Domain -> ABSTAIN
    r_abst = router.route_request("What is the capital city of planet Mars?")
    assert r_abst.route == CanonicalRoute.ABSTAIN
    assert r_abst.component_origin == "fallback_policy"


def test_frozen_300_suite_integrity_and_results():
    """Verify evaluation suite manifest hash and results summary."""
    suite_file = DATA_DIR / "evaluation_suite_300.json"
    manifest_file = DATA_DIR / "EVALUATION_SUITE_MANIFEST.json"
    results_file = RESULTS_DIR / "BENCHMARK_RESULTS_300.json"

    assert suite_file.exists()
    assert manifest_file.exists()
    assert results_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(suite_file, "r", encoding="utf-8") as f:
        suite = json.load(f)

    assert len(suite) == 300
    assert manifest["total_tasks"] == 300
    assert _compute_sha256(suite_file) == manifest["sha256"]

    with open(results_file, "r", encoding="utf-8") as f:
        res = json.load(f)

    summary = res["summary"]
    assert summary["track1_arithmetic_accuracy"] >= 0.98
    assert summary["track2_verification_accuracy"] >= 0.90
    assert summary["track3_routing_accuracy"] >= 0.92
    assert summary["track4_e2e_accuracy"] >= 0.80
    assert summary["overall_success_rate"] >= 0.90
