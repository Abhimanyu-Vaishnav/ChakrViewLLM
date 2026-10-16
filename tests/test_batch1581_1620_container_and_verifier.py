"""
Unit and Regression Tests for Batches 1581–1620:
Container Isolation, Dependency-Aware Verification & Failure-Driven Validation.
Location: tests/test_batch1581_1620_container_and_verifier.py
"""
import pytest
import os
import tempfile
from pathlib import Path

from chakrview.runtime.oci_container_backend import (
    OciContainerBackend,
    ContainerExecutionRequest,
    ContainerExecutionStatus,
)
from chakrview.runtime.structured_verifier import (
    StructuredVerifier,
    StructuredClaim,
    VerificationVerdict,
)
from chakrview.runtime.governed_tool_router import (
    GovernedToolRouter,
    CanonicalRoute,
)
from chakrview.runtime.conversation_state import (
    ConversationStateManager,
    ToolResultType,
)
from chakrverify.core import DocumentChunk


# =============================================================================
# 1. Container Runtime Discovery & Fail-Closed Tests
# =============================================================================

def test_container_runtime_discovery_fail_closed():
    backend = OciContainerBackend()
    # On Windows without docker/podman, runtime should be unavailable
    if not backend.is_available:
        req = ContainerExecutionRequest(
            task_id="task_fail_closed",
            code="print('Hello')",
        )
        res = backend.execute(req)
        assert res.status == ContainerExecutionStatus.RUNTIME_UNAVAILABLE
        assert res.exit_code is None
        assert "not installed or available" in res.error


def test_container_rejects_unapproved_image():
    backend = OciContainerBackend()
    req = ContainerExecutionRequest(
        task_id="task_bad_img",
        code="print('Hello')",
        image="ubuntu:latest",
    )
    res = backend.execute(req)
    assert res.status == ContainerExecutionStatus.UNSAFE_CONFIG_REJECTED
    assert "Unapproved container image" in res.error


def test_container_rejects_custom_host_mounts():
    backend = OciContainerBackend()
    req = ContainerExecutionRequest(
        task_id="task_bad_mount",
        code="print('Hello')",
        mount_paths=["/etc", "/var/run/docker.sock"],
    )
    res = backend.execute(req)
    assert res.status == ContainerExecutionStatus.UNSAFE_CONFIG_REJECTED
    assert "Unsafe mount paths rejected" in res.error


def test_container_rejects_network_enabled():
    backend = OciContainerBackend()
    req = ContainerExecutionRequest(
        task_id="task_bad_net",
        code="print('Hello')",
        network_enabled=True,
    )
    res = backend.execute(req)
    assert res.status == ContainerExecutionStatus.UNSAFE_CONFIG_REJECTED
    assert "Network access is prohibited" in res.error


def test_container_rejects_invalid_memory_limit():
    backend = OciContainerBackend()
    req = ContainerExecutionRequest(
        task_id="task_bad_mem",
        code="print('Hello')",
        memory_limit="1024g",
    )
    res = backend.execute(req)
    assert res.status == ContainerExecutionStatus.UNSAFE_CONFIG_REJECTED
    assert "Invalid memory limit" in res.error


# =============================================================================
# 2. Dependency-Aware Structured Verification Regression Tests
# =============================================================================

def test_verifier_multi_entity_pricing_isolation():
    """Verify that Starter Tier ($150) and Professional Tier ($450) do not cross-talk."""
    chunk = DocumentChunk(
        doc_id="billing_subscription_tiers.md",
        chunk_index=0,
        text=(
            "# Enterprise Subscription Pricing & Tiers\n\n"
            "## 1. Tier Pricing Matrix\n"
            "- Starter Tier: $150 per month, includes up to 2 concurrent worker processes.\n"
            "- Professional Tier: $450 per month, includes up to 8 concurrent worker processes and priority telemetry.\n"
        ),
        start_char=0,
        end_char=200,
    )
    verifier = StructuredVerifier(chunks=[chunk])

    # 1. Professional Tier = $450 -> SUPPORTED
    res1 = verifier.verify_claim(StructuredClaim(
        claim_id="test_prop_202",
        subject_entity="Subscription Pricing",
        attribute_name="Professional Tier",
        claimed_value="$450",
    ))
    assert res1.verdict == VerificationVerdict.SUPPORTED
    assert "450" in str(res1.evidence_value)

    # 2. Professional Tier = $150 -> CONTRADICTED
    res2 = verifier.verify_claim(StructuredClaim(
        claim_id="test_prop_217",
        subject_entity="Subscription Pricing",
        attribute_name="Professional Tier",
        claimed_value="$150",
    ))
    assert res2.verdict == VerificationVerdict.CONTRADICTED


def test_verifier_multi_parameter_symbol_isolation():
    """Verify that k1 parameter (1.5) and b parameter (0.75) do not cross-talk."""
    chunk = DocumentChunk(
        doc_id="storage_bm25_index.md",
        chunk_index=0,
        text=(
            "# BM25 Lexical Knowledge Index\n\n"
            "## 1. BM25 Hyperparameters\n"
            "- Term saturation parameter k1: 1.5.\n"
            "- Document length normalization parameter b: 0.75.\n"
        ),
        start_char=0,
        end_char=200,
    )
    verifier = StructuredVerifier(chunks=[chunk])

    # b parameter = 0.75 -> SUPPORTED
    res1 = verifier.verify_claim(StructuredClaim(
        claim_id="test_prop_204",
        subject_entity="BM25 Hyperparameters",
        attribute_name="b parameter",
        claimed_value="0.75",
    ))
    assert res1.verdict == VerificationVerdict.SUPPORTED

    # b parameter = 1.5 -> CONTRADICTED
    res2 = verifier.verify_claim(StructuredClaim(
        claim_id="test_prop_219",
        subject_entity="BM25 Hyperparameters",
        attribute_name="b parameter",
        claimed_value="1.5",
    ))
    assert res2.verdict == VerificationVerdict.CONTRADICTED


def test_verifier_prepositional_timeout_clause():
    """Verify that drain timeout (60 seconds) is extracted properly without bleed."""
    chunk = DocumentChunk(
        doc_id="sop_cluster_drain.md",
        chunk_index=0,
        text=(
            "# Node Maintenance and Drain Protocol\n\n"
            "## 1. Drain Sequencing\n"
            "- Step 3: Wait for graceful termination timeout of 60 seconds before host shutdown.\n"
        ),
        start_char=0,
        end_char=150,
    )
    verifier = StructuredVerifier(chunks=[chunk])

    res = verifier.verify_claim(StructuredClaim(
        claim_id="test_prop_214",
        subject_entity="Drain Sequencing",
        attribute_name="graceful termination timeout",
        claimed_value="60 seconds",
    ))
    assert res.verdict == VerificationVerdict.SUPPORTED


# =============================================================================
# 3. Router Integration & End-to-End Routing Tests
# =============================================================================

def test_router_handles_check_if_verification():
    router = GovernedToolRouter()
    resp = router.route_request("Check if mesh network data port is 80", request_id="test_route_279")
    assert resp.route == CanonicalRoute.STRUCTURED_VERIFICATION


def test_router_handles_container_execution_request():
    router = GovernedToolRouter()
    resp = router.route_request("run python: print(123)", request_id="test_cont_exec")
    assert resp.route == CanonicalRoute.CONTAINER_EXECUTION
    # On Windows without docker, execution status is FAILED and validation REJECTED
    assert resp.execution_status == "FAILED"
    assert "RUNTIME_UNAVAILABLE" in str(resp.abstention_reason)
