"""Tests for Batches 1541-1580: Multi-Turn Tool Chaining, Governed State, Sandbox Security & Benchmark Integrity."""

import pytest
import os
import tempfile
from chakrview.runtime.conversation_state import (
    ConversationStateManager,
    ToolExecutionRecord,
    ToolResultType,
)
from chakrview.runtime.sandboxed_python_tool import (
    SandboxedPythonTool,
    SandboxExecutionResult,
)
from chakrview.runtime.governed_tool_router import (
    GovernedToolRouter,
    CanonicalRoute,
)
from chakrview.runtime.safe_calculator import SafeCalculator
from chakrview.runtime.structured_verifier import (
    StructuredVerifier,
    StructuredClaim,
    VerificationRecord,
    VerificationVerdict,
)
from chakrverify.core import DocumentChunk


# ============================================================================
# Phase 3 & 4: Multi-Turn Conversation State & Chaining Tests
# ============================================================================

def test_conversation_state_basic_recording():
    mgr = ConversationStateManager(max_chain_length=5)
    rec = mgr.record_tool_result(
        conversation_id="conv_1",
        tenant_id="tenant_a",
        tool_id="safe_calculator",
        operation="arithmetic_evaluation",
        arguments={"expression": "120 + 80"},
        result=200,
        result_type=ToolResultType.NUMERIC,
        success=True,
        provenance={"expr": "120 + 80"},
    )
    assert rec.turn_id == 1
    assert rec.result == 200

    session = mgr.get_or_create_session("conv_1", tenant_id="tenant_a")
    assert len(session.tool_history) == 1
    assert session.tool_history[0].result == 200


def test_conversation_state_anaphora_resolution():
    mgr = ConversationStateManager()
    mgr.record_tool_result(
        conversation_id="conv_chain",
        tenant_id="tenant_x",
        tool_id="safe_calculator",
        operation="arithmetic_evaluation",
        arguments={"expression": "120 + 80"},
        result=200,
        result_type=ToolResultType.NUMERIC,
        success=True,
        provenance={"expr": "120 + 80"},
    )

    # Test "that"
    res1 = mgr.resolve_reference("Now divide that by 4", conversation_id="conv_chain", tenant_id="tenant_x")
    assert res1.resolved is True
    assert res1.referenced_value == 200
    assert "200 / 4" in res1.resolved_expression

    # Test "it"
    res2 = mgr.resolve_reference("subtract 50 from it", conversation_id="conv_chain", tenant_id="tenant_x")
    assert res2.resolved is True
    assert res2.referenced_value == 200

    # Test "the result"
    res3 = mgr.resolve_reference("multiply the result by 3", conversation_id="conv_chain", tenant_id="tenant_x")
    assert res3.resolved is True
    assert res3.referenced_value == 200


def test_conversation_isolation_and_cross_tenant_protection():
    mgr = ConversationStateManager()
    mgr.record_tool_result(
        conversation_id="conv_a",
        tenant_id="tenant_alpha",
        tool_id="safe_calculator",
        operation="arithmetic_evaluation",
        arguments={"expression": "900 + 99"},
        result=999,
        result_type=ToolResultType.NUMERIC,
        success=True,
        provenance={"expr": "900 + 99"},
    )

    # Attempt cross-tenant access to same conversation_id
    res_cross = mgr.resolve_reference(
        "divide that by 3", conversation_id="conv_a", tenant_id="tenant_beta"
    )
    assert res_cross.resolved is False

    # Attempt cross-conversation access
    res_diff_conv = mgr.resolve_reference(
        "divide that by 3", conversation_id="conv_b", tenant_id="tenant_alpha"
    )
    assert res_diff_conv.resolved is False


def test_conversation_reset_fail_closed():
    mgr = ConversationStateManager()
    mgr.record_tool_result(
        conversation_id="conv_reset",
        tenant_id="tenant_a",
        tool_id="safe_calculator",
        operation="arithmetic_evaluation",
        arguments={"expression": "250 * 2"},
        result=500,
        result_type=ToolResultType.NUMERIC,
        success=True,
        provenance={"expr": "250 * 2"},
    )
    # Reset
    mgr.reset_session("conv_reset", tenant_id="tenant_a")

    # Reference resolution after reset must fail closed
    res = mgr.resolve_reference("add 10 to that", conversation_id="conv_reset", tenant_id="tenant_a")
    assert res.resolved is False


def test_governed_router_arithmetic_chaining():
    router = GovernedToolRouter()
    # Turn 1
    resp1 = router.route_request(
        "Calculate 120 + 80",
        conversation_id="c_chain",
        tenant_id="t1",
        request_id="req_1",
    )
    assert resp1.route == CanonicalRoute.CALCULATOR
    assert resp1.execution_status == "SUCCESS"
    assert resp1.result_payload == 200

    # Turn 2: Chaining with 'that'
    resp2 = router.route_request(
        "Now divide that by 4",
        conversation_id="c_chain",
        tenant_id="t1",
        request_id="req_2",
    )
    assert resp2.route == CanonicalRoute.CALCULATOR
    assert resp2.execution_status == "SUCCESS"
    assert resp2.result_payload == 50.0


# ============================================================================
# Phase 5 & 6: Sandboxed Python Tool Tests
# ============================================================================

def test_sandbox_blocks_forbidden_imports():
    sandbox = SandboxedPythonTool(allow_arbitrary_code=False)
    bad_code = "import os\nprint(os.getcwd())"
    res = sandbox.execute_script(bad_code, execution_id="test_import_os")
    assert res.status == "SECURITY_VIOLATION"
    assert "os" in res.error


def test_sandbox_blocks_socket_import():
    sandbox = SandboxedPythonTool(allow_arbitrary_code=False)
    bad_code = "import socket\ns = socket.socket()"
    res = sandbox.execute_script(bad_code, execution_id="test_import_socket")
    assert res.status == "SECURITY_VIOLATION"
    assert "socket" in res.error


def test_sandbox_timeout_enforcement():
    sandbox = SandboxedPythonTool(allow_arbitrary_code=True)
    loop_code = "import time\nwhile True:\n    pass"
    res = sandbox.execute_script(loop_code, timeout=0.5, execution_id="test_timeout")
    assert res.status == "TIMEOUT"
    assert res.timed_out is True


def test_sandbox_output_truncation():
    sandbox = SandboxedPythonTool(allow_arbitrary_code=True)
    big_output_code = "print('X' * 100000)"
    res = sandbox.execute_script(big_output_code, execution_id="test_big_out")
    assert res.output_truncated is True
    assert len(res.stdout.encode("utf-8")) <= sandbox.MAX_OUTPUT_BYTES


def test_sandbox_authorized_execution_and_cleanup():
    sandbox = SandboxedPythonTool(allow_arbitrary_code=True)
    safe_code = "print(sum([x * 2 for x in range(5)]))"
    res = sandbox.execute_script(safe_code, execution_id="test_safe")
    assert res.status == "SUCCESS"
    assert res.stdout.strip() == "20"
    assert res.workspace_cleaned is True


# ============================================================================
# Phase 7: Structured Proposition Extraction & Verification Tests
# ============================================================================

def test_structured_verifier_multi_entity_isolation():
    chunks = [
        DocumentChunk(
            doc_id="doc1",
            chunk_index=0,
            text="# Service Specifications\n- Service A storage capacity: 50 GB.\n- Service B storage capacity: 200 GB.",
            start_char=0,
            end_char=100,
        )
    ]
    verifier = StructuredVerifier(chunks=chunks)
    claim_a = StructuredClaim(
        claim_id="c1",
        subject_entity="Service A",
        attribute_name="storage capacity",
        claimed_value="50 GB",
    )
    rec_a = verifier.verify_claim(claim_a)
    assert rec_a.verdict == VerificationVerdict.SUPPORTED
    assert "50" in str(rec_a.evidence_value)
    assert "200" not in str(rec_a.evidence_value)


def test_structured_verifier_verdict_supported():
    chunks = [
        DocumentChunk(
            doc_id="doc1",
            chunk_index=0,
            text="# RPC Configuration\n- Default timeout threshold: 30 seconds for all RPC workers.",
            start_char=0,
            end_char=100,
        )
    ]
    verifier = StructuredVerifier(chunks=chunks)
    claim = StructuredClaim(
        claim_id="c2",
        subject_entity="RPC Configuration",
        attribute_name="default timeout threshold",
        claimed_value="30 seconds",
    )
    result = verifier.verify_claim(claim)
    assert result.verdict == VerificationVerdict.SUPPORTED


def test_structured_verifier_verdict_contradicted():
    chunks = [
        DocumentChunk(
            doc_id="doc1",
            chunk_index=0,
            text="# RPC Configuration\n- Default timeout threshold: 30 seconds for all RPC workers.",
            start_char=0,
            end_char=100,
        )
    ]
    verifier = StructuredVerifier(chunks=chunks)
    claim = StructuredClaim(
        claim_id="c3",
        subject_entity="RPC Configuration",
        attribute_name="default timeout threshold",
        claimed_value="120 seconds",
    )
    result = verifier.verify_claim(claim)
    assert result.verdict == VerificationVerdict.CONTRADICTED
