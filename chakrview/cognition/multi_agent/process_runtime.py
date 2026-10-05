"""
ChakrView Step 114: Process-Isolated Worker Subprocess Entry Point.

This script executes as an independent OS process, reads a RequestEnvelope
from standard input (or dedicated pipe), invokes the specialized worker
with strictly bounded context, and emits a validated ResponseEnvelope to stdout.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chakrview.cognition.multi_agent.contracts import (
    WorkerRole,
    WorkerContract,
    WorkerPackage,
)
from chakrview.cognition.multi_agent.result import (
    WorkerResult,
    WorkerExecutionStatus,
)
from chakrview.cognition.multi_agent.transport import (
    PROTOCOL_VERSION_V1,
    RequestEnvelope,
    ResponseEnvelope,
)
from chakrview.cognition.multi_agent.worker import (
    ProjectAnalystWorker,
    PlannerWorker,
    ImplementerWorker,
    TestEngineerWorker,
    ReviewerWorker,
)
from chakrview.cognition.autonomous_benchmark import build_benchmark_tool_gate


def run_worker_process(req_data: Dict[str, Any], workspace_root: Path) -> Dict[str, Any]:
    """Execute worker within this isolated process."""
    req_env = RequestEnvelope.from_dict(req_data)
    req_env.validate()

    # Reconstruct contract and package
    role = WorkerRole(req_env.role)
    contract = WorkerContract(
        worker_id=req_env.worker_id,
        role=role,
        task_id=req_env.task_id,
        allowed_tools=req_env.allowed_tools,
        allowed_files=req_env.allowed_files,
        context_budget=req_env.context_budget,
        expected_output_schema=req_env.expected_output_schema,
    )
    contract.validate()

    pkg_payload = req_env.package_payload
    package = WorkerPackage(
        task_id=pkg_payload.get("task_id", req_env.task_id),
        objective=pkg_payload.get("objective", ""),
        targeted_code_context=pkg_payload.get("targeted_code_context", {}),
        dependency_outputs=pkg_payload.get("dependency_outputs", {}),
        relevant_ppb_records=pkg_payload.get("relevant_ppb_records", []),
        acceptance_criteria=pkg_payload.get("acceptance_criteria", ""),
        measured_context_tokens=pkg_payload.get("measured_context_tokens", req_env.context_token_count),
    )

    # Initialize tool gate in this worker process
    tool_gate, _ = build_benchmark_tool_gate(workspace_root)

    # Instantiate specialized worker
    worker_map = {
        WorkerRole.PROJECT_ANALYST: lambda: ProjectAnalystWorker(req_env.worker_id, tool_gate, workspace_root),
        WorkerRole.PLANNER: lambda: PlannerWorker(req_env.worker_id),
        WorkerRole.IMPLEMENTER: lambda: ImplementerWorker(req_env.worker_id, tool_gate, workspace_root),
        WorkerRole.TEST_ENGINEER: lambda: TestEngineerWorker(req_env.worker_id, tool_gate, workspace_root),
        WorkerRole.REVIEWER: lambda: ReviewerWorker(req_env.worker_id, tool_gate, workspace_root),
    }

    if role not in worker_map:
        raise ValueError(f"Unsupported role for isolated worker: {role}")

    worker = worker_map[role]()
    result: WorkerResult = worker.execute(contract, package)

    # Check for schema validation
    try:
        result.validate_against_schema(contract.expected_output_schema)
    except Exception as e:
        result.status = WorkerExecutionStatus.MALFORMED_OUTPUT
        result.error = f"Malformed output schema: {str(e)}"

    # Format result payload
    res_dict = {
        "worker_id": result.worker_id,
        "task_id": result.task_id,
        "role": result.role.value,
        "status": result.status.value,
        "evidence": result.evidence,
        "proposed_changes": result.proposed_changes,
        "verification_output": result.verification_output,
        "failures": result.failures,
        "confidence": result.confidence,
        "context_tokens_used": result.context_tokens_used,
        "error": result.error,
    }
    payload_bytes = json.dumps(res_dict, sort_keys=True).encode("utf-8")
    result_hash = hashlib.sha256(payload_bytes).hexdigest()

    resp_env = ResponseEnvelope(
        protocol_version=PROTOCOL_VERSION_V1,
        worker_id=req_env.worker_id,
        task_id=req_env.task_id,
        package_id=req_env.package_id,
        result_status=result.status.value,
        result_payload=res_dict,
        result_hash=result_hash,
        error=result.error,
    )
    return resp_env.to_dict()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python process_runtime.py <workspace_root>\n")
        sys.exit(1)

    workspace_root = Path(sys.argv[1]).resolve()
    # Read RequestEnvelope from stdin
    raw_input = sys.stdin.read()
    if not raw_input.strip():
        sys.stderr.write("Empty input envelope\n")
        sys.exit(2)

    try:
        req_data = json.loads(raw_input)
        resp_data = run_worker_process(req_data, workspace_root)
        sys.stdout.write(json.dumps(resp_data))
        sys.stdout.flush()
        sys.exit(0)
    except Exception as exc:
        err_dict = {
            "protocol_version": PROTOCOL_VERSION_V1,
            "worker_id": "unknown",
            "task_id": "unknown",
            "package_id": "unknown",
            "result_status": "FAILED",
            "result_payload": {},
            "result_hash": hashlib.sha256(b"{}").hexdigest(),
            "error": str(exc),
        }
        sys.stdout.write(json.dumps(err_dict))
        sys.stdout.flush()
        sys.exit(1)
