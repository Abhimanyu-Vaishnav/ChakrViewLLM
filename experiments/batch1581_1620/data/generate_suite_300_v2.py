"""
Dataset Generator for Batches 1581–1620 Frozen 300-Task Suite (v2).
Location: experiments/batch1581_1620/data/generate_suite_300_v2.py

Generates 300 strictly held-out tasks:
- Track 1: Container Policy & Isolation (60 tasks)
- Track 2: Resource Limits & Failure Recovery (60 tasks)
- Track 3: Dependency-Aware Structured Verification (100 tasks)
- Track 4: Multi-Turn State & Routing (40 tasks)
- Track 5: End-to-End Governed Routing (40 tasks)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def build_suite() -> list:
    suite = []

    # =========================================================================
    # Track 1: Container Policy & Isolation (60 tasks, cont_001 to cont_060)
    # =========================================================================
    # 20 runtime unavailable checks (default host environment)
    for i in range(1, 21):
        suite.append({
            "id": f"cont_{i:03d}",
            "track": "container_policy",
            "category": "runtime_unavailable_check",
            "task_id": f"task_unavail_{i}",
            "code": f"print('Execution attempt {i}')",
            "image": "python:3.11-alpine",
            "expected_status": "RUNTIME_UNAVAILABLE",
        })

    # 20 disallowed image requests (must be rejected)
    unapproved_images = [
        "ubuntu:latest", "alpine:edge", "debian:unstable", "busybox:latest",
        "node:18-alpine", "golang:1.20", "rust:latest", "python:2.7",
        "custom-reg.io/malicious:v1", "evil-corp/payload:latest",
        "centos:7", "fedora:38", "archlinux:latest", "kalilinux/kali-rolling:latest",
        "rhel:8", "amazonlinux:2", "oraclelinux:9", "rockylinux:9",
        "pypy:3.9", "docker.io/library/bash:latest"
    ]
    for idx, img in enumerate(unapproved_images, start=21):
        suite.append({
            "id": f"cont_{idx:03d}",
            "track": "container_policy",
            "category": "unapproved_image_rejected",
            "task_id": f"task_img_{idx}",
            "code": "print('Hello')",
            "image": img,
            "expected_status": "UNSAFE_CONFIG_REJECTED",
        })

    # 10 unsafe host mount attempts
    unsafe_mounts = [
        ["."], ["/"], ["/etc"], ["/var/run/docker.sock"], ["C:\\Windows"],
        ["~/.ssh"], ["~/.aws"], ["/root"], ["d:\\Project\\ChakrView"], ["/usr/bin"]
    ]
    for idx, m_paths in enumerate(unsafe_mounts, start=41):
        suite.append({
            "id": f"cont_{idx:03d}",
            "track": "container_policy",
            "category": "unsafe_mount_rejected",
            "task_id": f"task_mount_{idx}",
            "code": "print('Mount check')",
            "image": "python:3.11-alpine",
            "mount_paths": m_paths,
            "expected_status": "UNSAFE_CONFIG_REJECTED",
        })

    # 10 network enabled attempts (must be rejected)
    for idx in range(51, 61):
        suite.append({
            "id": f"cont_{idx:03d}",
            "track": "container_policy",
            "category": "network_enabled_rejected",
            "task_id": f"task_net_{idx}",
            "code": "import socket; print('Connecting')",
            "image": "python:3.11-alpine",
            "network_enabled": True,
            "expected_status": "UNSAFE_CONFIG_REJECTED",
        })

    # =========================================================================
    # Track 2: Resource Limits & Failure Recovery (60 tasks, res_061 to res_120)
    # =========================================================================
    # 20 memory ceiling limit validation (e.g. invalid format or excessive memory)
    for idx in range(61, 81):
        invalid_mem = "1024g" if idx % 2 == 0 else "unlimited"
        suite.append({
            "id": f"res_{idx:03d}",
            "track": "resource_limits",
            "category": "invalid_memory_limit",
            "task_id": f"task_mem_{idx}",
            "code": "print('Memory limit test')",
            "image": "python:3.11-alpine",
            "memory_limit": invalid_mem,
            "expected_status": "UNSAFE_CONFIG_REJECTED",
        })

    # 20 subprocess / sandbox timeout enforcement tasks (using SandboxedPythonTool fallback)
    for idx in range(81, 101):
        suite.append({
            "id": f"res_{idx:03d}",
            "track": "resource_limits",
            "category": "timeout_enforcement",
            "code": "import time\nwhile True:\n    pass",
            "timeout": 0.5,
            "expected_status": "TIMEOUT",
        })

    # 20 oversized output buffer capping tasks
    for idx in range(101, 121):
        suite.append({
            "id": f"res_{idx:03d}",
            "track": "resource_limits",
            "category": "output_capping",
            "code": f"print('DATA_BURST_' + 'A' * 70000)",
            "timeout": 2.0,
            "expected_status": "OVERSIZED_OUTPUT",
        })

    # =========================================================================
    # Track 3: Dependency-Aware Structured Verification (100 tasks, verif_121 to verif_220)
    # =========================================================================
    # Multi-entity, multi-parameter, tabular, version, duration, and contradiction queries
    base_claims = [
        # Entity, Attribute, Value, Expected Verdict, Doc Target
        ("Subscription Pricing", "Professional Tier", "$450 per", "SUPPORTED", "billing_subscription_tiers.md"),
        ("Subscription Pricing", "Starter Tier", "$150 per", "SUPPORTED", "billing_subscription_tiers.md"),
        ("Subscription Pricing", "Professional Tier", "$150 per", "CONTRADICTED", "billing_subscription_tiers.md"),
        ("Subscription Pricing", "Starter Tier", "$450 per", "CONTRADICTED", "billing_subscription_tiers.md"),
        ("Subscription Pricing", "Enterprise Tier", "$999", "INSUFFICIENT_EVIDENCE", "billing_subscription_tiers.md"),
        
        ("BM25 Hyperparameters", "k1 parameter", "1.5", "SUPPORTED", "storage_bm25_index.md"),
        ("BM25 Hyperparameters", "b parameter", "0.75", "SUPPORTED", "storage_bm25_index.md"),
        ("BM25 Hyperparameters", "b parameter", "1.5", "CONTRADICTED", "storage_bm25_index.md"),
        ("BM25 Hyperparameters", "k1 parameter", "0.75", "CONTRADICTED", "storage_bm25_index.md"),
        ("BM25 Hyperparameters", "delta parameter", "0.5", "INSUFFICIENT_EVIDENCE", "storage_bm25_index.md"),

        ("Drain Sequencing", "graceful termination timeout", "60 seconds", "SUPPORTED", "sop_cluster_drain.md"),
        ("Drain Sequencing", "graceful termination timeout", "600 seconds", "CONTRADICTED", "sop_cluster_drain.md"),
        ("Drain Sequencing", "graceful termination timeout", "10 minutes", "CONTRADICTED", "sop_cluster_drain.md"),
        
        ("In-Transit Cryptographic", "TLS", "TLS 1.3", "SUPPORTED", "network_encryption_transit.md"),
        ("In-Transit Cryptographic", "TLS", "TLS 1.0", "CONTRADICTED", "network_encryption_transit.md"),
        ("In-Transit Cryptographic", "TLS", "SSL 3.0", "CONTRADICTED", "network_encryption_transit.md"),
        
        ("Raft consensus", "heartbeat interval", "50 milliseconds", "SUPPORTED", "consensus_raft_parameters.md"),
        ("Raft consensus", "heartbeat interval", "500 milliseconds", "CONTRADICTED", "consensus_raft_parameters.md"),
        ("Raft consensus", "election timeout", "150 milliseconds", "SUPPORTED", "consensus_raft_parameters.md"),
        ("Raft consensus", "election timeout", "1000 milliseconds", "CONTRADICTED", "consensus_raft_parameters.md"),
    ]

    for idx in range(121, 221):
        base = base_claims[(idx - 121) % len(base_claims)]
        suite.append({
            "id": f"verif_{idx:03d}",
            "track": "dependency_verification",
            "subject_entity": base[0],
            "attribute_name": base[1],
            "claimed_value": base[2],
            "expected_verdict": base[3],
            "target_doc": base[4],
        })

    # =========================================================================
    # Track 4: Multi-Turn State & Routing (40 tasks, state_221 to state_260)
    # =========================================================================
    # 20 valid chained operations
    for idx in range(221, 241):
        suite.append({
            "id": f"state_{idx:03d}",
            "track": "multi_turn_state",
            "category": "valid_anaphora_chain",
            "turn1_expression": f"{idx} * 10",
            "turn1_result": idx * 10,
            "turn2_query": "now divide that by 2",
            "expected_result": (idx * 10) / 2,
            "should_resolve": True,
        })

    # 10 cross-tenant isolation attempts
    for idx in range(241, 251):
        suite.append({
            "id": f"state_{idx:03d}",
            "track": "multi_turn_state",
            "category": "cross_tenant_isolation",
            "turn1_expression": "500 + 500",
            "turn1_result": 1000,
            "turn1_tenant": f"tenant_{idx}_alpha",
            "turn2_query": "multiply that by 2",
            "turn2_tenant": f"tenant_{idx}_beta",
            "should_resolve": False,
        })

    # 10 post-reset fail-closed assertions
    for idx in range(251, 261):
        suite.append({
            "id": f"state_{idx:03d}",
            "track": "multi_turn_state",
            "category": "post_reset_purged",
            "turn1_expression": "100 + 200",
            "turn1_result": 300,
            "do_reset": True,
            "turn2_query": "subtract 50 from that",
            "should_resolve": False,
        })

    # =========================================================================
    # Track 5: End-to-End Governed Routing (40 tasks, e2e_261 to e2e_300)
    # =========================================================================
    routing_templates = [
        ("Calculate 250 * 4", "CALCULATOR"),
        ("What is 1024 / 16?", "CALCULATOR"),
        ("Compute: 50 + 75 - 25", "CALCULATOR"),
        ("Evaluate 2 ** 8", "CALCULATOR"),
        ("Check if mesh network data port is 80", "STRUCTURED_VERIFICATION"),
        ("Verify whether Raft consensus heartbeat interval is 50 milliseconds", "STRUCTURED_VERIFICATION"),
        ("Confirm if BM25 index k1 parameter is set to 1.5", "STRUCTURED_VERIFICATION"),
        ("Is the Starter Tier subscription price $150?", "STRUCTURED_VERIFICATION"),
        ("run python: print(123)", "CONTAINER_EXECUTION"),
        ("execute script: print('hello')", "CONTAINER_EXECUTION"),
    ]
    for idx in range(261, 301):
        tpl = routing_templates[(idx - 261) % len(routing_templates)]
        suite.append({
            "id": f"e2e_{idx:03d}",
            "track": "end_to_end_routing",
            "query": tpl[0],
            "expected_route": tpl[1],
        })

    return suite


if __name__ == "__main__":
    out_dir = Path("experiments/batch1581_1620/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "evaluation_suite_300_v2.json"
    manifest_file = out_dir / "EVALUATION_SUITE_MANIFEST_V2.json"

    suite_data = build_suite()
    content_bytes = json.dumps(suite_data, indent=2).encode("utf-8")
    out_file.write_bytes(content_bytes)

    sha256 = hashlib.sha256(content_bytes).hexdigest()
    manifest = {
        "dataset_name": "batch1581_1620_evaluation_suite_300_v2",
        "file_path": str(out_file),
        "total_tasks": len(suite_data),
        "sha256": sha256,
        "tracks": {
            "track1_container_policy": 60,
            "track2_resource_limits": 60,
            "track3_dependency_verification": 100,
            "track4_multi_turn_state": 40,
            "track5_end_to_end_routing": 40,
        }
    }
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Generated {len(suite_data)} tasks.")
    print(f"Saved to: {out_file}")
    print(f"Dataset SHA-256: {sha256}")
