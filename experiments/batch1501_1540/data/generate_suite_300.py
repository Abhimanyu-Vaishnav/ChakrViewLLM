"""
Dataset Generator for Batches 1501–1540 Frozen Evaluation Suite (300 Tasks).
Location: experiments/batch1501_1540/data/generate_suite_300.py

Generates:
- 100 Arithmetic Tasks (addition, subtraction, multiplication, division, modulo, exponentiation, unary, nested parens, edge cases, division by zero, adversarial security attempts).
- 100 Structured Verification Tasks (supported, contradicted, insufficient evidence, conflicting evidence, invalid claims).
- 50 Routing Tasks (direct chat, calculator dispatch, knowledge retrieval, structured verification, abstention, error).
- 50 End-to-End Assistant Tasks (arithmetic queries, factual retrieval, false-premise refutations, unanswerable abstentions, follow-up queries).

Outputs to experiments/batch1501_1540/data/evaluation_suite_300.json
"""
import hashlib
import json
from pathlib import Path

OUT_FILE = Path(__file__).resolve().parent / "evaluation_suite_300.json"
MANIFEST_FILE = Path(__file__).resolve().parent / "EVALUATION_SUITE_MANIFEST.json"


def build_suite() -> list[dict]:
    tasks = []

    # ==========================================
    # TRACK 1: ARITHMETIC (100 Tasks)
    # ==========================================
    # 1a. Addition & Subtraction (20 tasks)
    add_sub_cases = [
        ("147 + 258", 405), ("1024 - 512", 512), ("999 + 1", 1000), ("450 - 150", 300),
        ("384 + 128", 512), ("73 + 89", 162), ("10000 - 9999", 1), ("12345 + 54321", 66666),
        ("500 - 750", -250), ("0 + 0", 0), ("-15 + 30", 15), ("-40 - 60", -100),
        ("2.5 + 3.75", 6.25), ("10.5 - 4.2", 6.3), ("1000 + 2000 + 3000", 6000),
        ("50 - 20 - 10", 20), ("100 + (-25)", 75), ("(-50) + (-50)", -100),
        ("999999 + 1", 1000000), ("5000 - 1234", 3766)
    ]
    for idx, (expr, exp) in enumerate(add_sub_cases, 1):
        tasks.append({
            "id": f"arith_{idx:03d}",
            "track": "arithmetic",
            "category": "add_sub",
            "expression": expr,
            "expected_success": True,
            "expected_result": exp,
            "expected_error": None,
        })

    # 1b. Multiplication & Division (20 tasks)
    mul_div_cases = [
        ("12 * 12", 144), ("100 / 4", 25), ("1024 / 8", 128), ("4096 / 16", 256),
        ("512 / 64", 8), ("50 * 20", 1000), ("32 * 32", 1024), ("1000 / 10", 100),
        ("7 * 8", 56), ("9 * 9", 81), ("15 * 0", 0), ("0 / 5", 0),
        ("2.5 * 4", 10), ("7.5 / 2.5", 3), ("100 * 0.05", 5), ("128 * 8", 1024),
        ("10000 / 100", 100), ("11 * 13", 143), ("16 * 16", 256), ("64 / 2 / 2", 16)
    ]
    for idx, (expr, exp) in enumerate(mul_div_cases, 21):
        tasks.append({
            "id": f"arith_{idx:03d}",
            "track": "arithmetic",
            "category": "mul_div",
            "expression": expr,
            "expected_success": True,
            "expected_result": exp,
            "expected_error": None,
        })

    # 1c. Modulo & Exponentiation (20 tasks)
    mod_pow_cases = [
        ("17 % 5", 2), ("100 % 7", 2), ("64 % 8", 0), ("255 % 16", 15),
        ("10 % 3", 1), ("1024 % 10", 4), ("2 ** 3", 8), ("2 ** 8", 256),
        ("2 ** 10", 1024), ("3 ** 4", 81), ("5 ** 3", 125), ("10 ** 4", 10000),
        ("4 ** 0", 1), ("0 ** 5", 0), ("100 % 100", 0), ("7 % 10", 7),
        ("2 ** 16", 65536), ("9 ** 2", 81), ("50 % 6", 2), ("3 ** 3", 27)
    ]
    for idx, (expr, exp) in enumerate(mod_pow_cases, 41):
        tasks.append({
            "id": f"arith_{idx:03d}",
            "track": "arithmetic",
            "category": "mod_pow",
            "expression": expr,
            "expected_success": True,
            "expected_result": exp,
            "expected_error": None,
        })

    # 1d. Nested Precedence & Parentheses (20 tasks)
    paren_cases = [
        ("10 + 20 * 3", 70), ("(10 + 20) * 3", 90), ("100 - 20 * 3", 40),
        ("(100 - 20) * 3", 240), ("(50 + 50) / (2 * 5)", 10), ("100 / (2 + 3)", 20),
        ("2 * (3 + 4 * (5 - 2))", 30), ("((10 + 5) * 2) - 10", 20),
        ("(100 // 3) + (100 % 3)", 34), ("-(10 + 5) * 2", -30),
        ("2 ** (3 + 1)", 16), ("(2 + 3) ** 2", 25), ("100 - (50 - (20 - 10))", 60),
        ("((5 * 4) + (6 * 3)) / 2", 19), ("10 + (20 / 4) - (3 * 2)", 9),
        ("(8 * 4) + (64 - 16)", 80), ("(1024 / 8) + (512 / 16)", 160),
        ("((15 + 5) * (12 - 2)) / 10", 20), ("500 - (100 * 4) + 50", 150),
        ("3 * (4 + 5) - 2 * (1 + 2)", 21)
    ]
    for idx, (expr, exp) in enumerate(paren_cases, 61):
        tasks.append({
            "id": f"arith_{idx:03d}",
            "track": "arithmetic",
            "category": "parens_precedence",
            "expression": expr,
            "expected_success": True,
            "expected_result": exp,
            "expected_error": None,
        })

    # 1e. Division by Zero, Domain Errors & Adversarial Security Attacks (20 tasks)
    security_cases = [
        ("10 / 0", False, None, "DIVISION_BY_ZERO"),
        ("1024 // 0", False, None, "DIVISION_BY_ZERO"),
        ("50 % 0", False, None, "DIVISION_BY_ZERO"),
        ("0 / 0", False, None, "DIVISION_BY_ZERO"),
        ("10 / (5 - 5)", False, None, "DIVISION_BY_ZERO"),
        ("__import__('os').system('dir')", False, None, "SECURITY_OR_DOMAIN_VIOLATION"),
        ("eval('2 + 2')", False, None, "SECURITY_OR_DOMAIN_VIOLATION"),
        ("exec('x = 1')", False, None, "SECURITY_OR_DOMAIN_VIOLATION"),
        ("open('secret.txt')", False, None, "SECURITY_OR_DOMAIN_VIOLATION"),
        ("[x for x in range(10)]", False, None, "SECURITY_OR_DOMAIN_VIOLATION"),
        ("lambda x: x + 1", False, None, "SECURITY_OR_DOMAIN_VIOLATION"),
        ("2 ** 500", False, None, "SECURITY_OR_DOMAIN_VIOLATION"),  # Exponent too large
        ("10000000 ** 20", False, None, "SECURITY_OR_DOMAIN_VIOLATION"),  # Base and exponent too large
        ("math.sqrt(16)", False, None, "SECURITY_OR_DOMAIN_VIOLATION"),  # Attribute / call rejected
        ("print('hello')", False, None, "SECURITY_OR_DOMAIN_VIOLATION"),
        ("'hello' + 'world'", False, None, "SECURITY_OR_DOMAIN_VIOLATION"),  # String constant rejected
        ("True + 1", False, None, "SECURITY_OR_DOMAIN_VIOLATION"),  # Boolean rejected
        ("None + 5", False, None, "SECURITY_OR_DOMAIN_VIOLATION"),
        ("10 ++* 20", False, None, "SYNTAX_ERROR"),
        ("((10 + 20)", False, None, "SYNTAX_ERROR"),
    ]
    for idx, (expr, exp_succ, exp_res, exp_err) in enumerate(security_cases, 81):
        tasks.append({
            "id": f"arith_{idx:03d}",
            "track": "arithmetic",
            "category": "security_and_domain_limits",
            "expression": expr,
            "expected_success": exp_succ,
            "expected_result": exp_res,
            "expected_error": exp_err,
        })

    # ==========================================
    # TRACK 2: STRUCTURED VERIFICATION (100 Tasks)
    # ==========================================
    # 2a. SUPPORTED claims (25 tasks)
    supported_claims = [
        ("BM25 Hyperparameters", "k1 parameter", "1.5", "storage_bm25_index.md"),
        ("BM25 Hyperparameters", "b parameter", "0.75", "storage_bm25_index.md"),
        ("BM25 Hyperparameters", "minimum indexed token length", "2", "storage_bm25_index.md"),
        ("BLOB store", "maximum single document size", "10485760 bytes", "storage_blob_store.md"),
        ("BLOB store", "maximum single document size", "10 MB", "storage_blob_store.md"),
        ("Raft consensus", "heartbeat interval", "50 milliseconds", "network_consensus_raft.md"),
        ("Raft consensus", "heartbeat interval", "50 ms", "network_consensus_raft.md"),
        ("mesh network", "maximum node cluster diameter", "16", "network_mesh_topology.md"),
        ("mesh network", "data transmission port", "8443", "network_mesh_topology.md"),
        ("Subscription Pricing", "Starter Tier", "$150", "billing_subscription_tiers.md"),
        ("Subscription Pricing", "Starter Tier", "150", "billing_subscription_tiers.md"),
        ("Subscription Pricing", "concurrent worker processes in Starter Tier", "2", "billing_subscription_tiers.md"),
        ("Subscription Pricing", "Professional Tier", "$450", "billing_subscription_tiers.md"),
        ("Subscription Pricing", "concurrent worker processes in Professional Tier", "8", "billing_subscription_tiers.md"),
        ("Security Audit Trail", "log retention minimum", "365", "security_audit_logging.md"),
        ("Token Lifecycle", "expiration window", "3600 seconds", "security_auth_tokens.md"),
        ("Gateway API", "rate limit threshold", "60", "network_gateway_api.md"),
        ("Gateway API", "query endpoint", "POST /v1/assistant/query", "network_gateway_api.md"),
        ("Vector Index", "index dimension", "64", "storage_vector_index.md"),
        ("Vector Index", "normalized vector length", "1.0", "storage_vector_index.md"),
        ("Uptime Guarantees", "downtime financial credit", "15%", "billing_sla_commitments.md"),
        ("Uptime Guarantees", "severe outage credit", "50%", "billing_sla_commitments.md"),
        ("Drain Sequencing", "graceful termination timeout", "60 seconds", "sop_cluster_drain.md"),
        ("In-Transit Cryptographic", "TLS", "TLS 1.3", "network_encryption_transit.md"),
        ("Refund Policy", "satisfaction guarantee", "30-day", "billing_refund_policy.md"),
    ]
    for idx, (ent, attr, val, doc) in enumerate(supported_claims, 1):
        tasks.append({
            "id": f"verif_{idx:03d}",
            "track": "verification",
            "subject_entity": ent,
            "attribute_name": attr,
            "claimed_value": val,
            "expected_verdict": "SUPPORTED",
            "target_doc": doc,
        })

    # 2b. CONTRADICTED claims (25 tasks)
    contradicted_claims = [
        ("BM25 Hyperparameters", "k1 parameter", "2.5", "storage_bm25_index.md"),
        ("BM25 Hyperparameters", "b parameter", "0.95", "storage_bm25_index.md"),
        ("BM25 Hyperparameters", "minimum indexed token length", "5", "storage_bm25_index.md"),
        ("BLOB store", "maximum single document size", "50 MB", "storage_blob_store.md"),
        ("BLOB store", "maximum single document size", "1000 bytes", "storage_blob_store.md"),
        ("Raft consensus", "heartbeat interval", "250 milliseconds", "network_consensus_raft.md"),
        ("Raft consensus", "heartbeat interval", "5 seconds", "network_consensus_raft.md"),
        ("mesh network", "maximum node cluster diameter", "64", "network_mesh_topology.md"),
        ("mesh network", "data transmission port", "80", "network_mesh_topology.md"),
        ("Subscription Pricing", "Starter Tier", "$250", "billing_subscription_tiers.md"),
        ("Subscription Pricing", "Starter Tier", "50", "billing_subscription_tiers.md"),
        ("Subscription Pricing", "concurrent worker processes in Starter Tier", "10", "billing_subscription_tiers.md"),
        ("Subscription Pricing", "Professional Tier", "$900", "billing_subscription_tiers.md"),
        ("Subscription Pricing", "concurrent worker processes in Professional Tier", "1", "billing_subscription_tiers.md"),
        ("Security Audit Trail", "log retention minimum", "30 days", "security_audit_logging.md"),
        ("Token Lifecycle", "expiration window", "2 hours", "security_auth_tokens.md"),
        ("Gateway API", "rate limit threshold", "500", "network_gateway_api.md"),
        ("Gateway API", "query endpoint", "GET /api/v2/search", "network_gateway_api.md"),
        ("Vector Index", "index dimension", "512", "storage_vector_index.md"),
        ("Vector Index", "normalized vector length", "5.0", "storage_vector_index.md"),
        ("Uptime Guarantees", "downtime financial credit", "50%", "billing_sla_commitments.md"),
        ("Uptime Guarantees", "severe outage credit", "5%", "billing_sla_commitments.md"),
        ("Drain Sequencing", "graceful termination timeout", "600 seconds", "sop_cluster_drain.md"),
        ("In-Transit Cryptographic", "TLS", "TLS 1.0", "network_encryption_transit.md"),
        ("Refund Policy", "satisfaction guarantee", "90-day", "billing_refund_policy.md"),
    ]
    for idx, (ent, attr, val, doc) in enumerate(contradicted_claims, 26):
        tasks.append({
            "id": f"verif_{idx:03d}",
            "track": "verification",
            "subject_entity": ent,
            "attribute_name": attr,
            "claimed_value": val,
            "expected_verdict": "CONTRADICTED",
            "target_doc": doc,
        })

    # 2c. INSUFFICIENT_EVIDENCE claims (25 tasks)
    insufficient_claims = [
        ("Quantum Cryptography Core", "key length", "4096 qbits"),
        ("Mars Relay Satellite", "orbit altitude", "400 km"),
        ("ChakrView Blockchain", "gas limit", "30000000"),
        ("Deep Space Telemetry", "baud rate", "115200"),
        ("PostgreSQL Database", "max connections", "500"),
        ("Redis Cache Cluster", "eviction policy", "volatile-lru"),
        ("Kubernetes Ingress", "controller replica count", "5"),
        ("Apache Kafka", "replication factor", "3"),
        ("Elasticsearch Node", "heap size", "32 GB"),
        ("Consul Service Discovery", "wan gossip port", "8302"),
        ("Vault Secret Engine", "lease duration", "720 hours"),
        ("Docker Daemon", "storage driver", "overlay2"),
        ("Terraform Cloud", "execution concurrency", "10"),
        ("Prometheus Server", "scrape interval", "15 seconds"),
        ("Grafana Dashboard", "refresh rate", "5 seconds"),
        ("NGINX Gateway", "worker processes", "auto"),
        ("RabbitMQ Broker", "prefetch count", "50"),
        ("MongoDB Shard", "chunk size", "64 MB"),
        ("Ceph Storage Pool", "pg count", "1024"),
        ("Zookeeper Ensemble", "election port", "3888"),
        ("Hadoop NameNode", "edit log retention", "7 days"),
        ("Spark Executor", "core allocation", "4"),
        ("Flink JobManager", "checkpoint interval", "10000 ms"),
        ("Envoy Proxy", "circuit breaker max connections", "1024"),
        ("WireGuard VPN", "keepalive interval", "25 seconds"),
    ]
    for idx, (ent, attr, val) in enumerate(insufficient_claims, 51):
        tasks.append({
            "id": f"verif_{idx:03d}",
            "track": "verification",
            "subject_entity": ent,
            "attribute_name": attr,
            "claimed_value": val,
            "expected_verdict": "INSUFFICIENT_EVIDENCE",
            "target_doc": None,
        })

    # 2d. CONFLICTING_EVIDENCE & INVALID_CLAIM (25 tasks)
    # The corpus actually has two conflicting specs for ChakrMicro architecture:
    # compute_arch_core.md specifies: Hidden dimension: 384 units, Primary context ceiling: 512 active tokens.
    # compute_arch_v07_draft.md specifies: Experimental hidden dimension: 512 units, Context window ceiling: 1024 tokens.
    for idx in range(76, 91):
        attr_name = "hidden dimension" if idx % 2 == 0 else "context ceiling"
        tasks.append({
            "id": f"verif_{idx:03d}",
            "track": "verification",
            "subject_entity": "ChakrMicro",
            "attribute_name": attr_name,
            "claimed_value": "384 units" if attr_name == "hidden dimension" else "512 tokens",
            "expected_verdict": "CONFLICTING_EVIDENCE",
            "target_doc": "compute_arch_core.md and compute_arch_v07_draft.md",
        })
    for idx in range(91, 101):
        tasks.append({
            "id": f"verif_{idx:03d}",
            "track": "verification",
            "subject_entity": "",  # Missing entity makes claim INVALID
            "attribute_name": "port",
            "claimed_value": "8080",
            "expected_verdict": "INVALID_CLAIM",
            "target_doc": None,
        })

    # ==========================================
    # TRACK 3: ROUTING DISPATCH (50 Tasks)
    # ==========================================
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

        # 10 Knowledge Retrieval routes
        ("What is the term saturation parameter k1 in BM25?", "KNOWLEDGE_RETRIEVAL"),
        ("What is the maximum single document size in the BLOB store?", "KNOWLEDGE_RETRIEVAL"),
        ("What is the heartbeat interval in Raft consensus?", "KNOWLEDGE_RETRIEVAL"),
        ("What TCP port handles data transmission in the mesh network?", "KNOWLEDGE_RETRIEVAL"),
        ("What is the monthly subscription price for the Starter Tier?", "KNOWLEDGE_RETRIEVAL"),
        ("What is the monthly subscription price for the Professional Tier?", "KNOWLEDGE_RETRIEVAL"),
        ("What credit is applied if monthly uptime drops below 99.9%?", "KNOWLEDGE_RETRIEVAL"),
        ("What is the document length normalization parameter b in BM25?", "KNOWLEDGE_RETRIEVAL"),
        ("What is the maximum node cluster diameter in the mesh network?", "KNOWLEDGE_RETRIEVAL"),
        ("What is the retention duration of audit logs?", "KNOWLEDGE_RETRIEVAL"),

        # 10 Honest Abstention / Out-of-Domain routes
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
    for idx, (q, r) in enumerate(routing_cases, 1):
        tasks.append({
            "id": f"route_{idx:03d}",
            "track": "routing",
            "query": q,
            "expected_route": r,
        })

    # ==========================================
    # TRACK 4: END-TO-END ASSISTANT (50 Tasks)
    # ==========================================
    e2e_cases = [
        # 15 Math via Router
        ("Calculate: 147 + 258", "405", "CALCULATOR"),
        ("What is 1024 / 8?", "128", "CALCULATOR"),
        ("Evaluate: 50 * 20", "1000", "CALCULATOR"),
        ("Calculate: 255 % 16", "15", "CALCULATOR"),
        ("What is 2 ** 10?", "1024", "CALCULATOR"),
        ("Compute: 100 - 20 * 3", "40", "CALCULATOR"),
        ("What is 10000 / 100?", "100", "CALCULATOR"),
        ("Calculate: 12 * 12", "144", "CALCULATOR"),
        ("Compute: 999 + 1", "1000", "CALCULATOR"),
        ("What is 512 / 64?", "8", "CALCULATOR"),
        ("Calculate: 17 % 5", "2", "CALCULATOR"),
        ("Evaluate: 2 ** 8", "256", "CALCULATOR"),
        ("Calculate: (50 + 50) / 10", "10", "CALCULATOR"),
        ("What is 4096 / 16?", "256", "CALCULATOR"),
        ("Calculate: 384 + 128", "512", "CALCULATOR"),

        # 15 Factual Knowledge via Assistant
        ("What is the term saturation parameter k1 in the BM25 index?", "1.5", "KNOWLEDGE_RETRIEVAL"),
        ("What is the document length normalization parameter b in BM25?", "0.75", "KNOWLEDGE_RETRIEVAL"),
        ("What is the maximum single document size in the BLOB store?", "10485760", "KNOWLEDGE_RETRIEVAL"),
        ("What is the heartbeat interval in Raft consensus?", "50", "KNOWLEDGE_RETRIEVAL"),
        ("What TCP port handles data transmission in the mesh network?", "8443", "KNOWLEDGE_RETRIEVAL"),
        ("What is the monthly subscription price for the Starter Tier?", "150", "KNOWLEDGE_RETRIEVAL"),
        ("What is the monthly subscription price for the Professional Tier?", "450", "KNOWLEDGE_RETRIEVAL"),
        ("What credit is applied if monthly uptime drops below 99.9%?", "15%", "KNOWLEDGE_RETRIEVAL"),
        ("What is the maximum node cluster diameter in the mesh network?", "16", "KNOWLEDGE_RETRIEVAL"),
        ("What credit is applied if monthly uptime drops below 98.0%?", "50%", "KNOWLEDGE_RETRIEVAL"),
        ("Could you tell me what value k1 is set to for term saturation in BM25?", "1.5", "KNOWLEDGE_RETRIEVAL"),
        ("Raft consensus algorithm me heartbeat interval kitne ms ka set hota hai?", "50", "KNOWLEDGE_RETRIEVAL"),
        ("Mesh network topology ke andar data communication port kaunsa use hota hai?", "8443", "KNOWLEDGE_RETRIEVAL"),
        ("Starter subscription tier ka per-month fees kitna charge hota hai?", "150", "KNOWLEDGE_RETRIEVAL"),
        ("Professional tier plan ka monthly billing cost kya specify kiya gaya hai?", "450", "KNOWLEDGE_RETRIEVAL"),

        # 10 Verification / False Premise Refutations
        ("Verify if the BM25 index k1 parameter is 2.5", "CONTRADICTED", "STRUCTURED_VERIFICATION"),
        ("Verify whether the Starter Tier monthly price is $250", "CONTRADICTED", "STRUCTURED_VERIFICATION"),
        ("Is the heartbeat interval in Raft consensus 250 milliseconds?", "CONTRADICTED", "STRUCTURED_VERIFICATION"),
        ("Confirm if TCP port is 80 in the mesh network", "CONTRADICTED", "STRUCTURED_VERIFICATION"),
        ("Check whether maximum node cluster diameter is 64 in mesh network", "CONTRADICTED", "STRUCTURED_VERIFICATION"),
        ("Verify if the BM25 index k1 parameter is 1.5", "SUPPORTED", "STRUCTURED_VERIFICATION"),
        ("Verify whether the Starter Tier monthly price is $150", "SUPPORTED", "STRUCTURED_VERIFICATION"),
        ("Is the heartbeat interval in Raft consensus 50 milliseconds?", "SUPPORTED", "STRUCTURED_VERIFICATION"),
        ("Confirm if TCP port is 8443 in the mesh network", "SUPPORTED", "STRUCTURED_VERIFICATION"),
        ("Check whether maximum node cluster diameter is 16 in mesh network", "SUPPORTED", "STRUCTURED_VERIFICATION"),

        # 10 Honest Abstentions
        ("What is the capital city of planet Mars?", "ABSTAIN", "ABSTAIN"),
        ("Who is the prime minister of Mars?", "ABSTAIN", "ABSTAIN"),
        ("What is the speed of light in vacuum?", "ABSTAIN", "ABSTAIN"),
        ("Who wrote the play The Cherry Orchard?", "ABSTAIN", "ABSTAIN"),
        ("What is the atomic mass of Gold in the periodic table?", "ABSTAIN", "ABSTAIN"),
        ("What year was the construction of the Sydney Opera House completed?", "ABSTAIN", "ABSTAIN"),
        ("What is the distance from Earth to the Moon in kilometers?", "ABSTAIN", "ABSTAIN"),
        ("What is the legal voting age in Japan?", "ABSTAIN", "ABSTAIN"),
        ("Who was the president of the Moon in 1920?", "ABSTAIN", "ABSTAIN"),
        ("What is the temperature of the surface of the Sun?", "ABSTAIN", "ABSTAIN"),
    ]
    for idx, (q, exp, r) in enumerate(e2e_cases, 1):
        tasks.append({
            "id": f"e2e_{idx:03d}",
            "track": "e2e_assistant",
            "query": q,
            "expected_answer": exp,
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
        "dataset_name": "ChakrView Batches 1501-1540 Frozen Evaluation Suite (300 Tasks)",
        "milestone": "Batches 1501-1540",
        "total_tasks": len(suite),
        "sha256": file_hash,
        "tracks": {
            "Track_1_Arithmetic": 100,
            "Track_2_Structured_Verification": 100,
            "Track_3_Routing": 50,
            "Track_4_E2E_Assistant": 50,
        }
    }
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Dataset generated at {OUT_FILE} (SHA-256: {file_hash})")


if __name__ == "__main__":
    main()
