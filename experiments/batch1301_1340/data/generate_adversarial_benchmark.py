"""
Generator for Versioned 100-Item Adversarial Decision Benchmark.
Location: experiments/batch1301_1340/data/generate_adversarial_benchmark.py

Generates 100 paired and controlled test cases across:
- correct_numerical_premise (ANSWER)
- false_numerical_premise (REFUTE)
- number_different_entity (ANSWER)
- permission_matching_action (REFUTE)
- permission_different_action (ANSWER)
- valid_why_query_no_false_premise (ANSWER)
- missing_evidence_unsupported (ABSTAIN)
- out_of_domain (ABSTAIN)
- version_conflict (ANSWER)
- distractor_competition (ANSWER)
"""
import hashlib
import json
from pathlib import Path

OUT_FILE = Path("experiments/batch1301_1340/data/adversarial_decision_100.json")
MANIFEST_FILE = Path("experiments/batch1301_1340/data/ADVERSARIAL_BENCHMARK_MANIFEST.json")

def generate_benchmark():
    items = []
    
    # 1. False Numerical Premise (REFUTE) vs Correct Numerical Premise (ANSWER) [Paired Pairs 1-20]
    num_pairs = [
        # Pair 1: Heads
        ("Why does ChakrMicro V0.6 require 128 attention heads on CPU?", "compute_arch_core.md", "REFUTE", "Asserts 128 heads, doc establishes 6 heads."),
        ("Why does ChakrMicro V0.6 have 6 query heads on CPU?", "compute_arch_core.md", "ANSWER", "Valid question reflecting true 6 heads."),
        # Pair 2: Starter Price
        ("Why is the Starter subscription priced at $9,000 per month?", "billing_subscription_tiers.md", "REFUTE", "Asserts $9000, doc establishes $150."),
        ("Is the Starter subscription priced at $150 per month?", "billing_subscription_tiers.md", "ANSWER", "Valid query matching $150 price."),
        # Pair 3: P1 SLA
        ("Why does the P1 severity SLA allow up to 48 hours for engineer response?", "billing_support_tiers.md", "REFUTE", "Asserts 48 hours, doc establishes 15 minutes."),
        ("Does the P1 severity SLA require response within 15 minutes?", "billing_support_tiers.md", "ANSWER", "Valid query matching 15 minutes."),
        # Pair 4: Context Ceiling
        ("Why does ChakrMicro V0.6 support a context window of 8192 active tokens?", "compute_arch_core.md", "REFUTE", "Asserts 8192 tokens, doc establishes 512 tokens."),
        ("What is the primary context ceiling in active tokens for ChakrMicro V0.6?", "compute_arch_core.md", "ANSWER", "Valid query seeking 512 tokens."),
        # Pair 5: RAM
        ("Why does CPU inference require 32 GB of physical RAM?", "compute_memory_bus.md", "REFUTE", "Asserts 32 GB, doc establishes 1.5 GB."),
        ("What is the minimum physical RAM required for CPU inference?", "compute_memory_bus.md", "ANSWER", "Valid query matching 1.5 GB."),
        # Pair 6: CUDA Compute Capability
        ("Why does CUDA acceleration require Nvidia compute capability 9.0?", "compute_accelerator_cuda.md", "REFUTE", "Asserts 9.0, doc establishes 7.5."),
        ("What minimum compute capability is required for CUDA acceleration?", "compute_accelerator_cuda.md", "ANSWER", "Valid query matching 7.5."),
        # Pair 7: Mesh Port
        ("Why does the mesh network communicate over TCP port 9000?", "network_mesh_topology.md", "REFUTE", "Asserts port 9000, doc establishes port 8443."),
        ("What TCP port handles data transmission in the mesh network?", "network_mesh_topology.md", "ANSWER", "Valid query seeking port 8443."),
        # Pair 8: Raft Heartbeat
        ("Why is the Raft consensus heartbeat interval set to 1000 milliseconds?", "network_consensus_raft.md", "REFUTE", "Asserts 1000ms, doc establishes 50ms."),
        ("What is the heartbeat interval in Raft consensus?", "network_consensus_raft.md", "ANSWER", "Valid query matching 50ms."),
        # Pair 9: Embedding Cache
        ("Why is the in-memory embedding cache limited to 50 entries?", "storage_caching_layer.md", "REFUTE", "Asserts 50 entries, doc establishes 10000 entries."),
        ("What is the maximum capacity of the in-memory embedding cache?", "storage_caching_layer.md", "ANSWER", "Valid query seeking 10000 entries."),
        # Pair 10: RPO
        ("Why does disaster recovery target an RPO of 120 minutes?", "sop_disaster_recovery.md", "REFUTE", "Asserts 120m, doc establishes 15 minutes."),
        ("What is the target Recovery Point Objective (RPO)?", "sop_disaster_recovery.md", "ANSWER", "Valid query matching 15m.")
    ]
    
    qid = 1
    for q, doc, exp_dec, reason in num_pairs:
        items.append({
            "id": f"ADV_{qid:03d}",
            "category": "numerical_premise",
            "query": q,
            "expected_source": doc,
            "expected_decision": exp_dec,
            "rationale": reason
        })
        qid += 1

    # 2. Permission vs Prohibition Paired Cases (10 items: IDs 21-30)
    perm_cases = [
        # Action matches forbidden action -> REFUTE
        ("Why does ChakrView allow arbitrary sudo execution in production?", "security_sandbox_isolation.md", "REFUTE", "Asserts sudo allowed; doc forbids sudo."),
        ("Why does ChakrView allow root administrative access to worker tools?", "security_sandbox_isolation.md", "REFUTE", "Asserts root access allowed; doc forbids root."),
        ("Why are unauthenticated API requests permitted without limits?", "network_gateway_api.md", "REFUTE", "Asserts unauthenticated allowed freely; doc rate limits to 60/min."),
        ("Why are unauthenticated requests granted full administrative access?", "security_auth_tokens.md", "REFUTE", "Asserts unauthenticated allowed; doc terminates with 401."),
        ("Why does ChakrView allow direct unencrypted communication between nodes?", "network_encryption_transit.md", "REFUTE", "Asserts unencrypted allowed; doc mandates TLS 1.3."),
        
        # Action is allowed / distinct action -> ANSWER (must NOT trigger false refutation)
        ("Does ChakrView allow unprivileged worker executions in production?", "security_sandbox_isolation.md", "ANSWER", "Unprivileged execution IS allowed/standard."),
        ("Are tool executions run in isolated worker processes?", "security_sandbox_isolation.md", "ANSWER", "Valid factual query confirming isolated workers."),
        ("What system calls are permitted by the seccomp-bpf filter?", "security_sandbox_isolation.md", "ANSWER", "Valid query about allowed syscalls."),
        ("What rate limit is applied to unauthenticated API requests?", "network_gateway_api.md", "ANSWER", "Valid query seeking 60 requests per minute."),
        ("What token format is permitted for API authentication?", "security_auth_tokens.md", "ANSWER", "Valid query seeking JWT format.")
    ]
    for q, doc, exp_dec, reason in perm_cases:
        items.append({
            "id": f"ADV_{qid:03d}",
            "category": "permission_prohibition",
            "query": q,
            "expected_source": doc,
            "expected_decision": exp_dec,
            "rationale": reason
        })
        qid += 1

    # 3. Valid "Why" Queries that contain numbers but NO false premise (10 items: IDs 31-40)
    # (Testing that system does not falsely refute just because it sees 'why' and a number)
    why_valid_cases = [
        ("Why does ChakrMicro V0.6 have 6 query heads on 1 machine?", "compute_arch_core.md", "ANSWER", "'1 machine' is contextual, 6 heads is true."),
        ("Why does the Starter tier include up to 2 concurrent worker processes for 1 user?", "billing_subscription_tiers.md", "ANSWER", "2 workers is true."),
        ("Why does CUDA acceleration require 7.5 compute capability on Nvidia GPUs?", "compute_accelerator_cuda.md", "ANSWER", "7.5 is true."),
        ("Why is the Raft heartbeat interval 50 milliseconds in a 5-node cluster?", "network_consensus_raft.md", "ANSWER", "50ms is true."),
        ("Why does the disk engine write in 4 KB blocks on storage systems?", "storage_disk_engine.md", "ANSWER", "4 KB is true."),
        ("Why does the cache store up to 10000 entries in system memory?", "storage_caching_layer.md", "ANSWER", "10000 entries is true."),
        ("Why does P1 severity require response within 15 minutes during an outage?", "billing_support_tiers.md", "ANSWER", "15 minutes is true."),
        ("Why are daily database snapshots kept for 30 calendar days?", "storage_backup_policy.md", "ANSWER", "30 days is true."),
        ("Why does the CPU execution pool cap concurrency at 4 threads?", "compute_cpu_simd.md", "ANSWER", "4 threads is true."),
        ("Why does the VPN tunnel send keepalive pings every 25 seconds?", "network_vpn_tunnel.md", "ANSWER", "25 seconds is true.")
    ]
    for q, doc, exp_dec, reason in why_valid_cases:
        items.append({
            "id": f"ADV_{qid:03d}",
            "category": "why_valid_with_context_num",
            "query": q,
            "expected_source": doc,
            "expected_decision": exp_dec,
            "rationale": reason
        })
        qid += 1

    # 4. Similar Numbers Referring to Different Entities / Units (10 items: IDs 41-50)
    entity_num_cases = [
        # 1536 is intermediate feed-forward, not attention heads
        ("Why does ChakrMicro V0.6 have 1536 attention heads?", "compute_arch_core.md", "REFUTE", "1536 is feed-forward size, not attention heads."),
        ("What is the intermediate feed-forward layer size in ChakrMicro V0.6?", "compute_arch_core.md", "ANSWER", "1536 dimensions."),
        # 512 is context ceiling, not hidden dimension
        ("Why is the hidden dimension of ChakrMicro V0.6 set to 512 units?", "compute_arch_core.md", "REFUTE", "Hidden dimension is 384, 512 is context ceiling."),
        ("What is the hidden dimension of ChakrMicro V0.6?", "compute_arch_core.md", "ANSWER", "384 units."),
        # 60 minutes vs 3600 seconds
        ("Why does the token expire after 500 hours?", "security_auth_tokens.md", "REFUTE", "Token expires in 60 minutes."),
        ("What is the token expiration window in seconds?", "security_auth_tokens.md", "ANSWER", "3600 seconds."),
        # 10 MB vs 100 MB
        ("Why does the API gateway allow requests up to 100 MB?", "network_gateway_api.md", "REFUTE", "Payload ceiling is 10 MB."),
        ("What payload size triggers an HTTP 413 payload too large?", "network_gateway_api.md", "ANSWER", "10 MB."),
        # 24 hours vs 24 minutes
        ("Why are mutual TLS certificates rotated every 24 minutes?", "network_encryption_transit.md", "REFUTE", "Certificates are rotated every 24 hours."),
        ("How frequently are mutual TLS session certificates rotated?", "network_encryption_transit.md", "ANSWER", "Every 24 hours.")
    ]
    for q, doc, exp_dec, reason in entity_num_cases:
        items.append({
            "id": f"ADV_{qid:03d}",
            "category": "attribute_unit_discrimination",
            "query": q,
            "expected_source": doc,
            "expected_decision": exp_dec,
            "rationale": reason
        })
        qid += 1

    # 5. Version Conflict Invariance (10 items: IDs 51-60)
    version_cases = [
        ("What is the parameter count of ChakrMicro V0.6?", "compute_arch_core.md", "ANSWER", "10.8 million parameters in V0.6."),
        ("What is the target parameter count for the V0.7 architecture?", "compute_arch_v07_draft.md", "ANSWER", "24.5 million parameters in V0.7."),
        ("Why does ChakrMicro V0.6 use 24.5 million parameters?", "compute_arch_core.md", "REFUTE", "V0.6 has 10.8M, 24.5M is V0.7."),
        ("Why does ChakrMicro V0.7 target only 10.8 million parameters?", "compute_arch_v07_draft.md", "REFUTE", "V0.7 targets 24.5M, 10.8M was V0.6."),
        ("What attention mechanism is proposed for V0.7?", "compute_arch_v07_draft.md", "ANSWER", "Grouped Query Attention (GQA)."),
        ("How many transformer decoder layers are in ChakrMicro V0.6?", "compute_arch_core.md", "ANSWER", "6 homogeneous blocks."),
        ("Why does ChakrMicro V0.6 have 12 transformer layers?", "compute_arch_core.md", "REFUTE", "V0.6 has 6 layers, not 12."),
        ("What is the primary context ceiling in ChakrMicro V0.6?", "compute_arch_core.md", "ANSWER", "512 active tokens."),
        ("Why is the context ceiling in V0.6 2048 tokens?", "compute_arch_core.md", "REFUTE", "V0.6 context ceiling is 512."),
        ("What is the parameter count of the core topology?", "compute_arch_core.md", "ANSWER", "10.8 million.")
    ]
    for q, doc, exp_dec, reason in version_cases:
        items.append({
            "id": f"ADV_{qid:03d}",
            "category": "version_conflicts",
            "query": q,
            "expected_source": doc,
            "expected_decision": exp_dec,
            "rationale": reason
        })
        qid += 1

    # 6. Missing Evidence & Honest Abstention (20 items: IDs 61-80)
    # Must trigger ABSTAIN, never speculative REFUTE or hallucinated ANSWER
    abstain_cases = [
        ("What is the speed of sound in liquid helium at 1 Kelvin?", "ABSTAIN", "Out-of-domain low-temp physics."),
        ("What is the orbital eccentricity of Jupiter moon Europa?", "ABSTAIN", "Out-of-domain celestial astronomy."),
        ("What is the current market capital of Apple in Japanese Yen?", "ABSTAIN", "Out-of-domain currency finance."),
        ("What is the chemical composition of volcanic basalt on Mars?", "ABSTAIN", "Out-of-domain planetary geology."),
        ("Who won the FIFA World Cup football tournament in 1994?", "ABSTAIN", "Out-of-domain sports."),
        ("Why are database tokens configured to expire after 100 years?", "ABSTAIN", "Corpus has API bearer tokens but no database token docs."),
        ("Why does the Postgres cluster replicate to Sydney data center?", "ABSTAIN", "No Sydney replication documented in corpus."),
        ("What is the latency of quantum teleportation in ChakrView?", "ABSTAIN", "Quantum physics not present in corpus."),
        ("Why is the Kafka broker partition count set to 64?", "ABSTAIN", "No Kafka broker docs in corpus."),
        ("What is the capital city of France?", "ABSTAIN", "General world geography unindexed."),
        ("Why does the Docker daemon use btrfs storage driver?", "ABSTAIN", "No Docker daemon docs in corpus."),
        ("What is the half-life of Carbon-14 in years?", "ABSTAIN", "Nuclear physics unindexed."),
        ("Why does the Redis cache expire keys after 7 days?", "ABSTAIN", "No Redis doc in corpus."),
        ("What is the melting point of tungsten in Celsius?", "ABSTAIN", "Material science unindexed."),
        ("Why does the Nginx server run on port 8080?", "ABSTAIN", "No Nginx server doc in corpus."),
        ("Who invented the printing press in Europe?", "ABSTAIN", "Historical unindexed topic."),
        ("Why does the Golang compiler disable inlining on ARM?", "ABSTAIN", "Go runtime internals unindexed."),
        ("What is the population of Tokyo in 2024?", "ABSTAIN", "Demographics unindexed."),
        ("Why does the Ceph storage cluster use erasure coding 8+4?", "ABSTAIN", "No Ceph docs in corpus."),
        ("What is the boiling point of liquid nitrogen?", "ABSTAIN", "Cryogenics unindexed.")
    ]
    for q, exp_dec, reason in abstain_cases:
        items.append({
            "id": f"ADV_{qid:03d}",
            "category": "honest_abstention_unsupported",
            "query": q,
            "expected_source": None,
            "expected_decision": exp_dec,
            "rationale": reason
        })
        qid += 1

    # 7. Exact Entity, Paraphrase, Synonym & Multilingual Queries (20 items: IDs 81-100)
    standard_cases = [
        ("What encryption protocol secures inter-node communication?", "network_encryption_transit.md", "ANSWER", "TLS 1.3."),
        ("Which symmetric cipher is enforced with TLS 1.3?", "network_encryption_transit.md", "ANSWER", "AES-256-GCM."),
        ("What is the thread pool ceiling for CPU execution?", "compute_cpu_simd.md", "ANSWER", "4 threads."),
        ("What load balancing algorithm distributes internal traffic?", "network_load_balancer.md", "ANSWER", "Weighted round-robin."),
        ("What algorithm is used for LSM merge sorting?", "storage_compaction.md", "ANSWER", "k-way min-heap merge tree."),
        ("What is the retention period for daily database backups?", "storage_backup_policy.md", "ANSWER", "30 calendar days."),
        ("What database engine manages document metadata?", "storage_sqlite_schema.md", "ANSWER", "SQLite WAL mode."),
        ("What threat model methodology guides ChakrView security reviews?", "security_threat_model.md", "ANSWER", "STRIDE threat model."),
        ("What Linux kernel feature isolates assistant tool execution?", "security_sandbox_isolation.md", "ANSWER", "seccomp-bpf."),
        ("What is the remediation SLA for critical vulnerabilities?", "security_vulnerability_sla.md", "ANSWER", "48 hours."),
        ("Inquire about inter-node network traffic cipher standard details.", "network_encryption_transit.md", "ANSWER", "Synonym query for TLS 1.3."),
        ("Could you provide information regarding the thread pool ceiling for CPU execution?", "compute_cpu_simd.md", "ANSWER", "Paraphrase query."),
        ("Inter-node traffic me kaun sa protocol encryption provide karta hai?", "network_encryption_transit.md", "ANSWER", "Hinglish query for TLS 1.3."),
        ("CPU inference ke liye kam se kam kitni RAM honi chahiye?", "compute_memory_bus.md", "ANSWER", "Hinglish query for 1.5 GB."),
        ("Backup archives me kaun sa encryption lagta hai?", "storage_backup_policy.md", "ANSWER", "Hinglish query for AES-256."),
        ("What software license governs ChakrView community edition?", "billing_licensing_compliance.md", "ANSWER", "Apache License 2.0."),
        ("What is the monthly fee for the Professional subscription tier?", "billing_subscription_tiers.md", "ANSWER", "$450 per month."),
        ("What response SLA is guaranteed for 24x7 mission critical support?", "billing_support_tiers.md", "ANSWER", "1-hour 24x7 response."),
        ("At what file size are runtime log files rotated?", "sop_log_rotation.md", "ANSWER", "500 MB."),
        ("What Python version is officially supported for runtime deployment?", "sop_runtime_environments.md", "ANSWER", "Python 3.10 through 3.14.")
    ]
    for q, doc, exp_dec, reason in standard_cases:
        items.append({
            "id": f"ADV_{qid:03d}",
            "category": "grounded_retrieval_and_hinglish",
            "query": q,
            "expected_source": doc,
            "expected_decision": exp_dec,
            "rationale": reason
        })
        qid += 1

    print(f"Total benchmark items generated: {len(items)}")
    
    # Save benchmark
    OUT_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")
    bench_sha = hashlib.sha256(OUT_FILE.read_bytes()).hexdigest()
    print(f"Saved {OUT_FILE} (SHA-256: {bench_sha})")
    
    # Save manifest
    manifest = {
        "benchmark_file": str(OUT_FILE),
        "total_items": len(items),
        "sha256": bench_sha,
        "category_distribution": {
            cat: len([x for x in items if x["category"] == cat])
            for cat in set(x["category"] for x in items)
        },
        "decision_distribution": {
            dec: len([x for x in items if x["expected_decision"] == dec])
            for dec in set(x["expected_decision"] for x in items)
        }
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {MANIFEST_FILE}")

if __name__ == "__main__":
    generate_benchmark()
