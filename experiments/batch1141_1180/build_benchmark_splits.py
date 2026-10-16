"""
Builds Development (30 items) and Held-Out (70 items) Benchmark Splits.
Location: experiments/batch1141_1180/build_benchmark_splits.py

Categories included:
- exact_entity: direct fact extraction
- paraphrase: semantic equivalence with altered phrasing
- synonym: uses alternate terms not in the document text
- distractor_competition: queries matching numbers/words shared across documents
- version_conflict: v0.6 vs v0.7 deprecations and version disambiguation
- multi_sentence: requires multi-sentence contextual evidence
- follow_up: conversational follow-up with pronouns ('it', 'that tier')
- multilingual_hinglish: queries in Hinglish / Hindi domain terms
- false_premise: queries with incorrect factual assumptions (refutation)
- abstention: unindexed topics requiring honest abstention
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

DATA_DIR = Path("experiments/batch1141_1180/data")

# 30 Development Queries (for analysis, threshold selection & parameter tuning)
DEV_QUERIES = [
    {
        "id": 1,
        "category": "exact_entity",
        "query": "What is the hidden dimension of ChakrMicro V0.6?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_core.md",
        "expected_substring": "384 units",
    },
    {
        "id": 2,
        "category": "exact_entity",
        "query": "What is the intermediate feed-forward layer size in V0.6?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_core.md",
        "expected_substring": "1536",
    },
    {
        "id": 3,
        "category": "paraphrase",
        "query": "What is the peak memory transfer throughput supported by the bus?",
        "expected_type": "ANSWER",
        "expected_source": "compute_memory_bus.md",
        "expected_substring": "68.2 GB/s",
    },
    {
        "id": 4,
        "category": "synonym",
        "query": "What cryptographic cipher protects wire transmission?",
        "expected_type": "ANSWER",
        "expected_source": "network_encryption_transit.md",
        "expected_substring": "TLS 1.3 with AES-256-GCM",
    },
    {
        "id": 5,
        "category": "distractor_competition",
        "query": "What is the maximum token context length supported in V0.6?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_core.md",
        "expected_substring": "512",
    },
    {
        "id": 6,
        "category": "version_conflict",
        "query": "Can ChakrMicro V0.6 checkpoints be loaded inside V0.7 runtimes?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_v07_draft.md",
        "expected_substring": "non-loadable in V0.7 runtimes",
    },
    {
        "id": 7,
        "category": "exact_entity",
        "query": "What is the monthly fee for the Starter subscription tier?",
        "expected_type": "ANSWER",
        "expected_source": "billing_subscription_tiers.md",
        "expected_substring": "150",
    },
    {
        "id": 8,
        "category": "exact_entity",
        "query": "How many worker processes are included in the Professional Tier?",
        "expected_type": "ANSWER",
        "expected_source": "billing_subscription_tiers.md",
        "expected_substring": "8",
    },
    {
        "id": 9,
        "category": "distractor_competition",
        "query": "What is the maximum allowed response delay for P1 severity incidents?",
        "expected_type": "ANSWER",
        "expected_source": "billing_support_tiers.md",
        "expected_substring": "15 minutes",
    },
    {
        "id": 10,
        "category": "distractor_competition",
        "query": "What is the target Recovery Time Objective for disaster recovery?",
        "expected_type": "ANSWER",
        "expected_source": "sop_disaster_recovery.md",
        "expected_substring": "15 minutes",
    },
    {
        "id": 11,
        "category": "paraphrase",
        "query": "What happens when an unauthenticated client tries to access the API?",
        "expected_type": "ANSWER",
        "expected_source": "security_auth_tokens.md",
        "expected_substring": "terminated immediately with HTTP 401",
    },
    {
        "id": 12,
        "category": "synonym",
        "query": "How long do signed authorization credentials remain valid?",
        "expected_type": "ANSWER",
        "expected_source": "security_auth_tokens.md",
        "expected_substring": "3600 seconds",
    },
    {
        "id": 13,
        "category": "follow_up",
        "context_query": "What is the monthly fee for the Starter subscription tier?",
        "query": "How many concurrent worker processes are included in that tier?",
        "expected_type": "ANSWER",
        "expected_source": "billing_subscription_tiers.md",
        "expected_substring": "2",
    },
    {
        "id": 14,
        "category": "follow_up",
        "context_query": "What encryption standard protects in-transit communication?",
        "query": "What symmetric cipher algorithm is used with it?",
        "expected_type": "ANSWER",
        "expected_source": "network_encryption_transit.md",
        "expected_substring": "AES-256-GCM",
    },
    {
        "id": 15,
        "category": "multilingual_hinglish",
        "query": "Starter tier ka monthly kharcha kitna hai?",
        "expected_type": "ANSWER",
        "expected_source": "billing_subscription_tiers.md",
        "expected_substring": "150",
    },
    {
        "id": 16,
        "category": "multilingual_hinglish",
        "query": "V0.6 model me kitne transformer layers hote hain?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_core.md",
        "expected_substring": "6",
    },
    {
        "id": 17,
        "category": "false_premise",
        "query": "Why does ChakrMicro require 128 transformer layers on mobile devices?",
        "expected_type": "REFUTE",
        "expected_source": "compute_arch_core.md",
        "expected_substring": None,
    },
    {
        "id": 18,
        "category": "false_premise",
        "query": "Why does the Starter Tier cost $5000 per month?",
        "expected_type": "REFUTE",
        "expected_source": "billing_subscription_tiers.md",
        "expected_substring": None,
    },
    {
        "id": 19,
        "category": "abstention",
        "query": "What is the population of Tokyo in 2026?",
        "expected_type": "ABSTAIN",
        "expected_source": None,
        "expected_substring": None,
    },
    {
        "id": 20,
        "category": "abstention",
        "query": "What is the quantum coherence time of the optical qubit bus?",
        "expected_type": "ABSTAIN",
        "expected_source": None,
        "expected_substring": None,
    },
    {
        "id": 21,
        "category": "exact_entity",
        "query": "What instruction set provides AVX2 vector extensions for x86 CPUs?",
        "expected_type": "ANSWER",
        "expected_source": "compute_cpu_simd.md",
        "expected_substring": "AVX2",
    },
    {
        "id": 22,
        "category": "paraphrase",
        "query": "How many threads are assigned to each CUDA streaming multiprocessor block?",
        "expected_type": "ANSWER",
        "expected_source": "compute_accelerator_cuda.md",
        "expected_substring": "256",
    },
    {
        "id": 23,
        "category": "synonym",
        "query": "What is the filesystem permission mask applied to stored session histories?",
        "expected_type": "ANSWER",
        "expected_source": "storage_disk_engine.md",
        "expected_substring": "0600",
    },
    {
        "id": 24,
        "category": "distractor_competition",
        "query": "How many representative domain tokens are used to calibrate quantization?",
        "expected_type": "ANSWER",
        "expected_source": "compute_quantization.md",
        "expected_substring": "512",
    },
    {
        "id": 25,
        "category": "version_conflict",
        "query": "What is the context window size of the experimental V0.7 model?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_v07_draft.md",
        "expected_substring": "1024",
    },
    {
        "id": 26,
        "category": "exact_entity",
        "query": "On which day of the month are invoices created?",
        "expected_type": "ANSWER",
        "expected_source": "billing_payment_terms.md",
        "expected_substring": "1st",
    },
    {
        "id": 27,
        "category": "paraphrase",
        "query": "What percentage billing rebate applies if monthly availability falls below 99.9%?",
        "expected_type": "ANSWER",
        "expected_source": "billing_sla_commitments.md",
        "expected_substring": "15%",
    },
    {
        "id": 28,
        "category": "abstention",
        "query": "Who is the CEO of Microsoft in 2026?",
        "expected_type": "ABSTAIN",
        "expected_source": None,
        "expected_substring": None,
    },
    {
        "id": 29,
        "category": "false_premise",
        "query": "Why does ChakrMicro use 48 attention heads on CPU?",
        "expected_type": "REFUTE",
        "expected_source": "compute_arch_core.md",
        "expected_substring": None,
    },
    {
        "id": 30,
        "category": "multilingual_hinglish",
        "query": "Production database me token expiration time kitna hai?",
        "expected_type": "ANSWER",
        "expected_source": "security_auth_tokens.md",
        "expected_substring": "3600 seconds",
    },
]


# 70 Frozen Held-Out Test Queries
HELD_OUT_QUERIES = [
    # 1-10: Exact Entities
    {
        "id": 101,
        "category": "exact_entity",
        "query": "What is the parameter count of ChakrMicro V0.6?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_core.md",
        "expected_substring": "10.8 million",
    },
    {
        "id": 102,
        "category": "exact_entity",
        "query": "What is the minimum physical RAM required for CPU inference?",
        "expected_type": "ANSWER",
        "expected_source": "compute_memory_bus.md",
        "expected_substring": "1.5 GB",
    },
    {
        "id": 103,
        "category": "exact_entity",
        "query": "What minimum compute capability is required for CUDA acceleration?",
        "expected_type": "ANSWER",
        "expected_source": "compute_accelerator_cuda.md",
        "expected_substring": "7.5",
    },
    {
        "id": 104,
        "category": "exact_entity",
        "query": "What is the thread pool ceiling for CPU execution?",
        "expected_type": "ANSWER",
        "expected_source": "compute_cpu_simd.md",
        "expected_substring": "4",
    },
    {
        "id": 105,
        "category": "exact_entity",
        "query": "What TCP port handles data transmission in the mesh network?",
        "expected_type": "ANSWER",
        "expected_source": "network_mesh_topology.md",
        "expected_substring": "8443",
    },
    {
        "id": 106,
        "category": "exact_entity",
        "query": "What is the heartbeat interval in Raft consensus?",
        "expected_type": "ANSWER",
        "expected_source": "network_consensus_raft.md",
        "expected_substring": "50 milliseconds",
    },
    {
        "id": 107,
        "category": "exact_entity",
        "query": "What is the maximum capacity of the in-memory embedding cache?",
        "expected_type": "ANSWER",
        "expected_source": "storage_caching_layer.md",
        "expected_substring": "10000",
    },
    {
        "id": 108,
        "category": "exact_entity",
        "query": "What is the term saturation parameter k1 in the BM25 index?",
        "expected_type": "ANSWER",
        "expected_source": "storage_bm25_index.md",
        "expected_substring": "1.5",
    },
    {
        "id": 109,
        "category": "exact_entity",
        "query": "What is the length normalization parameter b in BM25?",
        "expected_type": "ANSWER",
        "expected_source": "storage_bm25_index.md",
        "expected_substring": "0.75",
    },
    {
        "id": 110,
        "category": "exact_entity",
        "query": "What is the maximum single document size in the BLOB store?",
        "expected_type": "ANSWER",
        "expected_source": "storage_blob_store.md",
        "expected_substring": "10485760 bytes",
    },

    # 11-20: Paraphrases
    {
        "id": 111,
        "category": "paraphrase",
        "query": "How many query attention heads does ChakrMicro incorporate?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_core.md",
        "expected_substring": "6",
    },
    {
        "id": 112,
        "category": "paraphrase",
        "query": "What is the maximum throughput across the memory interconnect?",
        "expected_type": "ANSWER",
        "expected_source": "compute_memory_bus.md",
        "expected_substring": "68.2 GB/s",
    },
    {
        "id": 113,
        "category": "paraphrase",
        "query": "How frequently are telemetry metrics harvested by the collector?",
        "expected_type": "ANSWER",
        "expected_source": "compute_telemetry.md",
        "expected_substring": "1000 milliseconds",
    },
    {
        "id": 114,
        "category": "paraphrase",
        "query": "What speedup is achieved on CPU by fusing kernels into linear layers?",
        "expected_type": "ANSWER",
        "expected_source": "compute_kernel_fusion.md",
        "expected_substring": "18.5%",
    },
    {
        "id": 115,
        "category": "paraphrase",
        "query": "How many participant machines can join a single cluster diameter?",
        "expected_type": "ANSWER",
        "expected_source": "network_mesh_topology.md",
        "expected_substring": "16",
    },
    {
        "id": 116,
        "category": "paraphrase",
        "query": "When does automatic re-keying take place on encrypted communication lines?",
        "expected_type": "ANSWER",
        "expected_source": "network_encryption_transit.md",
        "expected_substring": "24 hours",
    },
    {
        "id": 117,
        "category": "paraphrase",
        "query": "How long does a client socket remain idle before keepalive probes trigger?",
        "expected_type": "ANSWER",
        "expected_source": "network_socket_tuning.md",
        "expected_substring": "300 seconds",
    },
    {
        "id": 118,
        "category": "paraphrase",
        "query": "How many consecutive failed probes cause a node to be marked unhealthy?",
        "expected_type": "ANSWER",
        "expected_source": "network_load_balancer.md",
        "expected_substring": "3",
    },
    {
        "id": 119,
        "category": "paraphrase",
        "query": "What is the maximum allowed query frequency per minute for a client IP?",
        "expected_type": "ANSWER",
        "expected_source": "network_gateway_api.md",
        "expected_substring": "60 requests per minute",
    },
    {
        "id": 120,
        "category": "paraphrase",
        "query": "What dead tuple percentage triggers automated compaction on database files?",
        "expected_type": "ANSWER",
        "expected_source": "storage_compaction.md",
        "expected_substring": "25%",
    },

    # 21-30: Synonyms & Non-Overlapping Vocabulary
    {
        "id": 121,
        "category": "synonym",
        "query": "What vector dimension does the learned neural bi-encoder project into?",
        "expected_type": "ANSWER",
        "expected_source": "storage_vector_index.md",
        "expected_substring": "64",
    },
    {
        "id": 122,
        "category": "synonym",
        "query": "How long must security audit log entries be preserved before deletion?",
        "expected_type": "ANSWER",
        "expected_source": "security_audit_logging.md",
        "expected_substring": "365 calendar days",
    },
    {
        "id": 123,
        "category": "synonym",
        "query": "How many incorrect PIN attempts trigger instant zeroization of the root key?",
        "expected_type": "ANSWER",
        "expected_source": "security_key_management.md",
        "expected_substring": "5",
    },
    {
        "id": 124,
        "category": "synonym",
        "query": "Within what timeframe must security breaches be escalated internally?",
        "expected_type": "ANSWER",
        "expected_source": "security_incident_response.md",
        "expected_substring": "30 minutes",
    },
    {
        "id": 125,
        "category": "synonym",
        "query": "What is the maximum allowed remediation window for critical severity vulnerabilities?",
        "expected_type": "ANSWER",
        "expected_source": "security_vulnerability_sla.md",
        "expected_substring": "24 hours",
    },
    {
        "id": 126,
        "category": "synonym",
        "query": "What annual contract price applies to the Dedicated Cluster subscription?",
        "expected_type": "ANSWER",
        "expected_source": "billing_subscription_tiers.md",
        "expected_substring": "12,000",
    },
    {
        "id": 127,
        "category": "synonym",
        "query": "What payment duration is permitted after an invoice has been issued?",
        "expected_type": "ANSWER",
        "expected_source": "billing_payment_terms.md",
        "expected_substring": "Net 30 days",
    },
    {
        "id": 128,
        "category": "synonym",
        "query": "What monthly interest surcharge applies to delinquent invoices?",
        "expected_type": "ANSWER",
        "expected_source": "billing_payment_terms.md",
        "expected_substring": "8.5%",
    },
    {
        "id": 129,
        "category": "synonym",
        "query": "What availability percentage is guaranteed for hosted control planes?",
        "expected_type": "ANSWER",
        "expected_source": "billing_sla_commitments.md",
        "expected_substring": "99.95%",
    },
    {
        "id": 130,
        "category": "synonym",
        "query": "What fee is charged per month for each additional worker process?",
        "expected_type": "ANSWER",
        "expected_source": "billing_metered_usage.md",
        "expected_substring": "40",
    },

    # 31-40: Distractor Competition (Same numbers in different contexts)
    {
        "id": 131,
        "category": "distractor_competition",
        "query": "What is the maximum context length in tokens for ChakrMicro V0.6?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_core.md",
        "expected_substring": "512",
    },
    {
        "id": 132,
        "category": "distractor_competition",
        "query": "How many tokens are in the calibration dataset for model quantization?",
        "expected_type": "ANSWER",
        "expected_source": "compute_quantization.md",
        "expected_substring": "512",
    },
    {
        "id": 133,
        "category": "distractor_competition",
        "query": "What is the hidden dimension of the experimental V0.7 draft model?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_v07_draft.md",
        "expected_substring": "512",
    },
    {
        "id": 134,
        "category": "distractor_competition",
        "query": "What is the maximum response time for Severity 1 support requests?",
        "expected_type": "ANSWER",
        "expected_source": "billing_support_tiers.md",
        "expected_substring": "15 minutes",
    },
    {
        "id": 135,
        "category": "distractor_competition",
        "query": "What is the Recovery Time Objective during disaster recovery failover?",
        "expected_type": "ANSWER",
        "expected_source": "sop_disaster_recovery.md",
        "expected_substring": "15 minutes",
    },
    {
        "id": 136,
        "category": "distractor_competition",
        "query": "What billing credit is applied when uptime drops below 99.9%?",
        "expected_type": "ANSWER",
        "expected_source": "billing_sla_commitments.md",
        "expected_substring": "15%",
    },
    {
        "id": 137,
        "category": "distractor_competition",
        "query": "What percentage discount is awarded for a 2-year upfront commitment?",
        "expected_type": "ANSWER",
        "expected_source": "billing_enterprise_discounts.md",
        "expected_substring": "15%",
    },
    {
        "id": 138,
        "category": "distractor_competition",
        "query": "What is the monthly subscription price of the Starter Tier?",
        "expected_type": "ANSWER",
        "expected_source": "billing_subscription_tiers.md",
        "expected_substring": "150",
    },
    {
        "id": 139,
        "category": "distractor_competition",
        "query": "What is the lower bound timeout for Raft leader elections in milliseconds?",
        "expected_type": "ANSWER",
        "expected_source": "network_consensus_raft.md",
        "expected_substring": "150ms",
    },
    {
        "id": 140,
        "category": "distractor_competition",
        "query": "What is the feed-forward dimension of ChakrMicro V0.6?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_core.md",
        "expected_substring": "1536",
    },

    # 41-46: Version Conflicts
    {
        "id": 141,
        "category": "version_conflict",
        "query": "What is the hidden dimension of ChakrMicro V0.6?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_core.md",
        "expected_substring": "384",
    },
    {
        "id": 142,
        "category": "version_conflict",
        "query": "What is the hidden dimension in the V0.7 specification?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_v07_draft.md",
        "expected_substring": "512",
    },
    {
        "id": 143,
        "category": "version_conflict",
        "query": "Are V0.6 model checkpoints compatible with V0.7 runtimes?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_v07_draft.md",
        "expected_substring": "non-loadable in V0.7 runtimes",
    },
    {
        "id": 144,
        "category": "version_conflict",
        "query": "What is the intermediate feed-forward dimension in the V0.7 draft?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_v07_draft.md",
        "expected_substring": "2048",
    },
    {
        "id": 145,
        "category": "version_conflict",
        "query": "What Python versions are officially certified for deployment?",
        "expected_type": "ANSWER",
        "expected_source": "sop_runtime_environments.md",
        "expected_substring": "Python 3.10",
    },
    {
        "id": 146,
        "category": "version_conflict",
        "query": "Is Python 3.9 accepted in the certified deployment environment?",
        "expected_type": "ANSWER",
        "expected_source": "sop_runtime_environments.md",
        "expected_substring": "unsupported",
    },

    # 47-54: Follow-Up Questions (Conversational anaphora)
    {
        "id": 147,
        "category": "follow_up",
        "context_query": "What is the monthly fee for the Professional Tier?",
        "query": "How many concurrent worker processes are included in that tier?",
        "expected_type": "ANSWER",
        "expected_source": "billing_subscription_tiers.md",
        "expected_substring": "8",
    },
    {
        "id": 148,
        "category": "follow_up",
        "context_query": "What is the uptime target for hosted control planes?",
        "query": "What billing credit is deducted if it drops below 99.9%?",
        "expected_type": "ANSWER",
        "expected_source": "billing_sla_commitments.md",
        "expected_substring": "15%",
    },
    {
        "id": 149,
        "category": "follow_up",
        "context_query": "What is the token expiration window for OAuth2 tokens?",
        "query": "What HTTP status code is returned when that request is unauthenticated?",
        "expected_type": "ANSWER",
        "expected_source": "security_auth_tokens.md",
        "expected_substring": "401",
    },
    {
        "id": 150,
        "category": "follow_up",
        "context_query": "What encryption standard protects network communication?",
        "query": "What symmetric cipher algorithm is used with it?",
        "expected_type": "ANSWER",
        "expected_source": "network_encryption_transit.md",
        "expected_substring": "AES-256-GCM",
    },
    {
        "id": 151,
        "category": "follow_up",
        "context_query": "What is the hidden dimension of ChakrMicro V0.6?",
        "query": "How many transformer decoder layers does it have?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_core.md",
        "expected_substring": "6",
    },
    {
        "id": 152,
        "category": "follow_up",
        "context_query": "Where are database backup snapshots created?",
        "query": "Where is the offsite replica archive stored?",
        "expected_type": "ANSWER",
        "expected_source": "storage_backup_policy.md",
        "expected_substring": "offsite",
    },
    {
        "id": 153,
        "category": "follow_up",
        "context_query": "What latency threshold triggers a Severity 1 incident?",
        "query": "What is the maximum engineer response time for that incident rating?",
        "expected_type": "ANSWER",
        "expected_source": "billing_support_tiers.md",
        "expected_substring": "15 minutes",
    },
    {
        "id": 154,
        "category": "follow_up",
        "context_query": "What is the discount rate for a 2-year commitment?",
        "query": "What discount is provided for a 3-year commitment?",
        "expected_type": "ANSWER",
        "expected_source": "billing_enterprise_discounts.md",
        "expected_substring": "25%",
    },

    # 55-60: Multilingual / Hinglish
    {
        "id": 155,
        "category": "multilingual_hinglish",
        "query": "ChakrMicro V0.6 me intermediate feed-forward layer size kitna hota hai?",
        "expected_type": "ANSWER",
        "expected_source": "compute_arch_core.md",
        "expected_substring": "1536",
    },
    {
        "id": 156,
        "category": "multilingual_hinglish",
        "query": "Professional Tier subscription ka per month rate kya hai?",
        "expected_type": "ANSWER",
        "expected_source": "billing_subscription_tiers.md",
        "expected_substring": "450",
    },
    {
        "id": 157,
        "category": "multilingual_hinglish",
        "query": "P1 severity issue aane par engineer kitne time me respond karega?",
        "expected_type": "ANSWER",
        "expected_source": "billing_support_tiers.md",
        "expected_substring": "15 minutes",
    },
    {
        "id": 158,
        "category": "multilingual_hinglish",
        "query": "Production API pe unauthenticated access par konsa error status code milta hai?",
        "expected_type": "ANSWER",
        "expected_source": "security_auth_tokens.md",
        "expected_substring": "401",
    },
    {
        "id": 159,
        "category": "multilingual_hinglish",
        "query": "In-transit network communication me konsa encryption cipher use hota hai?",
        "expected_type": "ANSWER",
        "expected_source": "network_encryption_transit.md",
        "expected_substring": "AES-256-GCM",
    },
    {
        "id": 160,
        "category": "multilingual_hinglish",
        "query": "CPU par inference chalane ke liye minimum kitni physical RAM honi chahiye?",
        "expected_type": "ANSWER",
        "expected_source": "compute_memory_bus.md",
        "expected_substring": "1.5 GB",
    },

    # 61-65: False Premise Refutations
    {
        "id": 161,
        "category": "false_premise",
        "query": "Why does ChakrMicro V0.6 require 128 attention heads on CPU?",
        "expected_type": "REFUTE",
        "expected_source": "compute_arch_core.md",
        "expected_substring": None,
    },
    {
        "id": 162,
        "category": "false_premise",
        "query": "Why is the Starter subscription priced at $9,000 per month?",
        "expected_type": "REFUTE",
        "expected_source": "billing_subscription_tiers.md",
        "expected_substring": None,
    },
    {
        "id": 163,
        "category": "false_premise",
        "query": "Why does ChakrView allow arbitrary sudo execution in production?",
        "expected_type": "REFUTE",
        "expected_source": "security_sandbox_isolation.md",
        "expected_substring": None,
    },
    {
        "id": 164,
        "category": "false_premise",
        "query": "Why are database tokens configured to expire after 100 years?",
        "expected_type": "REFUTE",
        "expected_source": "security_auth_tokens.md",
        "expected_substring": None,
    },
    {
        "id": 165,
        "category": "false_premise",
        "query": "Why does the P1 severity SLA allow up to 48 hours for engineer response?",
        "expected_type": "REFUTE",
        "expected_source": "billing_support_tiers.md",
        "expected_substring": None,
    },

    # 66-70: Honest Abstentions (Unindexed Out-of-Domain)
    {
        "id": 166,
        "category": "abstention",
        "query": "What is the speed of sound in liquid helium at 1 Kelvin?",
        "expected_type": "ABSTAIN",
        "expected_source": None,
        "expected_substring": None,
    },
    {
        "id": 167,
        "category": "abstention",
        "query": "What is the orbital eccentricity of Jupiter moon Europa?",
        "expected_type": "ABSTAIN",
        "expected_source": None,
        "expected_substring": None,
    },
    {
        "id": 168,
        "category": "abstention",
        "query": "What is the current market capital of Apple in Japanese Yen?",
        "expected_type": "ABSTAIN",
        "expected_source": None,
        "expected_substring": None,
    },
    {
        "id": 169,
        "category": "abstention",
        "query": "What is the chemical composition of volcanic basalt on Mars?",
        "expected_type": "ABSTAIN",
        "expected_source": None,
        "expected_substring": None,
    },
    {
        "id": 170,
        "category": "abstention",
        "query": "Who won the FIFA World Cup football tournament in 1994?",
        "expected_type": "ABSTAIN",
        "expected_source": None,
        "expected_substring": None,
    },
]


def save_splits():
    dev_p = DATA_DIR / "dev_30_benchmark.json"
    held_out_p = DATA_DIR / "held_out_70_benchmark.json"
    
    dev_content = json.dumps(DEV_QUERIES, indent=2)
    held_out_content = json.dumps(HELD_OUT_QUERIES, indent=2)

    dev_p.write_text(dev_content, encoding="utf-8")
    held_out_p.write_text(held_out_content, encoding="utf-8")

    dev_sha = hashlib.sha256(dev_content.encode("utf-8")).hexdigest()
    held_out_sha = hashlib.sha256(held_out_content.encode("utf-8")).hexdigest()

    manifest = {
        "dev_split": {
            "path": str(dev_p),
            "count": len(DEV_QUERIES),
            "sha256": dev_sha,
        },
        "held_out_split": {
            "path": str(held_out_p),
            "count": len(HELD_OUT_QUERIES),
            "sha256": held_out_sha,
        },
    }

    manifest_p = DATA_DIR / "BENCHMARK_SPLIT_MANIFEST.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated Dev Split: {len(DEV_QUERIES)} items (SHA-256: {dev_sha})")
    print(f"Generated Held-Out Split: {len(HELD_OUT_QUERIES)} items (SHA-256: {held_out_sha})")
    print(f"Saved manifest to {manifest_p}")


if __name__ == "__main__":
    save_splits()
