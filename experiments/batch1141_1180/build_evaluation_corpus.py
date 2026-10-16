"""
Corpus and Benchmark Builder for Batches 1141-1180.
Location: experiments/batch1141_1180/build_evaluation_corpus.py

Builds:
1. 52 controlled documents (~150-250 chunks) across 6 operational domains:
   - Compute architecture & accelerators
   - Networking & distributed consensus
   - Storage, caching, and database engines
   - Security policies, cryptography, and access control
   - Enterprise billing, licensing, and compliance
   - Standard Operating Procedures & disaster recovery
2. Includes deliberate distractors:
   - Same numbers (e.g. 512, 1536, 15, 600) across unrelated contexts.
   - Competing version documents (v0.6 vs v0.7 deprecations).
   - Domain-specific paraphrases and synonyms.
3. Benchmark splits:
   - Development split (30 queries) for parameter/threshold tuning.
   - Held-out test split (70 queries) strictly frozen for final evaluation.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
from typing import Dict, List, Any


CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")
DATA_DIR = Path("experiments/batch1141_1180/data")


DOCUMENTS = {
    # DOMAIN 1: Compute Architecture & Microkernels (Docs 1-10)
    "compute_arch_core.md": (
        "# ChakrMicro V0.6 Core Topology\n\n"
        "## 1. Dimensional Architecture\n"
        "- Hidden dimension: 384 units.\n"
        "- Multi-head attention count: 6 query heads and 2 key-value heads (grouped-query attention).\n"
        "- Intermediate feed-forward layer size: 1536 dimensions.\n"
        "- Transformer decoder layers: 6 homogeneous blocks.\n"
        "- Primary context ceiling: 512 active tokens.\n"
        "- Weight parameter volume: 10.8 million trainable float parameters.\n"
    ),
    "compute_arch_v07_draft.md": (
        "# ChakrMicro V0.7 Experimental Specification\n\n"
        "## 1. Upgraded Hyperparameters\n"
        "- Experimental hidden dimension: 512 units.\n"
        "- Attention heads: 8 full query heads.\n"
        "- Intermediate feed-forward layer size: 2048 dimensions.\n"
        "- Context window ceiling: 1024 tokens.\n"
        "- Deprecation Notice: V0.6 384-dimension checkpoints are non-loadable in V0.7 runtimes.\n"
    ),
    "compute_memory_bus.md": (
        "# Memory Bus & Interconnect Protocols\n\n"
        "## 1. Memory Subsystem\n"
        "- Bus architecture: Dual-channel unified memory bus.\n"
        "- Memory bandwidth ceiling: 68.2 GB/s peak transfer throughput.\n"
        "- Minimum operational RAM allocation for CPU inference: 1.5 GB physical memory.\n"
        "- Cache line size: 64 bytes aligned across L1, L2, and L3 caches.\n"
    ),
    "compute_accelerator_cuda.md": (
        "# CUDA Accelerator Runtime\n\n"
        "## 1. GPU Compute Specifications\n"
        "- Minimum CUDA compute capability: 7.5 (Turing architecture or newer).\n"
        "- Numerical precision execution mode: FP16 and BF16 tensor core acceleration.\n"
        "- Kernel block dimension: 256 threads per streaming multiprocessor block.\n"
        "- VRAM minimum reservation: 2.0 GB dedicated video memory.\n"
    ),
    "compute_accelerator_metal.md": (
        "# Apple Silicon Metal Performance Shaders\n\n"
        "## 1. Metal Runtime Details\n"
        "- Backend identifier: MPSGraph execution engine.\n"
        "- Supported precision format: Float16 native compute.\n"
        "- Unified memory pool allocation cap: 75% of available system unified RAM.\n"
    ),
    "compute_cpu_simd.md": (
        "# CPU Vectorization and SIMD Dispatch\n\n"
        "## 1. Instruction Sets\n"
        "- Primary x86 vector extension: AVX2 with FMA3 instruction interleaving.\n"
        "- Fallback x86 instruction set: SSE4.2 vector instructions.\n"
        "- ARM vectorization: ARM Neon 128-bit SIMD registers.\n"
        "- Thread pool worker ceiling for CPU execution: 4 physical worker threads.\n"
    ),
    "compute_quantization.md": (
        "# Quantization and Weight Compression\n\n"
        "## 1. Compression Schemes\n"
        "- Static linear quantization: INT8 symmetric per-channel weights.\n"
        "- Activation clipping range: Absolute magnitude bounded at [-6.0, 6.0].\n"
        "- Calibration dataset footprint: Exactly 512 representative domain tokens.\n"
        "- Quantized checkpoint file suffix: `.q8.pt`.\n"
    ),
    "compute_scheduler.md": (
        "# Local Job Scheduler and Task Prioritization\n\n"
        "## 1. Priority Queuing\n"
        "- Maximum queue depth: 64 concurrent pending tasks.\n"
        "- High-priority preemption latency: 50 milliseconds.\n"
        "- Default worker thread idle sleep interval: 5 milliseconds.\n"
    ),
    "compute_telemetry.md": (
        "# Runtime Telemetry and Metrics Harvesting\n\n"
        "## 1. Metrics Collection\n"
        "- Sampling frequency: 1000 milliseconds (1 second interval).\n"
        "- Memory high-watermark threshold: 85% total system RAM trigger.\n"
        "- Metrics export format: OpenMetrics plaintext telemetry line format.\n"
    ),
    "compute_kernel_fusion.md": (
        "# Kernel Fusion and Graph Compilation\n\n"
        "## 1. Optimization Passes\n"
        "- Fused operations: LayerNorm plus Linear projection kernel fusion.\n"
        "- Activation fusion: GELU activation folded directly into feed-forward linear layers.\n"
        "- Graph optimization speedup: 18.5% median latency reduction on CPU.\n"
    ),

    # DOMAIN 2: Networking & Distributed Consensus (Docs 11-19)
    "network_mesh_topology.md": (
        "# Distributed Mesh Network Topology\n\n"
        "## 1. Cluster Interconnect\n"
        "- Peer discovery mechanism: mDNS multicast beacon on port 5353.\n"
        "- Control plane transport protocol: gRPC over HTTP/2.\n"
        "- Data transmission port: TCP port 8443.\n"
        "- Maximum node cluster diameter: 16 distributed participant nodes.\n"
    ),
    "network_encryption_transit.md": (
        "# In-Transit Cryptographic Protocols\n\n"
        "## 1. Wire Encryption Standards\n"
        "- In-transit communication is strictly encrypted via TLS 1.3 with AES-256-GCM.\n"
        "- Session re-keying threshold: Every 24 hours or after transmitting 100 GB of traffic.\n"
        "- Certificate authority verification: Mutual TLS (mTLS) with custom X.509 certificates.\n"
    ),
    "network_consensus_raft.md": (
        "# Raft Consensus Implementation\n\n"
        "## 1. Consensus Parameters\n"
        "- Leader election timeout: Bounded randomly between 150ms and 300ms.\n"
        "- Heartbeat interval: Exactly 50 milliseconds.\n"
        "- Quorum requirement: Strict majority formula floor(N/2) + 1 active nodes.\n"
    ),
    "network_load_balancer.md": (
        "# Traffic Routing and Reverse Proxy\n\n"
        "## 1. Routing Algorithms\n"
        "- Primary load-balancing algorithm: Weighted round-robin based on CPU capacity.\n"
        "- Health probe endpoint: `/healthz` HTTP GET probe.\n"
        "- Health probe failure threshold: 3 consecutive failed probes.\n"
    ),
    "network_socket_tuning.md": (
        "# Socket Buffer and Kernel TCP Tuning\n\n"
        "## 1. TCP Parameters\n"
        "- Socket receive buffer ceiling: 4194304 bytes (4 MB buffer).\n"
        "- TCP keepalive idle time: 300 seconds.\n"
        "- TCP SYN retry maximum: 3 attempts.\n"
    ),
    "network_gateway_api.md": (
        "# External Gateway API Specifications\n\n"
        "## 1. API Endpoints\n"
        "- Ingestion endpoint: `POST /v1/documents/ingest`.\n"
        "- Query endpoint: `POST /v1/assistant/query`.\n"
        "- Rate limit threshold: 60 requests per minute per authenticated client IP.\n"
    ),
    "network_dns_resolution.md": (
        "# Internal Name Resolution\n\n"
        "## 1. DNS Settings\n"
        "- Cluster local domain: `cluster.local`.\n"
        "- DNS cache Time-To-Live (TTL): 30 seconds.\n"
        "- Upstream fallback resolver: 127.0.0.1:853 over DNS-over-TLS.\n"
    ),
    "network_firewall_rules.md": (
        "# Firewall Configuration & Port Matrix\n\n"
        "## 1. Permitted Inbound Traffic\n"
        "- Inbound port 8443: Encrypted API traffic.\n"
        "- Inbound port 9090: Prometheus telemetry scraping.\n"
        "- Blocked ports: All outbound and inbound UDP traffic except port 5353 mDNS.\n"
    ),
    "network_vpn_tunnel.md": (
        "# Site-to-Site WireGuard Tunneling\n\n"
        "## 1. Tunnel Specifications\n"
        "- Tunnel protocol: WireGuard ChaCha20-Poly1305 encapsulation.\n"
        "- Keepalive interval: 25 seconds.\n"
        "- Tunnel MTU setting: 1420 bytes.\n"
    ),

    # DOMAIN 3: Storage, Database & Caching (Docs 20-27)
    "storage_disk_engine.md": (
        "# Persistent Disk Storage Engine\n\n"
        "## 1. Storage Architecture\n"
        "- Primary persistent storage: Append-only log with SQLite index backing.\n"
        "- Storage block size: 4096 bytes per sector.\n"
        "- File permission mask on disk: Octal mode 0600 (read/write owner only).\n"
    ),
    "storage_caching_layer.md": (
        "# In-Memory Embedding Cache Architecture\n\n"
        "## 1. Caching Specifications\n"
        "- Cache eviction policy: Least Recently Used (LRU) with composite hash keys.\n"
        "- Maximum cache capacity: 10000 vector embeddings.\n"
        "- Cache key generation: SHA-256 hash of model identifier plus text content.\n"
        "- Stale cache invalidation trigger: Document SHA-256 modification or file deletion.\n"
    ),
    "storage_vector_index.md": (
        "# In-Memory Vector Search Index\n\n"
        "## 1. Vector Index Geometry\n"
        "- Vector indexing method: Flat exact cosine inner product search.\n"
        "- Normalized vector length: Unit L2 norm (magnitude 1.0).\n"
        "- Index dimension: 64 float elements for ChakrBiEncoder.\n"
    ),
    "storage_bm25_index.md": (
        "# BM25 Lexical Knowledge Index\n\n"
        "## 1. BM25 Hyperparameters\n"
        "- Term saturation parameter k1: 1.5.\n"
        "- Document length normalization parameter b: 0.75.\n"
        "- Minimum indexed token length: 2 characters.\n"
    ),
    "storage_sqlite_schema.md": (
        "# Metadata Database Schema\n\n"
        "## 1. Schema Tables\n"
        "- Table `documents`: Tracks doc_id, file_path, file_size_bytes, mtime, sha256.\n"
        "- Table `chunks`: Tracks chunk_id, doc_id, chunk_index, start_char, end_char, chunk_hash.\n"
        "- WAL mode: Write-Ahead Logging enabled with synchronous NORMAL.\n"
    ),
    "storage_blob_store.md": (
        "# Local Binary Large Object (BLOB) Store\n\n"
        "## 1. Blob Storage Limits\n"
        "- Maximum single document size: 10485760 bytes (10 MB ceiling).\n"
        "- Storage path directory: `var/storage/blobs`.\n"
        "- Blob hashing algorithm: SHA-256 content addressable storage.\n"
    ),
    "storage_compaction.md": (
        "# Storage Compaction and Vacuum SOP\n\n"
        "## 1. Compaction Triggers\n"
        "- Fragmentation threshold: 25% dead tuple volume triggers auto-vacuum.\n"
        "- Scheduled compaction time: Nightly at 02:00 UTC.\n"
        "- Temporary disk space headroom required: 50% free volume capacity.\n"
    ),
    "storage_backup_policy.md": (
        "# Database Backup and Snapshot Policy\n\n"
        "## 1. Snapshot Schedule\n"
        "- Automated snapshot frequency: Hourly local differential snapshots.\n"
        "- Daily full backup time: 00:00 UTC daily.\n"
        "- Offsite replica archive target: Encrypted offsite cold storage volume.\n"
    ),

    # DOMAIN 4: Security, Cryptography & Access Control (Docs 28-36)
    "security_sandbox_isolation.md": (
        "# Subprocess Sandbox Isolation Policy\n\n"
        "## 1. Isolation Guarantees\n"
        "- Subprocess isolation: All tool executions run in isolated unprivileged worker processes.\n"
        "- Administrative access: Root and sudo commands are permanently forbidden.\n"
        "- Allowed system calls: Bounded strictly by seccomp-bpf white-list filter.\n"
    ),
    "security_auth_tokens.md": (
        "# Authentication and Token Lifecycle\n\n"
        "## 1. Token Policy\n"
        "- Token format: Cryptographically signed OAuth2 JSON Web Tokens (JWT).\n"
        "- Token expiration window: Exactly 3600 seconds (60 minutes).\n"
        "- Unauthenticated API access: Terminated immediately with HTTP 401 Unauthorized.\n"
    ),
    "security_at_rest_encryption.md": (
        "# At-Rest Data Encryption Standards\n\n"
        "## 1. Encryption Methods\n"
        "- Disk block encryption: Transparent AES-256-XTS filesystem encryption.\n"
        "- Key rotation cycle: Rotating HMAC keys renewed every 90 days.\n"
        "- File permission restriction: Mode 0600 on all stored session histories.\n"
    ),
    "security_role_rbac.md": (
        "# Role-Based Access Control (RBAC) Matrix\n\n"
        "## 1. User Roles\n"
        "- Role `Admin`: Full permissions including ingestion, deletion, and index rebuild.\n"
        "- Role `Operator`: Read, query, and verify permissions.\n"
        "- Role `Viewer`: Read-only question asking without document management.\n"
    ),
    "security_audit_logging.md": (
        "# Security Audit Trail and Event Logging\n\n"
        "## 1. Audit Requirements\n"
        "- Event log destination: Append-only signed audit log at `var/logs/audit.log`.\n"
        "- Log retention minimum: 365 calendar days.\n"
        "- Monitored events: Authentication attempts, privilege elevations, document ingestion, and index pruning.\n"
    ),
    "security_key_management.md": (
        "# Hardware Security Module & Key Custody\n\n"
        "## 1. Master Key Hierarchy\n"
        "- Root master key: 256-bit entropy stored in hardware keystore.\n"
        "- Key wrapping algorithm: AES-KeyWrap (RFC 3394).\n"
        "- Zeroization procedure: Instant key erasure on 5 consecutive invalid PIN entries.\n"
    ),
    "security_threat_model.md": (
        "# Threat Modeling & Attack Surface\n\n"
        "## 1. Passive Document Containment\n"
        "- Untrusted content policy: All ingested documents are treated as untrusted passive data.\n"
        "- Prompt-injection resistance: Instructions inside retrieved documents never override system execution policies.\n"
        "- Tool invocation barrier: Document Q&A cannot trigger filesystem modifications or network egress.\n"
    ),
    "security_incident_response.md": (
        "# Security Breach Incident Response\n\n"
        "## 1. Response Protocol\n"
        "- Breach notification deadline: Internal escalation within 30 minutes of incident confirmation.\n"
        "- Containment action: Immediate network isolation of affected compute nodes.\n"
        "- Forensics memory dump path: `/var/crash/forensics.dmp`.\n"
    ),
    "security_vulnerability_sla.md": (
        "# Vulnerability Remediation SLAs\n\n"
        "## 1. Patch Timelines\n"
        "- Critical severity CVEs (CVSS 9.0+): Patch deployment within 24 hours.\n"
        "- High severity CVEs (CVSS 7.0-8.9): Patch deployment within 7 calendar days.\n"
        "- Medium severity CVEs (CVSS 4.0-6.9): Patch deployment within 30 calendar days.\n"
    ),

    # DOMAIN 5: Enterprise Billing, SLA & Licensing (Docs 37-44)
    "billing_subscription_tiers.md": (
        "# Enterprise Subscription Pricing & Tiers\n\n"
        "## 1. Tier Pricing Matrix\n"
        "- Starter Tier: $150 per month, includes up to 2 concurrent worker processes.\n"
        "- Professional Tier: $450 per month, includes up to 8 concurrent worker processes and priority telemetry.\n"
        "- Dedicated Cluster Tier: Custom annual contract starting at $12,000 per year with unlimited workers.\n"
    ),
    "billing_payment_terms.md": (
        "# Invoicing Cycles and Payment Conditions\n\n"
        "## 1. Payment Terms\n"
        "- Invoice generation schedule: 1st day of each calendar month.\n"
        "- Payment terms: Net 30 days from invoice generation date.\n"
        "- Late payment penalty: Annualized interest surcharge of 8.5% compounded monthly.\n"
    ),
    "billing_sla_commitments.md": (
        "# Service Level Agreement Availability Targets\n\n"
        "## 1. Uptime Guarantees\n"
        "- Platform availability target: 99.95% monthly uptime for hosted control planes.\n"
        "- Downtime financial credit: 15% billing credit applied if monthly uptime drops below 99.9%.\n"
        "- Severe outage credit: 50% billing credit applied if monthly uptime drops below 98.0%.\n"
    ),
    "billing_support_tiers.md": (
        "# Technical Support Response Time SLAs\n\n"
        "## 1. Support Severity Response Matrix\n"
        "- Severity 1 (P1 - Complete Outage): Maximum response time of 15 minutes.\n"
        "- Severity 2 (P2 - Major Degradation): Maximum response time of 2 hours.\n"
        "- Severity 3 (P3 - Minor Defect): Maximum response time of 24 hours.\n"
    ),
    "billing_refund_policy.md": (
        "# Refund and Early Cancellation Policy\n\n"
        "## 1. Cancellation Terms\n"
        "- Monthly plans: Cancellation effective at end of current billing cycle without prorated refund.\n"
        "- Annual plans: 30-day money-back satisfaction guarantee from initial contract signature.\n"
        "- Dispute resolution window: Written notice required within 60 days of invoice date.\n"
    ),
    "billing_metered_usage.md": (
        "# Overage and Metered Compute Rates\n\n"
        "## 1. Overage Fees\n"
        "- Additional worker process unit: $40 per month per additional worker process.\n"
        "- Storage overage rate: $0.10 per gigabyte per month above allocated plan quota.\n"
        "- Burst query surcharge: $0.005 per query beyond monthly allocation.\n"
    ),
    "billing_licensing_compliance.md": (
        "# Software License and Usage Rights\n\n"
        "## 1. License Restrictions\n"
        "- License grant: Commercial, non-exclusive, non-transferable local deployment license.\n"
        "- Maximum registered physical machines: 5 hardware machines per Starter license.\n"
        "- Source code redistribution: Redistribution of compiled or raw binaries is strictly prohibited.\n"
    ),
    "billing_enterprise_discounts.md": (
        "# Volume and Multi-Year Discount Structure\n\n"
        "## 1. Discount Rates\n"
        "- 2-year upfront commitment discount: 15% discount on total annual subscription.\n"
        "- 3-year upfront commitment discount: 25% discount on total subscription.\n"
        "- Academic / Non-profit institution discount: 30% reduction across all tiers.\n"
    ),

    # DOMAIN 6: SOPs, Release Guide & Operations (Docs 45-52)
    "sop_incident_management.md": (
        "# Incident Management Standard Operating Procedure\n\n"
        "## 1. Incident Severity Definitions\n"
        "- Severity 1 (P1): Production cluster completely unresponsive, latency exceeds 500ms for > 5% requests.\n"
        "- Severity 2 (P2): Redundancy lost but primary control plane operational.\n"
        "- Severity 3 (P3): Minor documentation discrepancy or non-blocking bug.\n"
    ),
    "sop_disaster_recovery.md": (
        "# Disaster Recovery & Failover Procedure\n\n"
        "## 1. Recovery Time and Point Objectives\n"
        "- Recovery Time Objective (RTO): Under 15 minutes to restore secondary node cluster.\n"
        "- Recovery Point Objective (RPO): Under 5 minutes of transactional data loss.\n"
        "- Failover trigger: Automatic DNS redirection after 3 consecutive missed heartbeat intervals.\n"
    ),
    "sop_document_reindex.md": (
        "# Document Ingestion and Re-Indexing Procedure\n\n"
        "## 1. Re-indexing Steps\n"
        "- Step 1: Scan target directory for files matching `.txt` and `.md` extensions.\n"
        "- Step 2: Compare file modification time and SHA-256 hash against metadata table.\n"
        "- Step 3: Delete orphaned chunks for modified or removed files.\n"
        "- Step 4: Re-chunk modified files at 800-character boundaries with 100-character overlap.\n"
    ),
    "sop_cluster_drain.md": (
        "# Node Maintenance and Drain Protocol\n\n"
        "## 1. Drain Sequencing\n"
        "- Step 1: Mark target compute node unschedulable using `chakr node cordon <node_id>`.\n"
        "- Step 2: Evict active worker processes to neighboring peer nodes.\n"
        "- Step 3: Wait for graceful termination timeout of 60 seconds before host shutdown.\n"
    ),
    "sop_release_versioning.md": (
        "# Release Versioning & Semantic Guidelines\n\n"
        "## 1. SemVer 2.0 Enforcement\n"
        "- Major version bump: Backwards-incompatible architecture or breaking API modifications.\n"
        "- Minor version bump: Backwards-compatible feature additions or performance optimizations.\n"
        "- Patch version bump: Backwards-compatible bug fixes, security patches, or documentation corrections.\n"
    ),
    "sop_runtime_environments.md": (
        "# Certified Deployment Platforms and Python Versions\n\n"
        "## 1. Supported Environments\n"
        "- Certified Python runtimes: Python 3.10, Python 3.11, Python 3.12, Python 3.13, and Python 3.14.\n"
        "- Deprecated / Unsupported versions: Python 3.8 and Python 3.9 are strictly unsupported.\n"
        "- Supported operating systems: Windows 11, Ubuntu 22.04 LTS, Ubuntu 24.04 LTS, and macOS 14+.\n"
    ),
    "sop_release_checklist.md": (
        "# Production Release Sign-Off Checklist\n\n"
        "## 1. Mandatory Release Gates\n"
        "- Regression suite pass rate: Exactly 100% test pass on certified platform.\n"
        "- Security scan clearance: Zero critical and zero high severity vulnerabilities.\n"
        "- Canonical checkpoint SHA-256 hash verification: Verified identical to baseline.\n"
        "- Documentation synchronization: All API docs updated to match release commit.\n"
    ),
    "sop_log_rotation.md": (
        "# System Log Rotation and Retention SOP\n\n"
        "## 1. Log Maintenance\n"
        "- Maximum individual log file size: 52428800 bytes (50 MB cap).\n"
        "- Rotated backup count: 5 compressed archive generations retained.\n"
        "- Compression format: gzip level 6 compression.\n"
    ),
}


def build_corpus_files():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for filename, content in DOCUMENTS.items():
        file_path = CORPUS_DIR / filename
        file_path.write_text(content, encoding="utf-8")
        manifest[filename] = {
            "size": len(content),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
    
    with open(DATA_DIR / "CORPUS_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated {len(DOCUMENTS)} documents in {CORPUS_DIR}")


if __name__ == "__main__":
    build_corpus_files()
