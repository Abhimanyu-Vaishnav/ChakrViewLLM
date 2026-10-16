"""
Build domain-adaptive contrastive retrieval dataset with hard negatives for ChakrBiEncoder.
Target: ~500 query-positive-hard_negative pairs from the 52-doc technical corpus.
Strict separation:
- Train split: 70% (~350 pairs)
- Dev split: 15% (~75 pairs)
- Test split: 15% (~75 pairs)
"""
import hashlib
import json
import random
from pathlib import Path

random.seed(42)

CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")
OUT_DIR = Path("experiments/batch1181_1220/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Define topic families and file associations
TOPIC_FAMILIES = {
    "storage": [
        "storage_backup_policy.md", "storage_blob_store.md", "storage_bm25_index.md",
        "storage_caching_layer.md", "storage_compaction.md", "storage_disk_engine.md",
        "storage_sqlite_schema.md", "storage_vector_index.md"
    ],
    "network": [
        "network_consensus_raft.md", "network_dns_resolution.md", "network_encryption_transit.md",
        "network_firewall_rules.md", "network_gateway_api.md", "network_load_balancer.md",
        "network_mesh_topology.md", "network_socket_tuning.md", "network_vpn_tunnel.md"
    ],
    "security": [
        "security_at_rest_encryption.md", "security_audit_logging.md", "security_auth_tokens.md",
        "security_incident_response.md", "security_key_management.md", "security_role_rbac.md",
        "security_sandbox_isolation.md", "security_threat_model.md", "security_vulnerability_sla.md"
    ],
    "compute": [
        "compute_accelerator_cuda.md", "compute_accelerator_metal.md", "compute_arch_core.md",
        "compute_arch_v07_draft.md", "compute_cpu_simd.md", "compute_kernel_fusion.md",
        "compute_memory_bus.md", "compute_quantization.md", "compute_scheduler.md", "compute_telemetry.md"
    ],
    "sop": [
        "sop_cluster_drain.md", "sop_disaster_recovery.md", "sop_document_reindex.md",
        "sop_incident_management.md", "sop_log_rotation.md", "sop_release_checklist.md",
        "sop_release_versioning.md", "sop_runtime_environments.md"
    ],
    "billing": [
        "billing_enterprise_discounts.md", "billing_licensing_compliance.md", "billing_metered_usage.md",
        "billing_payment_terms.md", "billing_refund_policy.md", "billing_sla_commitments.md",
        "billing_subscription_tiers.md", "billing_support_tiers.md"
    ]
}

def load_documents():
    docs = {}
    for f in CORPUS_DIR.glob("*.md"):
        content = f.read_text(encoding="utf-8")
        docs[f.name] = content
    return docs

def generate_contrastive_dataset():
    docs = load_documents()
    examples = []
    ex_id = 1

    # Hand-curated seed templates per doc with positive passage extract, query variations (exact, synonym, paraphrase, hinglish),
    # and hard-negative document from the same topic family.
    doc_fact_specs = {
        "storage_backup_policy.md": [
            ("What is the retention period for daily database backups?", "Daily snapshots are retained for 30 calendar days.", "retention schedule", "Backup ka retention kitne din ka hota hai?"),
            ("Where are geo-redundant backups replicated?", "Replicated cross-region to the secondary cold storage bunker.", "backup replication site", "Backups kahan replicate hote hain?"),
            ("What encryption is applied to cold backup archives?", "Cold backup archives are encrypted with AES-256.", "archive encryption standard", "Backup archives me kaun sa encryption lagta hai?")
        ],
        "storage_blob_store.md": [
            ("What is the maximum object size supported in the blob store?", "The blob store supports objects up to 5 TB in size.", "blob size limit", "Blob store me max object size kitna hai?"),
            ("How many replicas are maintained for blob chunks?", "Each blob chunk is synchronously replicated across 3 failure zones.", "blob redundancy count", "Blob chunks ke kitne replicas bante hain?"),
            ("What chunk size is used for multipart uploads?", "Multipart uploads partition files into 64 MB blocks.", "multipart chunk partition", "Multipart upload ka chunk block size kya hai?")
        ],
        "storage_bm25_index.md": [
            ("What are the default BM25 parameters k1 and b?", "The BM25 index uses k1=1.2 and b=0.75.", "BM25 tuning coefficients", "BM25 me k1 aur b ki default values kya hain?"),
            ("What tokenizer is used for lexical indexing?", "Lexical indexing uses the indigenous ChakrView regex word tokenizer.", "lexical index tokenizer", "Lexical index ke liye kaun sa tokenizer chalta hai?"),
            ("How are deleted documents handled in BM25?", "Deleted documents are marked with a bitmask tombstone and purged during compaction.", "tombstone deletion BM25", "Deleted docs BM25 me kaise handle hote hain?")
        ],
        "storage_caching_layer.md": [
            ("What is the maximum capacity of the in-memory embedding cache?", "The in-memory embedding cache has a maximum capacity of 10000 entries.", "embedding cache max capacity", "Embedding cache ki max entry limit kya hai?"),
            ("What eviction policy is used by the caching layer?", "The caching layer evicts items using Least Recently Used (LRU) policy.", "cache eviction mechanism", "Cache eviction ka kaun sa rule follow hota hai?"),
            ("What is the TTL for transient query embeddings?", "Query embeddings have a TTL of 300 seconds.", "query embedding cache lifetime", "Query embedding ka TTL kitna second hai?")
        ],
        "storage_compaction.md": [
            ("When does background compaction trigger?", "Compaction triggers automatically when deleted tombstone ratio exceeds 20%.", "compaction trigger threshold", "Background compaction kab start hota hai?"),
            ("What algorithm is used for LSM merge sorting?", "LSM merge sorting employs a k-way min-heap merge tree.", "compaction merge tree algorithm", "Compaction me merge sort ka algorithm kaun sa hai?")
        ],
        "storage_disk_engine.md": [
            ("What block size does the storage disk engine write?", "The disk engine writes in aligned 4 KB blocks.", "disk write block alignment", "Disk engine kitne KB blocks me data write karta hai?"),
            ("What write mode ensures durable persistence?", "Synchronous WAL mode ensures zero-loss durable persistence.", "durable persistence write mode", "Durable persistence ke liye kaun sa mode use hota hai?")
        ],
        "storage_sqlite_schema.md": [
            ("What database engine manages document metadata?", "Document metadata is tracked using SQLite WAL mode.", "metadata database engine", "Document metadata kaun sa database store karta hai?"),
            ("What table stores chunk embeddings and offsets?", "The chunk_records table stores document ID, start_char, end_char, and vector blob.", "chunk metadata storage table", "Chunk embeddings kis table me save hoti hain?")
        ],
        "storage_vector_index.md": [
            ("What vector similarity metric is standard in ChakrView?", "Cosine similarity is the canonical metric for vector indexing.", "vector distance measure", "Vector index me kaun sa similarity metric lagta hai?"),
            ("What is the vector dimension of ChakrBiEncoder?", "ChakrBiEncoder produces dense 64-dimensional normalized vectors.", "bi-encoder embedding dimensionality", "ChakrBiEncoder ka vector dimension kitna hota hai?")
        ],
        "network_consensus_raft.md": [
            ("What is the heartbeat interval in Raft consensus?", "The leader sends heartbeats every 50 milliseconds.", "Raft leader pulse rate", "Raft consensus me heartbeat interval kya hota hai?"),
            ("What is the election timeout range in Raft?", "Follower election timeout is randomized between 150ms and 300ms.", "Raft leader election delay window", "Election timeout range kitna milliseconds hota hai?"),
            ("How many nodes constitute a Raft quorum in a 5-node cluster?", "A 5-node cluster requires 3 affirmative votes for quorum.", "consensus quorum size", "5-node cluster me quorum ke liye kitne votes chahiye?")
        ],
        "network_dns_resolution.md": [
            ("What is the fallback DNS resolution timeout?", "DNS resolution times out after 2000 milliseconds before falling back.", "nameserver query limit", "DNS fallback timeout kitne ms ka hai?"),
            ("Which internal DNS daemon resolves cluster service names?", "CoreDNS handles internal service name resolution on port 53.", "cluster nameserver service", "Cluster service names ko kaun sa DNS resolve karta hai?")
        ],
        "network_encryption_transit.md": [
            ("What encryption protocol secures inter-node communication?", "Inter-node communication is strictly encrypted with TLS 1.3.", "network traffic cipher standard", "Inter-node traffic me kaun sa protocol encryption provide karta hai?"),
            ("Which symmetric cipher is enforced with TLS 1.3?", "The TLS 1.3 pipeline enforces AES-256-GCM cipher suites.", "symmetric encryption cipher", "TLS 1.3 ke saath kaun sa symmetric cipher algorithm use hota hai?"),
            ("How frequently are mutual TLS session certificates rotated?", "mTLS session certificates are rotated every 24 hours.", "certificate renewal interval", "mTLS certificates kitne ghante me rotate hote hain?")
        ],
        "network_firewall_rules.md": [
            ("What default policy is applied by the firewall?", "The firewall operates on a default-deny ingress policy.", "default ingress rule", "Firewall ka default ingress rule kya hai?"),
            ("Which TCP port is open for cluster administration?", "Cluster admin API is accessible only on TCP port 9443.", "administrative access port", "Admin access ke liye kaun sa port open hota hai?")
        ],
        "network_gateway_api.md": [
            ("What is the rate limit for unauthenticated API requests?", "Unauthenticated requests are limited to 60 requests per minute.", "public request rate cap", "Public API requests ka rate limit kitna hai?"),
            ("What payload size triggers an HTTP 413 payload too large?", "Requests exceeding 10 MB in body size receive HTTP 413.", "maximum payload size limit", "API gateway max kitne MB ka payload allow karta hai?")
        ],
        "network_load_balancer.md": [
            ("What load balancing algorithm distributes internal traffic?", "Internal traffic is distributed using weighted round-robin.", "traffic dispatch policy", "Traffic distribute karne ke liye kaun sa algorithm lagta hai?"),
            ("How many consecutive failed health checks mark a backend down?", "Backends are marked down after 3 consecutive health check failures.", "unhealthy backend failure threshold", "Backend ko down mark karne ke liye kitne failed checks lagte hain?")
        ],
        "network_mesh_topology.md": [
            ("What TCP port handles data transmission in the mesh network?", "Mesh data transmission operates on TCP port 8443.", "mesh network port", "Mesh network data transmission ke liye kaun sa port use karta hai?"),
            ("What overlay protocol encapsulates peer-to-peer frames?", "Mesh peers communicate over WireGuard encapsulated UDP frames.", "mesh encapsulation overlay", "Mesh network me packets kis protocol se encapsulate hote hain?")
        ],
        "network_socket_tuning.md": [
            ("What is the recommended SOMAXCONN backlog setting?", "Set net.core.somaxconn to 4096 for high connection concurrency.", "socket backlog queue limit", "Socket backlog somaxconn ki setting kya honi chahiye?"),
            ("What TCP congestion control algorithm is deployed?", "TCP BBR congestion control is enabled on all cluster routers.", "congestion control scheme", "Network routers me kaun sa congestion control deploy hai?")
        ],
        "network_vpn_tunnel.md": [
            ("What keepalive interval prevents VPN tunnel idle disconnects?", "VPN tunnels send keepalive pings every 25 seconds.", "tunnel keepalive frequency", "VPN tunnel disconnect na ho iske liye keepalive kitne second me jata hai?"),
            ("What MTU size avoids VPN packet fragmentation?", "The tunnel network interface is configured with an MTU of 1420 bytes.", "tunnel MTU sizing", "VPN packet fragmentation rokne ke liye MTU kya rakhte hain?")
        ],
        "security_at_rest_encryption.md": [
            ("What encryption algorithm secures persistent volumes at rest?", "Data at rest is encrypted with XTS-AES-256.", "storage encryption cipher", "Storage volumes par kaun sa encryption algorithm lagta hai?"),
            ("Where is the volume master key held?", "The master encryption key is held in an external Hardware Security Module (HSM).", "encryption key storage location", "Volume master key kahan par store hoti hai?")
        ],
        "security_audit_logging.md": [
            ("What audit log retention is required for compliance?", "Audit event logs must be retained immutable for 365 days.", "audit trail duration", "Audit logs ko kitne din tak retain karna zaroori hai?"),
            ("What hashing algorithm signs tamper-evident log blocks?", "Log chunks are chained using SHA-256 cryptographic hashes.", "tamper-evident log hash", "Audit logs ko tamper-proof banane ke liye kaun sa hash use hota hai?")
        ],
        "security_auth_tokens.md": [
            ("What is the lifespan of an API bearer token?", "API bearer tokens expire after 15 minutes.", "token expiration interval", "Bearer token kitne minute me expire hota hai?"),
            ("What algorithm signs JSON Web Tokens in ChakrView?", "JWTs are cryptographically signed using Ed25519 asymmetric keys.", "JWT signature algorithm", "JWT token sign karne ke liye kaun sa algorithm use hota hai?")
        ],
        "security_incident_response.md": [
            ("What is the target acknowledgment SLA for a P1 security incident?", "P1 security incidents must be acknowledged within 15 minutes.", "P1 incident response target", "P1 security issue ka acknowledgment time kitna hota hai?"),
            ("Who authorizes isolation of compromised nodes?", "The designated Incident Commander authorizes immediate node isolation.", "node quarantine authority", "Compromised node ko isolate karne ki permission kaun deta hai?")
        ],
        "security_key_management.md": [
            ("How frequently are envelope encryption data keys rotated?", "Data encryption keys (DEKs) are rotated automatically every 90 days.", "data key renewal schedule", "Data encryption keys kitne din me rotate hoti hain?"),
            ("What minimum key length is required for RSA keypairs?", "RSA keys must have a minimum length of 3072 bits.", "RSA key bit length requirement", "RSA key ki minimum length kitni honi chahiye?")
        ],
        "security_role_rbac.md": [
            ("What permissions are assigned to the Operator role?", "Operators have read and restart permissions but cannot alter IAM roles.", "operator role rights", "Operator role ko kya kya permissions milti hain?"),
            ("Which role has administrative permissions to rotate keys?", "The SecurityAdmin role holds exclusive authority to rotate root keys.", "security admin authority", "Root keys rotate karne ka right kis role ke paas hai?")
        ],
        "security_sandbox_isolation.md": [
            ("What Linux kernel feature isolates assistant tool execution?", "Assistant tools execute inside unprivileged seccomp-bpf namespaces.", "process sandbox sandbox technology", "Assistant tools ko isolate karne ke liye kaun sa kernel feature lagta hai?"),
            ("Are assistant runtime workers permitted outbound internet access?", "Runtime sandboxes have outbound network traffic completely disabled.", "sandbox internet connectivity", "Kya assistant runtime workers ko internet chalane ki permission hai?")
        ],
        "security_threat_model.md": [
            ("What threat model methodology guides ChakrView security reviews?", "Security reviews are structured according to the STRIDE threat model.", "threat categorization framework", "Security review kis threat model ke basis par hota hai?"),
            ("How are prompt injection payloads treated in ingested documents?", "Retrieved documents are treated strictly as untrusted passive data.", "prompt injection containment", "Documents me prompt injection attacks ko kaise treat kiya jata hai?")
        ],
        "security_vulnerability_sla.md": [
            ("What is the remediation SLA for critical vulnerabilities?", "Critical vulnerabilities must be remediated within 48 hours.", "critical CVE patching timeline", "Critical vulnerability ko patch karne ka SLA kitne ghante ka hai?"),
            ("What is the remediation window for high severity defects?", "High severity vulnerabilities have a 14-day remediation window.", "high severity SLA", "High severity defect ko theek karne ke liye kitne din milte hain?")
        ],
        "compute_accelerator_cuda.md": [
            ("What minimum compute capability is required for CUDA acceleration?", "CUDA acceleration requires Nvidia GPU compute capability 7.5 or higher.", "Nvidia hardware compatibility", "CUDA chalane ke liye minimum compute capability kitni honi chahiye?"),
            ("What precision format does the CUDA kernel use for inference?", "CUDA inference executes in mixed FP16 / BF16 precision.", "GPU numerical precision format", "CUDA kernel inference me kaun sa precision format use karta hai?")
        ],
        "compute_accelerator_metal.md": [
            ("What minimum Apple silicon chip is required for Metal acceleration?", "Metal acceleration requires Apple M1 silicon or later.", "Apple hardware support", "Metal acceleration ke liye kaun sa Apple processor chahiye?"),
            ("What unified memory buffer allocation is recommended for Metal?", "Configure Metal unified memory allocation to 70% of system RAM.", "Apple unified memory limit", "Metal ke liye unified memory buffer kitna recommend kiya gaya hai?")
        ],
        "compute_arch_core.md": [
            ("What is the parameter count of ChakrMicro V0.6?", "ChakrMicro V0.6 has exactly 10.8 million parameters.", "model size in parameters", "ChakrMicro V0.6 me kitne parameters hain?"),
            ("What is the hidden dimension d_model of ChakrMicro?", "The hidden dimension d_model is 128.", "hidden state size", "ChakrMicro ka hidden dimension d_model kitna hai?"),
            ("How many transformer layers are in ChakrMicro V0.6?", "ChakrMicro V0.6 consists of 4 transformer layers.", "transformer depth layer count", "ChakrMicro V0.6 me transformer ki kitni layers hain?")
        ],
        "compute_arch_v07_draft.md": [
            ("What is the target parameter count for the V0.7 architecture?", "The draft V0.7 architecture targets 24.5 million parameters.", "next version parameter budget", "V0.7 version ka target parameter count kitna hai?"),
            ("What attention mechanism is proposed for V0.7?", "V0.7 proposes Grouped Query Attention (GQA) with 4 key-value heads.", "proposed attention variant", "V0.7 draft me kaun sa attention mechanism propose hua hai?")
        ],
        "compute_cpu_simd.md": [
            ("What is the thread pool ceiling for CPU execution?", "The CPU execution thread pool is capped at 4 threads.", "CPU concurrency cap", "CPU par parallel execution ke liye kitne threads allow hain?"),
            ("Which instruction set accelerates CPU matrix multiplications?", "Matrix operations are accelerated using AVX2 and FMA instructions.", "SIMD vector instructions", "CPU matrix calculation ke liye kaun sa instruction set use hota hai?")
        ],
        "compute_kernel_fusion.md": [
            ("What operations are combined in the fused attention kernel?", "The fused attention kernel combines softmax, dropout, and matrix multiply.", "fused kernel stages", "Fused attention kernel me kaun se operations combine hote hain?"),
            ("What speedup does kernel fusion achieve on CPU?", "Kernel fusion yields a 1.35x speedup in CPU inference latency.", "fusion latency reduction", "CPU par kernel fusion se kitna speedup milta hai?")
        ],
        "compute_memory_bus.md": [
            ("What is the minimum physical RAM required for CPU inference?", "CPU inference requires a minimum of 1.5 GB physical RAM.", "RAM footprint floor", "CPU inference ke liye kam se kam kitni RAM honi chahiye?"),
            ("What memory bandwidth is saturated during embedding lookup?", "Embedding lookup saturates up to 25 GB/s memory bandwidth.", "memory bus transfer rate", "Embedding lookup ke time memory bandwidth kitni consume hoti hai?")
        ],
        "compute_quantization.md": [
            ("What integer precision is used for weight quantization?", "Model weights are quantized to INT8 symmetric integer format.", "quantized weight format", "Weights quantization me kaun sa integer format use hota hai?"),
            ("What calibration algorithm determines clipping thresholds?", "Quantization clipping thresholds are determined using KL-divergence calibration.", "quantization calibration technique", "Clipping thresholds nikalne ke liye kaun sa calibration lagta hai?")
        ],
        "compute_scheduler.md": [
            ("What scheduling policy dispatches concurrent user queries?", "Concurrent queries are dispatched using priority fair-share queues.", "query scheduling discipline", "Concurrent queries ko schedule karne ke liye kaun si policy lagti hai?"),
            ("What is the maximum queue wait timeout before request abortion?", "Queued requests abort after waiting 5000 milliseconds.", "queue timeout threshold", "Request queue me max kitne ms tak wait kar sakti hai?")
        ],
        "compute_telemetry.md": [
            ("What port exposes Prometheus metrics for the compute node?", "Compute telemetry metrics are exposed on HTTP port 9090.", "Prometheus metrics port", "Metrics scrape karne ke liye Prometheus ka port kya hai?"),
            ("How often are latency percentiles published to telemetry?", "Latency metrics are aggregated and published every 10 seconds.", "metrics reporting interval", "Latency percentiles kitne second me update hoti hain?")
        ],
        "sop_cluster_drain.md": [
            ("What command cordons a node before maintenance?", "Nodes are cordoned using the chakr-ctl node cordon command.", "node maintenance isolation command", "Maintenance se pehle node cordon karne ki command kya hai?"),
            ("What grace period is granted to active inference queries during drain?", "In-flight inference queries are allowed a 30-second graceful completion window.", "drain grace period", "Node drain karte waqt running queries ko kitna time milta hai?")
        ],
        "sop_disaster_recovery.md": [
            ("What is the target Recovery Point Objective (RPO)?", "The disaster recovery plan targets an RPO of 15 minutes.", "data loss recovery target", "Disaster recovery me RPO kitne minute ka hai?"),
            ("What is the target Recovery Time Objective (RTO)?", "The target RTO for complete cluster restore is 60 minutes.", "service restoration timeline", "Cluster restore karne ka target RTO kitna time hai?")
        ],
        "sop_document_reindex.md": [
            ("What CLI command initiates a full corpus re-indexing?", "Re-indexing is initiated with python -m chakrview.tools.reindex --force.", "corpus rebuild command", "Full reindex initiate karne ke liye kaun si command chalate hain?"),
            ("Does corpus re-indexing invalidate the embedding cache?", "Corpus re-indexing purges and invalidates stale cached embeddings.", "cache invalidation on reindex", "Kya re-indexing karne se purana cache invalidate hota hai?")
        ],
        "sop_incident_management.md": [
            ("What channel hosts active Sev-1 incident bridge calls?", "Sev-1 incidents operate on dedicated voice bridge channel #incident-war-room.", "incident communication channel", "Sev-1 incident ke liye kaun sa channel use hota hai?"),
            ("Within how many hours must a post-incident retrospective be completed?", "Post-mortems must be published within 72 hours of incident resolution.", "post-mortem deadline", "Incident ke baad post-mortem report kitne ghante me submit karni hoti hai?")
        ],
        "sop_log_rotation.md": [
            ("At what file size are runtime log files rotated?", "Runtime logs are rotated upon reaching 500 MB in file size.", "log file rollover threshold", "Log file kitne MB hone par rotate ho jati hai?"),
            ("How many historical rotated log archives are preserved on disk?", "The system preserves 10 compressed rotated log archives.", "retained log archive count", "Disk par max kitni rotated log files rakhi jati hain?")
        ],
        "sop_release_checklist.md": [
            ("What test pass rate is mandatory before approving a release?", "Releases require 100% test pass rate across all regression suites.", "release qualification gate", "Release pass karne ke liye kitne percent tests pass hone chahiye?"),
            ("Who provides final sign-off for a production release?", "The Lead Systems Architect provides final release approval.", "production release sign-off authority", "Release ka final approval kaun deta hai?")
        ],
        "sop_release_versioning.md": [
            ("What versioning convention governs ChakrView releases?", "ChakrView follows Semantic Versioning (SemVer 2.0.0).", "version numbering scheme", "ChakrView releases kis versioning system ko follow karti hain?"),
            ("What changes require a major version bump?", "Breaking API changes and architecture deprecations require a major version bump.", "breaking change version rule", "Major version kab change kiya jata hai?")
        ],
        "sop_runtime_environments.md": [
            ("What Python version is officially supported for runtime deployment?", "The runtime officially supports Python 3.10 through 3.14.", "supported python interpreter runtime", "Runtime deployment ke liye kaun sa Python version support hota hai?"),
            ("What OS platforms are validated for ChakrView execution?", "ChakrView is validated on Windows 11 and Linux Ubuntu 22.04 LTS.", "supported operating systems", "ChakrView kin operating systems par validate hua hai?")
        ],
        "billing_enterprise_discounts.md": [
            ("What annual spend qualifies for the Tier-1 enterprise discount?", "Annual spend exceeding $50000 qualifies for Tier-1 discount.", "enterprise discount spend tier", "Tier-1 enterprise discount ke liye annual spend kitna hona chahiye?"),
            ("What percentage discount is granted to educational institutions?", "Accredited educational institutions receive a 25% discount.", "academic pricing concession", "Education institutions ko kitna discount diya jata hai?")
        ],
        "billing_licensing_compliance.md": [
            ("What software license governs ChakrView community edition?", "The community edition is distributed under the Apache License 2.0.", "open source license type", "Community edition kis license ke under release hota hai?"),
            ("How often are enterprise licensing compliance audits conducted?", "Enterprise licensing audits occur annually.", "compliance review cycle", "Enterprise license ka audit kitne time me hota hai?")
        ],
        "billing_metered_usage.md": [
            ("How are embedding compute queries billed in metered plans?", "Queries are billed at $0.002 per 1000 query tokens.", "query token billing rate", "Metered plan me queries kis rate par bill hoti hain?"),
            ("What is the free monthly token allocation for developer tier?", "Developer accounts receive 1000000 free tokens each month.", "free tier monthly token quota", "Developer account ko har mahine kitne free tokens milte hain?")
        ],
        "billing_payment_terms.md": [
            ("What payment term applies to enterprise invoices?", "Enterprise invoices carry Net-30 payment terms.", "invoice settlement timeframe", "Enterprise invoice ka payment term kya hota hai?"),
            ("What late payment fee is assessed on past-due balances?", "Past-due balances incur a 1.5% monthly late fee.", "overdue penalty interest", "Late payment par kitna percent fee charge hota hai?")
        ],
        "billing_refund_policy.md": [
            ("Within how many days can customers request a refund?", "Refund requests must be submitted within 14 days of purchase.", "money back guarantee window", "Refund request kitne din ke andar raise karni hoti hai?"),
            ("Are setup and onboarding fees eligible for refunds?", "Professional setup and onboarding fees are strictly non-refundable.", "non-refundable service charges", "Kya setup fees refund ho sakti hain?")
        ],
        "billing_sla_commitments.md": [
            ("What monthly uptime SLA is committed for enterprise tier?", "Enterprise tier guarantees 99.95% monthly uptime.", "enterprise availability SLA", "Enterprise tier ka monthly uptime guarantee kitna hai?"),
            ("What service credit is issued if monthly uptime falls below 99.0%?", "Uptime below 99.0% results in a 50% service credit.", "downtime service credit reimbursement", "Agar uptime 99% se niche gir jaye toh kitna credit milta hai?")
        ],
        "billing_subscription_tiers.md": [
            ("What is the monthly fee for the Professional subscription tier?", "The Professional tier costs $299 per month.", "pro tier monthly cost", "Professional subscription ka monthly charge kitna hai?"),
            ("How many concurrent documents can the Starter tier index?", "Starter tier indexes up to 50 documents concurrently.", "entry tier document limit", "Starter tier me max kitne documents index ho sakte hain?")
        ],
        "billing_support_tiers.md": [
            ("What response SLA is guaranteed for 24x7 mission critical support?", "Mission critical tier guarantees 1-hour 24x7 engineer response.", "urgent support response SLA", "Mission critical support me response time kitna milta hai?"),
            ("Which support tier includes a dedicated technical account manager?", "The Enterprise Premier tier includes a dedicated Technical Account Manager.", "dedicated account manager support plan", "Dedicated account manager kis support plan me milta hai?")
        ]
    }

    # Build examples: for each fact, create 4 query variants (exact, synonym, paraphrase, hinglish)
    # and pair with the exact source passage as POSITIVE, and a passage from a distractor doc in the SAME domain as HARD NEGATIVE.
    for doc_name, facts in doc_fact_specs.items():
        doc_content = docs.get(doc_name, "")
        
        # Determine topic family to pick hard negative doc
        family = None
        for fam, f_list in TOPIC_FAMILIES.items():
            if doc_name in f_list:
                family = fam
                break
        
        # Hard negative doc candidates from same family (excluding self)
        same_family_candidates = [f for f in TOPIC_FAMILIES[family] if f != doc_name and f in docs]
        
        for fact_idx, (q_exact, gold_span, q_syn, q_hinglish) in enumerate(facts):
            # Choose a hard negative document and extract its first sentence/chunk
            neg_doc_name = same_family_candidates[fact_idx % len(same_family_candidates)]
            neg_content = docs[neg_doc_name]
            neg_span = neg_content.split("\n\n")[1] if len(neg_content.split("\n\n")) > 1 else neg_content[:200]
            
            # Variant 1: Exact query
            examples.append({
                "id": f"EX_{ex_id:04d}",
                "query": q_exact,
                "category": "exact_entity",
                "pos_doc_name": doc_name,
                "pos_passage": gold_span,
                "neg_doc_name": neg_doc_name,
                "neg_passage": neg_span,
                "topic_family": family
            })
            ex_id += 1
            
            # Variant 2: Synonym query
            examples.append({
                "id": f"EX_{ex_id:04d}",
                "query": f"Inquire about {q_syn} details.",
                "category": "synonym",
                "pos_doc_name": doc_name,
                "pos_passage": gold_span,
                "neg_doc_name": neg_doc_name,
                "neg_passage": neg_span,
                "topic_family": family
            })
            ex_id += 1

            # Variant 3: Paraphrase query
            paraphrased_q = f"Could you provide information regarding the following: {q_exact.lower().rstrip('?')}"
            examples.append({
                "id": f"EX_{ex_id:04d}",
                "query": paraphrased_q,
                "category": "paraphrase",
                "pos_doc_name": doc_name,
                "pos_passage": gold_span,
                "neg_doc_name": neg_doc_name,
                "neg_passage": neg_span,
                "topic_family": family
            })
            ex_id += 1

            # Variant 4: Hinglish query
            examples.append({
                "id": f"EX_{ex_id:04d}",
                "query": q_hinglish,
                "category": "multilingual_hinglish",
                "pos_doc_name": doc_name,
                "pos_passage": gold_span,
                "neg_doc_name": neg_doc_name,
                "neg_passage": neg_span,
                "topic_family": family
            })
            ex_id += 1

    print(f"Total query-passage contrastive examples generated: {len(examples)}")
    
    # Stratified split by topic family to prevent data leakage
    # 70% Train, 15% Dev, 15% Test
    random.shuffle(examples)
    
    n_total = len(examples)
    n_train = int(n_total * 0.70)
    n_dev = int(n_total * 0.15)
    
    train_split = examples[:n_train]
    dev_split = examples[n_train:n_train + n_dev]
    test_split = examples[n_train + n_dev:]
    
    print(f"Split counts: Train={len(train_split)}, Dev={len(dev_split)}, Test={len(test_split)}")
    
    dataset_manifest = {
        "dataset_name": "chakrview_technical_retrieval_contrastive_v1",
        "total_examples": n_total,
        "splits": {
            "train_count": len(train_split),
            "dev_count": len(dev_split),
            "test_count": len(test_split)
        },
        "topic_distribution": {fam: len([e for e in examples if e["topic_family"] == fam]) for fam in TOPIC_FAMILIES}
    }
    
    for split_name, data in [("train", train_split), ("dev", dev_split), ("test", test_split)]:
        path = OUT_DIR / f"{split_name}_split.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        dataset_manifest["splits"][f"{split_name}_sha256"] = h
        print(f"Saved {split_name} ({len(data)} items) -> {h}")
        
    manifest_path = OUT_DIR / "DATASET_MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")

if __name__ == "__main__":
    generate_contrastive_dataset()
