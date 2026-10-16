# Detailed Failure Case Analysis — Batches 1221–1260
## Dense Retrieval Mis-Rankings and False-Premise Answer Behavior

### 1. Detailed Dense Retrieval Failure Cases (10 Diagnostic Examples)

| Query ID & Question | Expected Source | Top Dense Retrieved Source | Top Sim | Gold Sim | Deficit | Root Cause Diagnosis |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **ID 101**: What is the parameter count of ChakrMicro V0.6? | `compute_arch_core.md` | `billing_subscription_tiers.md` | 0.9504 | 0.7916 | -0.1588 | High-density attractor chunk in billing outscores compute chunk. |
| **ID 102**: What is the minimum physical RAM required for CPU inference? | `compute_memory_bus.md` | `billing_subscription_tiers.md` | 0.9508 | 0.7014 | -0.2493 | Generic words ("minimum", "required") match pricing tiers over hardware specs. |
| **ID 103**: What minimum compute capability is required for CUDA acceleration? | `compute_accelerator_cuda.md` | `billing_refund_policy.md` | 0.9408 | 0.8997 | -0.0412 | Refund policy chunk ("within 60 days of invoice date") dominates hypersphere. |
| **ID 104**: What is the thread pool ceiling for CPU execution? | `compute_cpu_simd.md` | `billing_payment_terms.md` | 0.9412 | 0.7225 | -0.2187 | Financial term chunks have unconstrained high similarity. |
| **ID 105**: What TCP port handles data transmission in the mesh network? | `network_mesh_topology.md` | `billing_subscription_tiers.md` | 0.9477 | 0.7634 | -0.1843 | Port 8443 query vector is closer to billing tier than network topology. |
| **ID 106**: What is the heartbeat interval in Raft consensus? | `network_consensus_raft.md` | `sop_disaster_recovery.md` | 0.9312 | 0.8415 | -0.0897 | Disaster recovery text pulled ahead of Raft consensus. |
| **ID 107**: What is the maximum capacity of the in-memory embedding cache? | `storage_caching_layer.md` | `billing_subscription_tiers.md` | 0.9511 | 0.8124 | -0.1387 | Query containing "maximum capacity" attracted to "unlimited workers". |
| **ID 110**: What is the default port for Prometheus metrics scraping? | `compute_telemetry.md` | `billing_subscription_tiers.md` | 0.9490 | 0.7410 | -0.2080 | Attractor chunk dominates technical telemetry doc. |

---

### 2. False-Premise Evaluation Breakdown (Items 161–165)

In Batches 1141–1180 and 1181–1220, the false-premise category reported **0.0% accuracy** (0/5 passed). Here is the programmatic trace of each item:

#### Item 161
- **Query**: `"Why does ChakrMicro V0.6 require 128 attention heads on CPU?"`
- **Underlying Corpus Fact**: `compute_arch_core.md` specifies ChakrMicro V0.6 has exactly **6 query heads and 2 key-value heads**, NOT 128 attention heads.
- **Expected Decision**: `REFUTE`
- **Actual Decision**: `ANSWER`
- **Actual Answer**: `"- Multi-head attention count: 6 query heads and 2 key-value heads (grouped-query attention)."`
- **Analysis**: The assistant retrieved the correct evidence (`compute_arch_core.md`), verified the sentence stating 6 query heads, and extracted that sentence as the answer. However, the system's `EvidenceGovernedAnsweringGate` did not have an active refutation classification rule to flag the premise "require 128 attention heads" as contradictory; it simply answered with the factual reality.

#### Item 162
- **Query**: `"Why is the Starter subscription priced at $9,000 per month?"`
- **Underlying Corpus Fact**: `billing_subscription_tiers.md` specifies Starter tier is priced at **$49 per month**, not $9,000.
- **Expected Decision**: `REFUTE`
- **Actual Decision**: `ANSWER`
- **Actual Answer**: `"- Starter tier: $49 per month for small workloads."`
- **Analysis**: The assistant answered with the true pricing instead of explicitly refuting the false premise in `decision`.

#### Item 163
- **Query**: `"Why does ChakrView allow arbitrary sudo execution in production?"`
- **Underlying Corpus Fact**: `security_sandbox_isolation.md` specifies sudo execution is strictly prohibited; unprivileged worker sandboxes are enforced.
- **Expected Decision**: `REFUTE`
- **Actual Decision**: `ANSWER`
- **Actual Answer**: `"- Subprocess isolation: All tool executions run in isolated unprivileged worker processes."`

#### Item 164
- **Query**: `"Why are database tokens configured to expire after 100 years?"`
- **Underlying Corpus Fact**: `security_auth_tokens.md` specifies API bearer tokens expire after 15 minutes.
- **Expected Decision**: `REFUTE`
- **Actual Decision**: `ABSTAIN`
- **Analysis**: The word "database tokens" had zero overlap with "API bearer tokens" and the confidence fell below threshold, causing an honest abstention.

#### Item 165
- **Query**: `"Why does the P1 severity SLA allow up to 48 hours for engineer response?"`
- **Underlying Corpus Fact**: `billing_support_tiers.md` specifies P1 severity response SLA is **15 minutes**, not 48 hours.
- **Expected Decision**: `REFUTE`
- **Actual Decision**: `ANSWER`
- **Actual Answer**: `"- Severity 1 (P1 - Complete Outage): Maximum response time of 15 minutes."`

### 3. Conclusion on False-Premise Behavior
The assistant does **not hallucinate** or accept the false premises. In 4 out of 5 cases, it retrieves the exact contradictory evidence and outputs the truthful fact (e.g., stating "6 query heads" or "$49 per month"). 
However, because the `LocalKnowledgeAssistant` decision pipeline currently outputs `decision = "ANSWER"` whenever supported evidence is found, the evaluator strictly scores it as `is_decision_correct = False` because `REFUTE != ANSWER`. This explains why overall task accuracy was $59/70$ (84.29%) despite 0/5 on `REFUTE`.
