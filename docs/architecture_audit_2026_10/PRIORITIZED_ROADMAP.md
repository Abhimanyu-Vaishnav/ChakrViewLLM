# Prioritized Roadmap: Engineering Sequence & Acceptance Milestones
**Document Date**: October 21, 2026  
**Auditor**: Antigravity AI Engine  
**Project**: ChakrView Indigenous Neural-Cognitive Architecture

---

## 1. Prioritization Rationale & Dependency Ladder

Prioritization follows the non-negotiable invariant of grounded empirical engineering:
1. **Never attempt multi-device distributed systems before local single-device execution is bulletproof.**
2. **Never attempt full-scale generative reasoning before context token extraction is solved.**
3. **Never mistake deterministic FSM syntax enforcement for learned neural capability.**

```
+---------------------------------------------------------------------------------+
| PHASE 1: Canonical Baseline & Preflight Stability (DONE)                        |
+---------------------------------------------------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
| PHASE 2: Field-Conditioned Pointer Attention (NEXT IMMEDIATE MILESTONE)         |
|   - Objective: Fix multi-field positional collapse in Exp 11/12                 |
|   - Scope: Trainable field-query embeddings injected into frozen backbone       |
+---------------------------------------------------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
| PHASE 3: Resource-Adaptive Context & Memory Hardening                           |
|   - Objective: Guarantee zero-crash execution under <2GB RAM edge profile       |
|   - Scope: Sliding-window KV-cache quant, PPB SQLite WAL tuning                 |
+---------------------------------------------------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
| PHASE 4: Sovereign Native Capability Benchmark Execution                        |
|   - Objective: Run 10-suite benchmark comparing Tiers A, B, C, D, and E         |
|   - Scope: Document genuine native language/reasoning vs symbolic floor         |
+---------------------------------------------------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
| PHASE 5: Governed Epistemic Engine & Evidence Grounding                         |
|   - Objective: Multi-document contradiction detection & calibrated uncertainty  |
|   - Scope: BM25/Dense hybrid fusion, citation span attribution                  |
+---------------------------------------------------------------------------------+
                                        |
                                        v
+---------------------------------------------------------------------------------+
| PHASE 6: Distributed Task Federation Prototype                                  |
|   - Objective: Multi-node task distribution over local physical LAN             |
|   - Scope: HMAC authenticated socket protocol, peer capability negotiation      |
+---------------------------------------------------------------------------------+
```

---

## 2. Detailed Milestone Specifications

### Milestone 1: Field-Conditioned Pointer Head Training (Immediate Next Step)
- **Problem Statement**:
  - In Experiment 12, the schema controller guaranteed 100% valid JSON syntax, but Whole-Object Exact Match on Test B was only 3.3%.
  - Root cause: The pointer head received an unconditioned prefix query and repeatedly extracted the first salient entity across all requested slots.
- **Proposed Solution**:
  - Introduce explicit field query tokens (`[QUERY_ENTITY]`, `[QUERY_STATUS]`, `[QUERY_CODE]`).
  - Project field embeddings into the query projection layer of the span pointer head.
  - Fine-tune strictly the pointer query/key projections while keeping the 3.4M causal LM backbone frozen.
- **Acceptance Criteria**:
  - Test B Whole-Object Exact Match improves from $3.3\%$ to $> 50.0\%$.
  - Canonical checkpoint and tokenizer hashes remain 100% untouched.

### Milestone 2: Hardware-Adaptive Edge Profile Hardening
- **Objective**: Guarantee reliable execution on Tier 1 (2–4 GB RAM, dual-core CPU) hardware without OOM crashes.
- **Actions**:
  - Implement dynamic INT8 quantization for KV-cache tensors during low-memory signals.
  - Implement an automated memory stress harness that limits process memory to $512\text{ MB}$.
- **Acceptance Criteria**:
  - 100 consecutive extraction tasks execute without a single process crash, swap thrashing lockup, or `MemoryError`.

### Milestone 3: Full 10-Suite Native Capability Benchmark Execution
- **Objective**: Establish the official baseline of ChakrMicro native intelligence across language modeling, negation, multi-step math, and instruction following.
- **Actions**:
  - Run the automated benchmark harness across Tiers A, B, C, D, and E.
  - Publish `CAPABILITY_BENCHMARK_RESULTS.json` and per-example logs.
- **Acceptance Criteria**:
  - Complete independent reproducibility of all reported metrics from saved CSV predictions.

### Milestone 4: Epistemic Retrieval & Contradiction Grounding
- **Objective**: Connect the native model to local workspace files via the hybrid retriever with calibrated uncertainty.
- **Actions**:
  - Integrate `HybridRetriever` with the field-conditioned pointer.
  - Enforce explicit `"none"` abstention when context does not contain the target answer.
- **Acceptance Criteria**:
  - $0\%$ hallucination on synthetic missing-value stress tests.

### Milestone 5: Heterogeneous LAN Distributed Inference Prototype
- **Objective**: Safely distribute independent subtasks across two separate physical machines on a local network.
- **Actions**:
  - Deploy `DistributedFederatedEngine` between two physical PCs via authenticated HMAC sockets.
  - Measure serialization latency vs. single-device execution.
- **Acceptance Criteria**:
  - Automatic task failover when secondary node is unceremoniously disconnected.
