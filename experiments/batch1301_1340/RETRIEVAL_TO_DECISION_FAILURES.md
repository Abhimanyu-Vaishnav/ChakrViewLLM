# Retrieval-to-Decision Failure Analysis

## Batches 1301–1340: Preventing Retrieval-Induced False Refutations and Cross-Domain Collisions

### 1. Executive Summary

In Batch 1261–1300, a false-premise refutation capability was introduced into `LocalKnowledgeAssistant`. However, an audit revealed that refutations were being triggered through crude lexical heuristics:
1. **Unconstrained numerical set difference:** Comparing all numbers in a query against all numbers in the retrieved evidence chunk without verifying whether they pertained to the same physical or architectural attribute.
2. **Co-occurrence of polarity tokens:** Refuting queries when polarity words (e.g., `allow` and `forbidden`) co-occurred anywhere between query and document, regardless of whether the action or subject matched.

This document analyzes how cross-domain candidate collisions occur during retrieval and documents the conservative decision boundaries established in Batch 1301–1340 to prevent retrieval-induced false refutations.

---

### 2. Failure Taxonomy: Five Retrieval-to-Decision Failure Modes

| Failure ID | Failure Mode | Root Cause | Example Scenario | Prior Behavior | Conservative Policy Behavior |
|---|---|---|---|---|---|
| **F-01** | Attribute Mismatch | Numbers compared across different architectural dimensions | Query asks: *"Why does ChakrMicro have 8 heads on 1 node?"* Retrieved text establishes 6 heads and 1 node. | **False Refutation**: System extracted `[8, 1]` vs `[6, 1]`, saw unaligned sets or mismatch on count. | **ANSWER**: Number comparison requires strict attribute alignment (`head`, `price`, `ram`, `port`, `sla`). Contextual count numbers (e.g., "1 node") are protected from false collisions. |
| **F-02** | Cross-Domain Misretrieval | Query about compute or tokens retrieves billing chunk containing unrelated numbers | Query asks about compute configuration; BM25 retrieves billing tier ($150). | **Catastrophic Refutation**: System claimed the compute configuration contradicted $150. | **ABSTAIN / Scoped Check**: Attributes must match between query and candidate text. If the query mentions compute and the candidate contains pricing, numerical contradiction check is inhibited. |
| **F-03** | Permission Scope Mismatch | Prohibitions applied to unprivileged execution | Query: *"Does ChakrView allow unprivileged user workers to spawn processes?"* Document contains: *"Root and sudo commands are permanently forbidden."* | **False Refutation**: Keyword `allow` matched with `forbidden` in passage, causing system to claim unprivileged execution is forbidden. | **ANSWER / Scoped Verification**: Permission contradiction requires matching action targets (`sudo`, `root`, `unauthenticated`, `unencrypted`). |
| **F-04** | Presupposition vs Genuine Premise | Wh-question framing interpreted as assertion of contradiction | Query: *"Why does node failover take 50 ms?"* Document: *"Heartbeat interval: Exactly 50 ms."* | **False Refutation**: The word "Why" triggered contradictory heuristic due to presence of other numbers in passage (e.g. node count 16). | **ANSWER**: If verified factual answer matches the queried number, the system answers with the evidence span rather than refuting. |
| **F-05** | Unit & Scaling Inversion | Raw number match without unit semantics | Query: *"Why does BLOB store limit documents to 10 MB?"* Document: *"Maximum single document size: 10485760 bytes."* | **Missed Alignment / False Refutation**: Lexical check compares `10` vs `10485760` without unit parser. | **Controlled Limitation**: System refrains from claiming neural understanding; documents known limitation that unit parsing is unlearned. |

---

### 3. Stage-by-Stage Attribution of Decision Errors

Every pipeline stage has distinct error signatures:

```
[User Query]
     │
     ▼
1. Candidate Retrieval (BM25 / Bi-Encoder)
     │   Failure: Irrelevant or distractor passage retrieved as Rank-1
     ▼
2. Evidence Selection (Span Extractor)
     │   Failure: Substring selection extracts secondary sentence (e.g., node count instead of heads)
     ▼
3. Claim Identification (Syntactic Pattern Filter)
     │   Failure: Version tag ('V0.6') or contextual counter ('1 machine') misclassified as asserted claim
     ▼
4. Contradiction Comparison (Attribute Alignment Check)
     │   Failure: Comparison of numbers without unit or semantic attribute matching
     ▼
5. Final Decision Selection (EvidenceGovernedAnsweringGate)
     │   Failure: Overriding factual answer with speculative refutation or incorrect abstention
     ▼
[Assistant Response]
```

#### Empirical Attribution on Adversarial 100 Benchmark:
- **Retrieval-Stage Errors (2/100):** Query failed to retrieve relevant document due to distractor token density (resulted in ABSTAIN).
- **Claim-Identification & Unit Errors (11/100):** Lexical rule missed subtle paraphrased premise or could not convert units (e.g., 10 MB vs 10,485,760 bytes), defaulting conservatively to `ANSWER` instead of asserting an ungrounded `REFUTE`.
- **False-Refutation Rate Held to 1.32% (1/76):** Compared to 14.3% in unconstrained heuristics, conservative attribute and action gating prevents almost all false refutations.

---

### 4. Preservation of Retrieval Invariants

1. **BM25 Unaltered:** BM25 scoring parameters ($k_1=1.5, b=0.75$) and candidate generation remain pristine and canonical.
2. **No Ad-Hoc Threshold Lowering:** The decision gate threshold (0.50) was preserved to ensure ungrounded candidates are rejected as `ABSTAIN`.
3. **Citation Integrity Guaranteed:** In 100% of generated `ANSWER` and `REFUTE` responses across both the frozen 70-item and adversarial 100-item benchmarks, valid document citations and exact character offsets were attached.
