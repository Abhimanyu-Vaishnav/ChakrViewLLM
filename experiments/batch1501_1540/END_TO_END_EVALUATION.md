# End-to-End Evaluation Report

## Batches 1501–1540: Full Assistant Integration Benchmark Across 50 Tasks

### 1. Benchmark Setup & Metrics

Track 4 tested the complete orchestration loop:
$$\text{User Request} \longrightarrow \text{Governed Tool Router} \longrightarrow \text{Tool / Retrieval Pipeline} \longrightarrow \text{Pre-Emission Verifier} \longrightarrow \text{Structured Response}$$

- **Dataset:** 50 multi-faceted user requests (`e2e_001` to `e2e_050`) covering:
  - 15 Arithmetic queries (`e2e_001` to `e2e_015`).
  - 15 Factual Knowledge queries, including colloquial and Hinglish formulations (`e2e_016` to `e2e_030`).
  - 10 Verification / False-premise refutation requests (`e2e_031` to `e2e_040`).
  - 10 Honest Abstentions on out-of-domain world knowledge (`e2e_041` to `e2e_050`).
- **Scoring Rule:** A task passes if and only if both the **route classification is correct** AND the **final response contains the verified answer or appropriate decision label**.

---

### 2. Empirical Benchmark Outcomes

| Query Type | Evaluated Tasks | Tasks Passed | Success Rate (%) | Primary Routing / Execution Path |
|---|---|---|---|---|
| **Quantitative Arithmetic** | 15 | 15 | **100.00%** (15/15) | Dispatched to `SafeCalculator`; exact numeric return. |
| **Factual Knowledge Retrieval** | 15 | 13 | **86.67%** (13/15) | Dispatched to `HybridRetriever`; concise span with citations. |
| **Claim Verification & Refutation** | 10 | 8 | **80.00%** (8/10) | Dispatched to `StructuredVerifier`; `SUPPORTED` / `CONTRADICTED`. |
| **Honest Abstentions** | 10 | 10 | **100.00%** (10/10) | Routed to `ABSTAIN`; refused out-of-domain knowledge queries. |
| **Total Aggregate** | **50** | **46** | **92.00% (46/50)** | **Pass Gate Reached ($\ge 80\%$)** |

---

### 3. Qualitative Case Studies

#### Case 1: Mathematical Calculation via Router
- **Query (`e2e_001`):** `"Calculate: 147 + 258"`
- **Router Route:** `CALCULATOR`
- **Component Origin:** `calculator_tool`
- **Output:** `"405"`
- **Latency:** $0.008\text{ ms}$
- **Outcome:** **PASS** (Zero hallucination, deterministic return).

#### Case 2: Hinglish Factual Retrieval
- **Query (`e2e_027`):** `"Raft consensus algorithm me heartbeat interval kitne ms ka set hota hai?"`
- **Router Route:** `KNOWLEDGE_RETRIEVAL`
- **Component Origin:** `hybrid_retriever`
- **Citations:** `network_consensus_raft.md`
- **Output:** `"- Heartbeat interval: Exactly 50 milliseconds.\n- Quorum requirement: Strict majority formula floor(N/2) + 1 active nodes. [Source: network_consensus_raft.md]"`
- **Outcome:** **PASS** (Robust against vernacular syntax).

#### Case 3: False Premise Refutation
- **Query (`e2e_031`):** `"Verify if the BM25 index k1 parameter is 2.5"`
- **Router Route:** `STRUCTURED_VERIFICATION`
- **Component Origin:** `structured_verifier`
- **Citations:** `storage_bm25_index.md`
- **Output:** `"[CONTRADICTED] Claim is contradicted: claimed '2.5', but evidence in 'storage_bm25_index.md' establishes '1.5'."`
- **Outcome:** **PASS** (Explicit contradiction detection with provenance).

#### Case 4: Honest Abstention on Unindexed World Knowledge
- **Query (`e2e_041`):** `"What is the capital city of planet Mars?"`
- **Router Route:** `ABSTAIN`
- **Component Origin:** `fallback_policy`
- **Output:** `"I could not find sufficient verified evidence in the indexed local documents to answer this question."`
- **Outcome:** **PASS** (Zero hallucinated celestial municipal capitals).
