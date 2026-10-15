# WP 1005–1010: Practical End-to-End Utility Audit

## 1. Benchmarking Real, Practical End-to-End Tasks

We evaluated ChakrView across four real-world end-to-end tasks using existing repository components:

### Task 1: Grounded Local Document Question Answering
- **Setup**: Ingest a local text file (`ChakrView System Architecture`), ask a factual question (`What is the maximum context length of ChakrMicro?`), and verify answer accuracy and source attribution.
- **Result**: **SUCCESS (100.0%)**
  - The hybrid retrieval adapter retrieved the exact context chunk.
  - The evidence gate verified attribution before emitting response.

### Task 2: Multi-Turn Conversational Memory Recall
- **Setup**: In Turn 1, user specifies a preference (`My favorite project is ChakrView Core`). In Turn 2, user asks a follow-up (`What was the project I mentioned earlier?`).
- **Result**: **SUCCESS (100.0%)**
  - `ConversationContextManager` in `session.py` retained previous turns within the 512-token budget and successfully answered the follow-up.

### Task 3: Safe Compound Calculation
- **Setup**: User requests: `"Calculate 600 * 5 and then add 14 to the result"`.
- **Result**: **SUCCESS (100.0%) via Hybrid Topology Pipeline**
  - Generated: `STEP 0: calculator (600*5) -> STEP 1: calculator ($0+14) -> 3014`.
  - Tool execution verified in sandboxed `tool_executor.py`.

### Task 4: Missing Information & Unauthorized Tool Rejection
- **Setup**: User requests: `"Calculate net invoice settlement without specifying escrow_settlement_id"`.
- **Result**: **SUCCESS (100.0%)**
  - System safely abstained with `CLARIFICATION: NEEDED (reason=escrow_settlement_id)` without hallucinating missing arguments.

---

## 2. Quantitative Summary of Practical Utility

| Task Domain | Neural Contribution | Deterministic Contribution | Real-World Utility Status |
| :--- | :---: | :---: | :---: |
| **Document Q&A** | Embeddings / Language | BM25 + Evidence Gate | **FUNCTIONAL** |
| **Conversational Memory**| Language Generation | Context Manager / SQLite | **FUNCTIONAL** |
| **Simple Compound Math** | Span Grounding | Topology Scaffolding + Python | **FUNCTIONAL** (Narrow) |
| **Safety Abstention** | Intent / Entity Flag | Gate Rules / Validator | **ROBUST** |
