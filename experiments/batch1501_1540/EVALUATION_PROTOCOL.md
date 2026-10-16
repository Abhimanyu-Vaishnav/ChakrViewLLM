# Evaluation Protocol

## Batches 1501–1540: Neuro-Symbolic Tool Routing, Arithmetic & Structured Refutation

### 1. Protocol Objective

To objectively test the new tool-routing, arithmetic evaluation, and structured verification architecture, this protocol establishes frozen test suites, pre-declared pass/fail gates, and explicit scoring functions across four evaluation tracks totaling **300 held-out tasks**.

---

### 2. Evaluation Tracks & Task Counts

| Track Identifier | Evaluation Domain | Task Count | Primary Metrics | Baseline Comparison |
|---|---|---|---|---|
| **Track 1: Arithmetic** | Safe AST Arithmetic Execution | 100 tasks | Exact Accuracy, Security Rejection Rate, Division by Zero Safety | Canonical Model (0.0% in Batch 1461–1500) |
| **Track 2: Verification**| Entity-Attribute Claim Verification | 100 tasks | Accuracy by Verdict (`SUPPORTED`, `CONTRADICTED`, `INSUFFICIENT`, `CONFLICTING`, `INVALID`) | Regex Baseline (50.0% in Batch 1461–1500) |
| **Track 3: Routing** | Request Classification & Dispatch | 50 tasks | Route Selection Accuracy, Intent Extraction Precision | CognitiveRouter Heuristic Baseline |
| **Track 4: End-to-End**| Full Assistant Integration | 50 tasks | End-to-End Correctness, Citation Validity, Abstention Precision | Batch 1461–1500 Knowledge Assistant |
| **Total Test Suite** | Multi-Track Frozen Suite | **300 tasks** | Overall Task Reliability | Disentangled System Matrix |

---

### 3. Pre-Declared Pass/Fail Criteria

1. **Arithmetic Evaluation (Track 1):**
   - Exact mathematical accuracy on valid expressions: **$\ge 98.0\%$** (target: 100%).
   - Security rejection rate on adversarial inputs (imports, calls, attributes, infinite exponents): **100.0%**.
   - Zero division handling without uncaught exception: **100.0%**.
2. **Structured Verification (Track 2):**
   - Overall verification accuracy across all 5 outcomes: **$\ge 90.0\%$**.
   - Contradiction detection accuracy on false premises: **$\ge 85.0\%$** (significant upgrade from the 50.0% regex baseline).
   - False-refutation rate on unmentioned attributes: **$0.0\%$** (absence of evidence must be classified as `INSUFFICIENT_EVIDENCE`).
3. **Tool Routing (Track 3):**
   - Route selection accuracy: **$\ge 92.0\%$**.
   - Zero execution of unvalidated code.
4. **End-to-End Assistant (Track 4):**
   - End-to-end task success rate: **$\ge 80.0\%$**.
   - Citation fidelity on answered claims: **100.0%**.
