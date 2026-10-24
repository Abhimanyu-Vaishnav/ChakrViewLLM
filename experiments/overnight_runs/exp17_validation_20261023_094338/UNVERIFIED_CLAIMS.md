# ChakrView Exp17 Validation — Unverified Claims & Missing Evidence

**Audit Reference**: Exp17 Instruction SFT v1 Validation  
**Auditor**: Independent ML Experiment Auditor  
**Date**: 2026-10-23  

---

## 1. Summary of Unverified or Partially Verified Claims

In this audit, we strictly distinguish between claims supported by concrete empirical proofs and claims that rely on heuristics or unexecuted live checks:

### Claim 1: "100% of Factual QA and Explanations were verified against external literature"
- **Status**: **`PARTIALLY_VERIFIED`**
- **Evidence Gap**:
  - In `generate_corrected_csv.py`, 42 out of 100 factual and explanation records matched specific dictionary keys (e.g., NASA, IUPAC, Rayleigh 1871).
  - The remaining 58 records fell back to static categorical labels:
    - `"National / International Standards Gazetteer"` (21 factual records)
    - `"Empirical physical / biological mechanism consensus"` (37 explanation records)
  - The script did not perform real-time external HTTP requests or interactive per-item lookup. While the underlying facts (e.g. capital of Poland is Warsaw, clouds condense from rising moist air) are accurate by standard consensus, the claim of independent literature citation per item is only partially verified.

---

### Claim 2: "SFT Model Achieved 100% Structured Formatting Success"
- **Status**: **`CONTRADICTED BY TASK ACCURACY`**
- **Evidence Gap**:
  - The model achieved a **100% EOS termination rate** on the 5 held-out structured formatting test prompts.
  - However, the actual text produced by the model under greedy decoding was:
    - 2 empty strings (`""`)
    - 3 docstring/code syntax fragments (`'\n        """\n        if not isinstance("metric_train_3")'`)
  - **Zero out of 5 outputs** were valid RFC 8259 JSON objects.
  - Calling this "100% success" conflated turn termination with format compliance.

---

### Claim 3: "SFT Model Learned Basic Arithmetic Reasoning"
- **Status**: **`CONTRADICTED BY TASK ACCURACY`**
- **Evidence Gap**:
  - In `EXP17_TEST_BASIC_ARITHMETIC_000` (Prompt: *"What is the sum of 503 and 505?"*), expected answer is `1008`. The model generated: `"\nThe sum of 2005 is 406."` `[STOP_TOKEN]`.
  - In `EXP17_TEST_BASIC_ARITHMETIC_001` (Prompt: *"What is the product of 507 multiplied by 2?"*), expected answer is `1014`. The model generated: `"\nThe primary of 21 is 406."` `[STOP_TOKEN]`.
  - While the model learned the phrase structure `The sum of...` and emitted `EOS`, actual mathematical reasoning was not acquired.

---

### Claim 4: "Exp17 SFT Model is Ready for Deployment as a Conversational Assistant"
- **Status**: **`UNVERIFIED / NOT SUPPORTED`**
- **Evidence Gap**:
  - The model has 3.44 million parameters and was trained for 225 optimal steps on 320 synthetic instruction pairs.
  - On the 20-prompt Normal Chat Gate v0.1 suite, task accuracy is **0.0%**.
  - The model has learned conversational turn boundaries, but cannot yet serve as a reliable conversational agent.
