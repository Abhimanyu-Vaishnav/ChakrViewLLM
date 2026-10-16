# Dataset Provenance: Batches 1461–1500

## Comprehensive 200-Task Capability Audit Suite

### 1. Dataset Overview and Composition

The capability evaluation dataset is versioned and frozen under:
`experiments/batch1461_1500/data/capability_audit_200.json`  
**Dataset SHA-256:** `931c877e766d427b1c6619f43b0dda1b3fa507d8f23ca54f4ac1da1e81d4c07a`

The suite comprises **200 structured evaluation tasks** categorized into 4 core functional domains plus control tracks:

| Track Identifier | Task Count | Purpose and Method | Primary Metrics |
|---|---|---|---|
| **Track B: Factual Recall & Grounding** | 60 tasks | Evaluates factual question answering in 3 modes: Model alone vs. Retrieval Pipeline vs. Control baseline. | Exact/Normalized Match, Evidence Citation Validity, Abstention Recall, False-Refutation Rate. |
| **Track C: Arithmetic & Mathematics** | 40 tasks | Arithmetic (addition, subtraction, multiplication, division, modulo, order of operations) and multi-step word problems. | Exact Numeric Match, Error Rate, Format Compliance. |
| **Track D: Coding & Program Execution** | 40 tasks | Python function completion evaluated against sandboxed unit test assertions. | Syntax Validity Rate, Functional Test Pass Rate, Execution Errors. |
| **Track E: Generalization & Robustness** | 60 tasks | 20 matched pairs (40 tasks) varying syntax/language (Hinglish & English), plus 20 out-of-domain unanswerable control tasks. | Pairwise Consistency, Paraphrase Robustness, Out-of-Domain Abstention Rate. |

---

### 2. Provenance & Sourcing Integrity

1. **Arithmetic & Mathematics (Track C):**
   - Synthesized using standard deterministic integer operations with explicit operands not present in identical combinations in prior training corpora.
2. **Coding with Assertions (Track D):**
   - Standard elementary Python function specifications covering list operations, string manipulation, boolean logic, and math helpers, paired with strict executable unit test assertions.
3. **Factual Grounding (Track B):**
   - Grounded in canonical technical documents from `experiments/batch1141_1180/data/corpus/*.md`.
4. **Generalization Pairs & Controls (Track E):**
   - Matched pairs designed to evaluate whether system decisions survive syntactic inversion, synonym substitution, and bilingual phrasing.
