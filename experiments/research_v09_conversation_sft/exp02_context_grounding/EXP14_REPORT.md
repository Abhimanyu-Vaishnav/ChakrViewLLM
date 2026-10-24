# ChakrView Experiment 14: Research Report
## Context Grounding and Evaluation Repair

### 1. Executive Summary

Experiment 14 continued directly from the findings of Experiment 13. While Experiment 13 achieved stable optimization loss reduction on CPU, forensic auditing revealed that its headline metric (18.33% response relevance) relied on an unigram token-overlap heuristic that scored nonsensical outputs as successful responses.

In Experiment 14, we implemented a completely repaired, rigorous, and deterministic evaluation foundation for conversational context grounding:
1. **Evaluation Methodology Repaired**: Eliminated token-overlap metrics. Implemented 10 deterministic criteria (Exact Entity Binding, Turn-Dependent Fact Retrieval, Anaphora Disambiguation, Superseded Instruction Overriding, User Correction Adherence, Distractor Rejection, Ambiguity Detection, Missing Context Abstention, Factual Uncertainty Acknowledgment, and AST Code Parsing).
2. **Context-Grounding Benchmark Constructed**: Created a versioned benchmark comprising 190 multi-turn scenarios (100 Train, 30 Val, 60 Held-Out Test). Audited for 0 prompt string leakage and 0 entity identifier leakage between splits.
3. **Rigorous Three-Arm Evaluation Executed**:
   - **Arm A (Frozen Canonical Step-10000)**: Evaluated zero-shot on 60 held-out test scenarios. Result: **0 / 60 (0.0%)**. Primary failures: repetitive n-gram attractor loops (16 cases), empty outputs (9 cases), and pretraining hallucinations.
   - **Arm B (Exp13 Candidate Best)**: Evaluated on identical 60 test scenarios. Result: **0 / 60 (0.0%)**. Primary failures: entity token mismatch (memorizing `train` entity names instead of binding dynamic prompt entities, 6 cases), value mismatch (6 cases), distractor intrusion (6 cases). This confirms that under honest, deterministic scoring, pure causal LM SFT achieves 0% held-out conversational entity binding.
   - **Arm C (Deterministic Rules Baseline)**: Non-neural rule/oracle reference. Result: **60 / 60 (100.0%)**.
4. **Field-Conditioned Pointer Research Formulated**: Prepared the architectural and mathematical specification for Exp15 (`POINTER_RESEARCH_DESIGN.md`) to resolve dynamic entity binding via slot-conditioned pointer attention while keeping the 3.44M causal LM backbone frozen.
5. **Zero Training & Zero Git Mutations**: No model training or fine-tuning was performed in this phase. Zero Git operations occurred.

### Final Experiment Status:
`EVALUATION FOUNDATION READY FOR REVIEW`

---

### 2. Verified Status of Deliverables

| Deliverable Component | Status | Verification Evidence |
| :--- | :---: | :--- |
| **`EXP14_DESIGN.md`** | **Implemented & Tested** | Comprehensive specification of principles, invariants, and benchmark categories. |
| **`EVALUATION_METRICS.md`** | **Implemented & Tested** | Mathematically rigorous metric definitions, formulas, and human review rubrics. |
| **`generate_context_benchmark.py`** | **Implemented & Tested** | Generated 190 disjoint scenarios (100 Train, 30 Val, 60 Test) across 10 categories. |
| **`audit_context_benchmark.py`** | **Implemented & Tested** | 0 prompt leakage, 0 entity leakage, max context 154 tokens ($\le 512$). |
| **`evaluate_context_grounding.py`** | **Implemented & Tested** | Evaluated Arm A (0.0%), Arm B (0.0%), and Arm C (100.0%) on 60 held-out test prompts. |
| **`test_context_grounding.py`** | **Implemented & Tested** | All 8 automated test suites pass cleanly with exit code 0 (`pytest`). |
| **`BENCHMARK_MANIFEST.json`** | **Implemented & Tested** | Cryptographic SHA-256 hashes and split sizes recorded. |
| **`BASELINE_RESULTS.json`** | **Implemented & Tested** | Complete raw outputs, scores, and failure classification records saved. |
| **`POINTER_RESEARCH_DESIGN.md`** | **Designed & Documented** | Architectural design for Exp15 field-conditioned pointer attention. |
| **`INTEGRITY_REPORT.json`** | **Implemented & Tested** | Machine-readable verification audit confirming bit-identical canonical artifacts. |

---

### 3. Quantitative Evaluation Results (Repaired vs. Historical)

| Model / Arm | Exp13 Flawed Heuristic | Exp14 Repaired Deterministic Scoring | Ground-Truth State |
| :--- | :---: | :---: | :--- |
| **Arm A: Frozen Canonical Step-10000** | 0.0% (0/60) | **0.0% (0/60)** | Collapses into pretraining attractor loops or empty outputs. |
| **Arm B: Exp13 Candidate (Trained LM)** | 18.33% (11/60)* | **0.0% (0/60)** | *Exp13 heuristic counted random unigrams (`"be"`, `"of"`). Repaired metric proves 0% true entity binding due to training name memorization. |
| **Arm C: Deterministic Rules Baseline** | N/A | **100.0% (60/60)** | Non-neural floor confirming benchmark test assertions are fully solvable. |

---

### 4. Integrity and Cryptographic Audit

All canonical, runtime, and historical assets remain untouched and byte-identical:
- `canonical_checkpoint_step_10000`: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` [VERIFIED]
- `tokenizer_vocab`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` [VERIFIED]
- `tokenizer_merges`: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` [VERIFIED]
- `runtime_session`: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` [VERIFIED]
- `exp13_candidate_best`: `72741e66baaa0bd5f0f7c02ca3a16c62c53d339beaa5c1a4ba7f1e1a21b4017d` [VERIFIED]

Zero Git mutations, commits, merges, or pushes were performed.

---

### 5. Recommended Next Step

**Milestone / Exp15**: **Field-Conditioned Pointer Attention Implementation**.
With the repaired benchmark and rigorous evaluation harness in place, proceed to implement the Exp15 field-conditioned pointer head specified in `POINTER_RESEARCH_DESIGN.md`. Fine-tune only the pointer query/key projections while keeping the 3.44M causal LM backbone frozen, aiming to elevate held-out entity binding from $0.0\%$ to $>65.0\%$.
