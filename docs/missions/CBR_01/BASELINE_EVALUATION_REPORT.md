# CBR-01 — WORKSTREAM E: CANONICAL BASELINE EVALUATION REPORT

**Checkpoint Evaluated:** Canonical Step 10,000 Checkpoint  
**Path:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`  
**SHA-256 Digest:** `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`  
**Evaluation Suite:** 60-Task Conversational Intelligence Suite (`m3_conversational_tasks.json`)  
**Hardware Mode:** Native CPU execution  

---

## 1. Baseline Performance Breakdown Across 11 Categories

| Category | Task Count | Passed Tasks | Category Accuracy | Qualitative Observation |
| :--- | :---: | :---: | :---: | :--- |
| **Greetings & Dialogue** | 5 | 0 | **0.0%** | Generates generic fragment continuations; does not return polite conversational replies |
| **Instruction Following** | 5 | 0 | **0.0%** | Fails to obey formatting or output constraints |
| **Follow-up Understanding** | 5 | 0 | **0.0%** | Cannot link subsequent query to preceding turns |
| **Multi-turn Context** | 10 | 0 | **0.0%** | Complete amnesia of entities established in Turn 1 |
| **Grounded QA** | 5 | 0 | **0.0%** | Hallucinates or outputs pretraining token attractors |
| **Basic Reasoning** | 5 | 0 | **0.0%** | Random entity emission |
| **Arithmetic & Logic** | 5 | 0 | **0.0%** | Outputs equation syntax without arithmetic correctness |
| **Summarization** | 5 | 0 | **0.0%** | Repetitive continuation without key concept extraction |
| **Format Adherence** | 5 | 0 | **0.0%** | Cannot generate valid JSON or bulleted structures |
| **Uncertainty Handling** | 5 | 0 | **0.0%** | Does not abstain or admit lack of information |
| **Topic Coherence** | 5 | 0 | **0.0%** | Drifts across disconnected topics |
| **TOTAL OVERALL** | **60** | **0** | **0.0%** | **Clean baseline zero floor** |

---

## 2. Additional Independent Benchmark Baseline Scores

- **Native Intelligence Suite (32 Tasks):** 1/32 (3.1%)
- **Coding Suite Functional Pass (12 Tasks):** 0/12 (0.0%)
- **Coding AST Syntax Validity:** 3/12 (25.0%)
- **Repetitive Attractor Rate:** 53.1% (greedy unpenalized)
- **Neural Harness Zero-Shot QA (30 Tasks):** 0/30 (0.0%)

---

## 3. Evaluation Integrity Notes
- The canonical baseline establishes an honest, unambiguous **0.0%** floor on the 60-task conversational benchmark.
- Any legitimate advancement in instruction following, greeting coherence, or entity recall will be cleanly measurable against this floor without confounding pre-existing accuracy.
