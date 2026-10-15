# Mission Summary: Batches 541–580
## Neural Intent, Structured Understanding & Tool-Use Learning

**Repository:** `D:\Project\ChakrView`  
**Branch:** `dev/v0.6-v0.7-validation`  
**Starting Commit:** `749ba8c` — `feat(batch501-540): audit scratchpad leakage, train multi-task rule curriculum, and evaluate 60-item generalization`  
**Current Milestone:** Batches 541–580  
**Canonical Step 10,000 Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`  
- SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified Immutable)  
**Previous Evaluated Candidate:** `experiments/batch501_540/checkpoints/candidate_step200_compositional_rule.pt`  
- SHA-256: `b18e69ebf4f0b396465af3c323f38bcfa84d5a38eabbd4a4220a10d6bfafee98`  
**New Trained Candidate:** `experiments/batch541_580/checkpoints/candidate_step250_intent_understanding.pt`  
- SHA-256: `9c3204906e8e94f5e80c5db9f6ffb0fa6903c6fe3159206fb164e44d52b8357e`  

---

### 1. Executive Summary & Core Research Findings

In Batches 541–580, we addressed the foundational question: **Can the 3.4M parameter ChakrMicro neural core learn to understand natural language intent, generate versioned structured representations (JSON), and safely dispatch actions to governed deterministic tools on held-out user requests?**

Key Findings:
1. **Emergence of Structured Syntactic Competence:**
   - The starting baseline candidate (`candidate_step200_compositional_rule.pt`) had **0.0% valid JSON rate** and **0.0% intent accuracy** on structured contracts.
   - The trained candidate (`candidate_step250_intent_understanding.pt`) achieved **88.1% Valid JSON emission** (52/59) and **88.1% clean EOS termination** on the frozen held-out test suite.
2. **Intent Generalization on Unseen Phrasings:**
   - Intent classification accuracy reached **37.3% overall** (22/59) on novel phrasings and unseen operand sets:
     - `calculator`: **65.5% (19/29)**
     - `knowledge_lookup`: **25.0% (1/4)**
     - `concept_definition`: **20.0% (1/5)**
     - `direct_chat`: **20.0% (1/5)**
     - Baseline candidate: **0.0% across all categories**.
3. **The Parameter Capacity Limit on Autoregressive Argument Copying:**
   - While the model reliably selects the `calculator` tool and syntax, **argument exact copy accuracy on unseen operands remains low (3.4%)**.
   - Similar to Batches 501–540's findings on arithmetic hallucinations, the pure autoregressive Transformer without pointer heads or cross-attention copy circuits struggles to copy multi-digit strings verbatim (`22 * 15` instead of `36 / 12`).
4. **Governed Runtime Safety & Complete Attribution:**
   - Through `NeuralIntentRouter` (`chakrview/runtime/neural_intent_router.py`), neural outputs are strictly validated against allowlisted tools (`CalculatorTool`, `TextUtilityTool`).
   - Consequential commands (such as shell scripts or OS modifications) are rejected under security policy.
   - Pre-flight verification by `EvidenceGovernedAnsweringGate` correctly intercepts and refutes false premises (e.g. "Capital of Mars" -> `REFUTE`, `ComponentOrigin.FALLBACK_POLICY`).
   - Every response provides transparent component provenance separating neural proposals from deterministic tool execution.

---

### 2. Comprehensive Held-Out Evaluation Matrix (N=59)

| Evaluation Metric | Baseline Candidate (`b18e69eb...`) | Trained Candidate (`9c320490...`) | Delta |
| :--- | :--- | :--- | :--- |
| **Valid JSON Contract Emission** | 0.0% (0/59) | **88.1% (52/59)** | **+88.1%** |
| **Overall Intent Classification Accuracy** | 0.0% (0/59) | **37.3% (22/59)** | **+37.3%** |
| - `calculator` Intent (N=29) | 0.0% (0/29) | **65.5% (19/29)** | **+65.5%** |
| - `knowledge_lookup` Intent (N=4) | 0.0% (0/4) | **25.0% (1/4)** | **+25.0%** |
| - `concept_definition` Intent (N=5) | 0.0% (0/5) | **20.0% (1/5)** | **+20.0%** |
| - `direct_chat` Intent (N=5) | 0.0% (0/5) | **20.0% (1/5)** | **+20.0%** |
| - `text_utility` Intent (N=6) | 0.0% (0/6) | 0.0% (0/6) | 0.0% |
| - `clarification_needed` Intent (N=5) | 0.0% (0/5) | 0.0% (0/5) | 0.0% |
| - `unsupported` Intent (N=5) | 0.0% (0/5) | 0.0% (0/5) | 0.0% |
| **Exact Argument Copy Accuracy** | 0.0% (0/59) | **3.4% (2/59)** | **+3.4%** |
| **Clean True EOS Rate (Token 1)** | 100.0% (59/59) | **88.1% (52/59)** | -11.9% |

---

### 3. Leakage and Dataset Integrity Audit

- **Zero Overlap Confirmed:** Exact prompt overlap across splits:
  - `train_val_prompt_overlap`: 0
  - `train_test_prompt_overlap`: 0
  - `val_test_prompt_overlap`: 0
- **Operand Space Separation:**
  - Training operands: `[10-49]`, `[100-499]`
  - Held-out test operands: `[50-99]`, `[500-999]`
- **Audit File:** `experiments/batch541_580/reports/DATASET_QUALITY_AND_LEAKAGE_AUDIT.json`

---

### 4. End-to-End Runtime Integration & Safety Verification

Tested via `NeuralIntentRouter` and regression tests (`tests/test_batch541_580_neural_intent.py`):
- **Tool Allowlist:** Only registered tools (`calculator`, `text_utility`) can be invoked.
- **AST Security Boundary:** Arbitrary Python code (e.g. `__import__('os').system(...)`) is blocked by `CalculatorTool` AST validation.
- **Evidence Gate Pre-Flight:** Refutation of false premises is guaranteed prior to neural generation.
- **User Modification Integrity:** User changes in `chakrview/runtime/session.py` remain preserved and unmodified.

---

### 5. Architectural Decision & Highest-Value Next Milestone

**Decision:** **Accept `candidate_step250_intent_understanding.pt` as an internal research checkpoint.**
The candidate demonstrably improves on structured JSON emission (88.1% vs 0.0%) and intent classification on novel phrasings (37.3% vs 0.0%).

**Root Cause of Low Argument Accuracy (3.4%):**
Autoregressive token generation in a 3.4M model is an associative memory, not a dynamic copy register. Attempting to generate novel numbers through standard next-token prediction causes digit drifting.

**Single Highest-Value Next Capability Milestone (Batches 581–620):**
Implement **Dynamic Pointer Binding for Slot Extraction**: Connect the neural core's hidden representations directly to the input tokens via a pointer head (`ChakrPointerHead` in `chakrview/runtime/pointer_mechanism.py`), allowing the neural core to *point* to exact spans in the user query rather than regenerating strings character-by-character.
