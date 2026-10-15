# Failure Analysis: Batches 701–740
## Neural Subtask Decomposition, Generalization & Learnability Investigation

**Analysis Date:** 2026-10-15  
**Candidate:** `ChakrSubtaskDecompositionModel` (`candidate_step150_neural_planner.pt`)  

---

### 1. Honest Assessment of Neural Learnability

Our investigation revealed critical boundaries between what the small `ChakrMicro` (192-dim, 6-layer) neural core can learn versus its limitations:

1. **Subtask Action & Tool Identification (Learned: 56–68%):**
   - The model learns to associate calculation phrases (*"Calculate", "Evaluate", "add"*) with `action: tool_call` and `tool: calculator`.
   - It reliably shifts to `tool: text_utility` on strings (*"Take the string...", "Draft phrase..."*).
2. **Dependency Edge Prediction (Learned: 56–100%):**
   - The model learns sequential chaining ($st\_02$ depends on $st\_01$) when connective cues are present (*"then", "and finally", "परिणाम में"*).
3. **Primary Failure Mode: Goal Type Generalization on Unseen Scenarios (Raw Acc: 36–48%):**
   - On 3-hop arithmetic queries (*"Multiply 12 by 4, then divide the result by 2, and finally add 10"*), the fixed 2-subtask prediction horizon forces truncation. The model predicts a 2-step plan, failing exact 3-step coverage.
   - On completely novel phrases with unknown tools (*"Apply unknown proprietary transformation on secret_payload"*), the model confused novel actions with text transformations, missing the clarification requirement unless explicit missing keywords were present.

---

### 2. The Power of the Hybrid Neural + Validator Architecture

When raw neural predictions were passed to the deterministic `TaskGraphValidator`:
- **Raw Neural Accuracy:** **48.0%** (12/25)
- **Deterministic Heuristic Planner:** **20.0%** (5/25, failed on Hindi and 3-hop structures)
- **Hybrid Neural + Validator Pipeline:** **80.0%** (20/25)

The validator intercepts malformed edge references, validates tool permissions, and enforces DAG acyclicity, allowing the neural model's generalization capabilities to operate safely in production.
