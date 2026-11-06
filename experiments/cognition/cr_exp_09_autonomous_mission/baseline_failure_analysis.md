# CR-EXP-09: BASELINE FAILURE ANALYSIS & RESEARCH BLUEPRINT
## Systematic Diagnosis of Bottlenecks in the Current Best Checkpoint (Arm B)

- **Target Checkpoint**: `cr_exp_08_arm_b_best_val.pt` (SHA-256: `98bd76f99d43dedd...`)
- **Baseline Capabilities Validated**:
  - Conversational Greeting & Persona Consistency: **100.0%**
  - Familiar Relational Binding: **71.67%**
  - OOD Entity Generalization: **63.33%**
  - Counterfactual Reassignment: **66.67%**

---

### 1. The Core Bottlenecks Blocking Advanced Capabilities

| Identified Bottleneck | Observable Symptom in Arm B | Underlying Architectural / Data Root Cause |
|---|---|---|
| **1. Missing-Info Fabrication** (0.0% Abstention) | Hallucinates a random city when asked about an unmentioned subject. | **Delimiter Condition Misalignment**: `"Answer: "` was trained exclusively with city names, conditioning the model that silence/abstention is forbidden. |
| **2. Distractor Recency Bias** (18.33% Resistance) | Outputs the most recent entity's value instead of the queried entity's value in multi-subject contexts. | **Unbalanced Attention Routing**: Without explicit distractor penalty or multi-subject contrastive examples during training, attention layers defaults to local recency. |
| **3. Lack of Multi-Turn State Tracking** | Unable to track updates across conversational dialogue turns. | **Single-Turn Training Format**: CR-EXP-08 trained exclusively on single-turn mapping prompts (`Mapping: ... Query: ... Answer: `). |
| **4. Zero Multi-Step Reasoning & Arithmetic** | Fails completely on numerical transformations or logical deduction. | **Absence of Compositional Supervision**: The model has never seen intermediate step sequences or numerical relationships. |
| **5. Token Continuation Suffix Artifacts** | Appends trailing conversational fragments (e.g., `"Kolkataknow"`, `"Nashikatchon"`). | **Unterminated Target Strings**: Targets lacked clean stop delimiters (e.g. newline `\n` or `<endoftext>`). |

---

### 2. The Four CR-EXP-09 Research Workstreams

To advance ChakrView systematically toward higher intelligence levels, CR-EXP-09 will execute four focused research workstreams:

1. **Workstream A: Epistemic Abstention & Delimiter Calibration**
   - Teach the model to explicitly state uncertainty when information is missing (`"The provided context does not specify..."`), breaking the mandatory-fabrication attractor.
2. **Workstream B: Distractor-Robust Multi-Entity Binding**
   - Train on counterbalanced 2-entity and 3-entity contexts with positional permutation to suppress recency bias and force entity-name attention grounding.
3. **Workstream C: Multi-Turn Conversational State Tracking**
   - Train on multi-turn dialogues where facts are established in Turn 1, updated in Turn 2, and queried in Turn 3, evaluating working memory retention across turns.
4. **Workstream D: Structured State Transformation & Elementary Reasoning**
   - Train on elementary state updates (e.g., item transfers: *"Ravi has 5 books, gives 2 to Aman"*) and basic conditional deductions, evaluating intermediate state tracking without external tools.

