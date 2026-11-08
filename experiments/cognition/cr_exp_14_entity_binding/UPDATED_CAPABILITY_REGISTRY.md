# CR-EXP-14 Phase 8: Updated Capability Registry

## 1. Registry Purpose & Scope
This registry records the validated neural capabilities of the ChakrMicro architecture as established by reproducible, independent evaluations up to and including CR-EXP-14. Capabilities are strictly segregated into **Validated Acquired Capabilities**, **Partially Acquired / Calibrated Capabilities**, and **Unacquired Capabilities / Architectural Floors**.

---

## 2. Neural Capability Status Matrix

| Capability Dimension | Evaluated Status | Benchmark Evidence | Critical Bottleneck / Notes |
| :--- | :---: | :---: | :--- |
| **Conversational Turn-Taking** | **Acquired** | Fluency benchmarks (CR-09 to CR-12) | Correctly handles dialogue formatting and punctuation cadence. |
| **Missing Info Abstention** | **Acquired (Baseline)** | 100.0% on CR-12 Parent (100/100 probes) | Calibrated abstention cues (`"does not specify"`, `"unknown"`) reliably expressed when unanswerable. |
| **In-Distribution Entity Tracking** | **Partially Acquired** | Diagnostic tests (Single entity in-context) | Can associate updated cities with specific training set entities, but degrades when multiple entities are present. |
| **Loose Recency Retrieval** | **Partially Acquired** | 27%-43% on loose benchmark | Can retrieve recent city tokens, but substitutes unrelated subject entities. |
| **Dynamic Subject-Value Binding (OOD)** | **Unacquired (0.00%)** | 0/1,100 probes on sealed benchmark | Fails on unseen entities; hallucinated training subject tokens dominate the output head. |
| **Multi-Entity Tracking (OOD)** | **Unacquired (0.00%)** | 0/200 probes on sealed benchmark | Cannot route distinct entity queries to distinct value slots for novel names. |
| **Counterfactual Reassignment (OOD)** | **Unacquired (0.00%)** | 0/150 probes on sealed benchmark | Overwritten values cannot be decoupled from prior state without subject binding. |
| **Distractor Resistance (OOD)** | **Unacquired (0.00%)** | 0/150 probes on sealed benchmark | Conversational noise compounds attention routing errors. |

---

## 3. Recommended Primary Checkpoint
- **Primary Parent Checkpoint:** CR-12 Candidate B (`cr_exp_12_candidate_b_best_val_copy.pt`, SHA-256: `60271c8b5e3d26b76237dc8f986e0d8d6ada2f08815cea57ec5d9a37102d9d2e`) remains the most capable and well-calibrated baseline model, maintaining 100% calibration on answerability without the abstention regression observed in fine-tuned candidates.
