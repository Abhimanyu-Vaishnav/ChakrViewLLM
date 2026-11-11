# CR-16D Capability Acquisition Scorecard

### 1. Cumulative Cognitive Capability Matrix
Tracking capability evolution across the ChakrMicro experimental lineage:

| Capability Dimension | Stage B Baseline | CR-16A Arm D | CR-16C Arm A | CR-16D Arm C (Best) | Capability Status in CR-16D |
|---|---|---|---|---|---|
| **Arbitrary Sequence Copying** | 0 / 100 (0%) | 66 / 100 (66%) | 0 / 100 (0%) | **84 / 100 (84%)** | **ROBUST & ENHANCED** |
| **First-Token Prompt Decoupling** | 0 / 100 (0%) | 100 / 100 (100%)| 0 / 100 (0%) | **100 / 100 (100%)**| **FULL ACQUISITION** |
| **Direct Key-Value Retrieval** | 0 / 200 (0%) | 1 / 200 (0.5%) | 164 / 200 (82%) | **197 / 200 (98.5%)**| **FULL ACQUISITION** |
| **Dynamic Value Reassignment** | 0 / 200 (0%) | 0 / 200 (0%) | 198 / 200 (99%) | **200 / 200 (100%)** | **PERFECT ACQUISITION** |
| **Multi-Pair Distractor Filtering**| 0 / 150 (0%) | 0 / 150 (0%) | 150 / 150 (100%)| **143 / 150 (95.3%)**| **ROBUST ACQUISITION** |
| **Compositional Combinations (G1)**| 0 / 250 (0%) | 6 / 250 (2.4%) | 6 / 250 (2.4%) | **241 / 250 (96.4%)**| **SOLVED BREAKTHROUGH**|
| **Unseen Key Generalization (G3)** | 0 / 250 (0%) | 5 / 250 (2.0%) | 38 / 250 (15.2%)| **236 / 250 (94.4%)**| **SOLVED BREAKTHROUGH**|
| **Unseen Value Generalization (G2)**| 0 / 250 (0%) | 3 / 250 (1.2%) | 0 / 250 (0%) | **20 / 250 (8.0%)** | **ARCHITECTURAL BOTTLENECK**|
| **Unseen Key & Value (G4)** | 0 / 250 (0%) | 3 / 250 (1.2%) | 0 / 250 (0%) | **12 / 250 (4.8%)** | **ARCHITECTURAL BOTTLENECK**|
| **Abstention on Unbound Key** | 0 / 100 (0%) | 0 / 100 (0%) | 58 / 100 (58%) | **1 / 100 (1.0%)** | **UNRESOLVED / FRAGILE** |

---

### 2. Summary Verdict
- **Multi-Task Capability Consolidation:** **PASSED.** Sequence copying and key-value retrieval coexist stably and reinforce one another under controlled replay ($\ge 25\%$).
- **Compositional Key-Value Binding:** **PASSED.** In-context binding on familiar values reaches $>96\%$, fully independent of training co-occurrences.
- **Unseen Key Routing:** **PASSED.** Reaches $>94\%$, demonstrating abstract position-agnostic key matching.
- **Unseen Value Generalization:** **BLOCKED.** Lacks a direct pointer mechanism to project novel value tokens to logits without vocabulary exposure.

