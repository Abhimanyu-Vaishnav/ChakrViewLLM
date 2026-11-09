# CR-EXP-15 Phase 6: Training Objective Ablation Design

## 1. Experimental Rationale
To isolate which cognitive training capability contributes to later multi-turn state tracking, we construct four controlled training objective arms using the identical baseline architecture ($d_{\text{model}}=192, 6\text{ layers}, 6\text{ heads}, d_{\text{head}}=32$) and the identical parent checkpoint (`CR-12 Candidate B`, SHA-256: `60271c8b...`):

### Candidate A: Standard Language-Model Control (Chit-Chat / Facts)
- **Objective:** Controls for fine-tuning exposure without introducing copy or pointer training.
- **Data:** Conversational facts and natural dialogue exchanges without explicit copy or binding tasks.

### Candidate B: Entity Copying & Entity Selection Curriculum
- **Objective:** Directly trains the autoregressive induction heads on reproducing unfamiliar names and selecting queried entities from multi-entity context.
- **Data:** 50% explicit name reproduction (`Repeat the exact name: {name}`), 50% role-based entity selection (`Between {e1} and {e2}... Who was mentioned first?`).

### Candidate C: Static Subject-Value Binding Curriculum
- **Objective:** Directly trains simultaneous subject-attribute association in a single context without state updates.
- **Data:** Dual-entity contexts (`{e1} lives in {c1}. {e2} lives in {c2}. Where is {target}?`).

### Candidate D: Dynamic State Updates & Interleaved Pointer-Routing Curriculum
- **Objective:** Trains full sequential state overwriting across multi-turn dialogues with interleaved updates.
- **Data:** 5-turn dialogues with multiple entity relocations and delayed queries.

---

## 2. Experimental Controls & Matched Conditions
- **Parent Checkpoint:** CR-12 Candidate B (`cr_exp_12_candidate_b_best_val_copy.pt`, SHA-256: `60271c8b5e3d26b76237dc8f986e0d8d6ada2f08815cea57ec5d9a37102d9d2e`).
- **Optimization Budget:** Exactly 200 optimizer steps per candidate.
- **Batch Size:** Mini-batch size 8, gradient accumulation 2 (effective batch size 16).
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay $0.01$).
- **Learning Rate:** Warmup to $2.5 \times 10^{-5}$ over 20 steps, cosine decay to $2.5 \times 10^{-6}$.
- **Gradient Clipping:** Max norm 0.5.
- **Zero Leakage:** Training pool entities ($N=50$) and cities ($N=30$) are strictly disjoint from the sealed test benchmark ($N=40$ entities, $N=25$ cities).

