# CR-EXP-14 Phase 5: Training Curriculum & Controlled Interventions Design

## 1. Scientific Hypothesis & Problem Formulation

In CR-EXP-14 Baseline Evaluations, all existing ChakrMicro checkpoints (CR-11 Candidate A, CR-12 Candidate A, CR-12 Candidate B) achieved **0.00% (0/1,100)** strict accuracy on multi-turn entity state tracking. Qualitative inspection revealed that when queried about an entity in context, the model outputs high-frequency training names (e.g., `Ananya`, `Komal`, `Aarav`) rather than dynamically copying and conditioning upon the queried subject entity.

### Primary Hypothesis
ChakrMicro ($3.44\text{M}$ parameters, 6 layers, 6 attention heads, $d_{\text{model}}=192$, rotary positional embeddings) can acquire dynamic subject-value binding if trained with controlled interventions that break static entity priors and teach induction heads to route information from the dialogue history and query.

---

## 2. Controlled Intervention Strategies (Candidates A, B, and C)

We design three distinct, principled interventions to systematically address entity-value binding:

### Candidate A: Explicit Subject-Value Conditioning (ESVC)
- **Concept:** Force the model to condition directly on the query subject by training on varied single-turn and multi-turn state updates where the response template strictly repeats the query entity name: `"[Subject] is currently in [Target City]."`.
- **Target Mechanism:** Encourages attention heads to copy the subject representation from the query prompt to the generated output tokens and bind it to the recent context value.
- **Data Balance:** Single-turn, multi-turn updates with diverse conversational phrasings.

### Candidate B: Interleaved State-Update Curriculum (ISUC)
- **Concept:** Expose the model to complex multi-entity multi-turn dialogues with interleaved updates, distractor dialogue turns, and delayed queries across 2, 3, and 4 turns.
- **Target Mechanism:** Forces the model to maintain separated memory slots for multiple subjects and retrieve the correct slot dynamically based on the subject token in the query.
- **Data Balance:** Multi-entity state tracking, counterfactual reassignments, interleaved dialogue distractors, and explicit negative/missing-information abstentions.

### Candidate C: Structured Entity Delimiters (SED)
- **Concept:** Wrap entities and values in explicit delimiter tags during dialogue context and response: `<ent>Subject</ent>` and `<val>Location</val>`.
- **Target Mechanism:** Delimiters act as explicit structural markers, providing clear token boundary cues to the self-attention mechanism without altering the vocabulary (verified to tokenize losslessly into existing ByteBPE tokens: `<ent>`, `</ent>`, `<val>`, `</val>`).
- **Data Balance:** Identical multi-entity update complexity as Candidate B, formatted with structured delimiters.

---

## 3. Strict Dataset Hygiene & Zero-Leakage Guarantee

To prevent contamination and guarantee zero data leakage:
- **Disjoint Entities:** All 40 entities in the sealed benchmark are strictly excluded from the training set. Training uses a pool of 60 completely distinct names.
- **Disjoint Locations:** All 21 locations in the sealed benchmark are strictly excluded from the training set. Training uses a pool of 40 completely distinct Indian cities.
- **Validation Split:** A held-out validation set of 200 dialogues using separate validation entities/cities is generated to monitor loss and prevent overfitting.
- **Zero Exposure:** The 1,200 sealed test probes remain completely untouched and unobserved by any training script.

---

## 4. Controlled Training Parameters
All three candidates will be trained under strictly identical optimization conditions:
- **Parent Checkpoint:** Immutable copy of CR-12 Candidate B (`experiments/cognition/cr_exp_14_entity_binding/checkpoints/baseline_cr12_b/cr_exp_12_candidate_b_best_val_copy.pt`, SHA-256: `60271c8b...`).
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay $0.01$).
- **Learning Rate:** $2.5 \times 10^{-5}$ with linear warmup (20 steps) and cosine decay.
- **Batch Size:** 8 (gradient accumulation: 2, effective batch size 16).
- **Training Budget:** 250 optimizer steps.
- **Gradient Clipping:** Max norm 0.5.
- **Validation Frequency:** Every 25 steps; best validation checkpoint checkpointed.
