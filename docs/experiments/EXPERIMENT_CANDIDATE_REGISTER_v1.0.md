# ChakrView Experiment Candidate Register v1.0
## Formal Register of Prospective Research Proposals & Pre-Flight Status

**Repository:** `ChakrView`  
**Location:** `docs/experiments/EXPERIMENT_CANDIDATE_REGISTER_v1.0.md`  
**Parent Governance Standard:** `docs/experiments/EXPERIMENT_GOVERNANCE_PROTOCOL_v1.0.md`  
**Version:** 1.0.0  
**Effective Date:** 2026-10-31  
**Status:** Living Candidate Register (Proposals Only — None Approved for Execution)  

---

## 1. Candidate Status Taxonomy

Every candidate experiment registered in this document is classified under one of four non-executable governance states:
- **`PROPOSED`:** Scientific concept drafted; research question, $H_1$, and $H_0$ formulated.
- **`NEEDS_DESIGN`:** Concept approved in principle; detailed factorial design, split construction, or evaluator specification is incomplete.
- **`BLOCKED`:** Blocked by upstream dataset governance holds, missing baseline evaluations, or prerequisite track dependencies.
- **`READY_FOR_REVIEW`:** Fully specified according to Protocol v1.0 Section 13 template; ready for formal Pre-Execution GO/NO-GO review.

*Strict Policy: The status `APPROVED` does not exist in this register. Approval to execute is granted solely via the formal signed Pre-Execution GO record defined in Protocol v1.0 Section 14.*

---

## 2. Prospective Experiment Candidates

### Candidate 1: Controlled Entity-Copying Baseline (CECB)
- **Candidate ID:** `CAND_01_CONTROLLED_ENTITY_COPYING`
- **Research Track:** Track 1 (Entity Association and Copying)
- **Research Question:** Can ChakrMicro learn to reliably copy an arbitrary token from prompt context into completion when static template noise is eliminated?
- **Hypothesis ($H_1$):** Training on minimal-template key-value pairs (`Input: [KEY] [VAL] -> Output: [VAL]`) will produce $>75\%$ copy accuracy on held-out unseen tokens, proving the architecture possesses functional copying capability.
- **Null Hypothesis ($H_0$):** Model copy accuracy remains $<5\%$, indicating architectural inability to route attention for direct token copying.
- **Independent Variable:** Token copy supervision format (minimal template vs. full sentence framing).
- **Control Condition:** Matched compute budget using full conversational sentence framing.
- **Intervention Condition:** Minimal key-value token framing with prompt loss masked.
- **Primary Metric:** Exact token copy accuracy on held-out test tokens (word-boundary matched).
- **Secondary Metrics:** Target token probability rank (out of 4096), attention entropy across context layers.
- **Evaluator Requirements:** Exact token equality script (`eval_token_copy.py`), verified SHA-256, boundary regex.
- **Dataset Requirements:** 1,000 synthetic disjoint key-value pairs; strict train/test entity token separation.
- **Parent Checkpoint:** Canonical Step 10,000 (`96357705a683482e...`).
- **Confounders:** Token frequency bias in vocabulary, subword token splitting.
- **Status:** **`READY_FOR_REVIEW`**

---

### Candidate 2: Entity-Only vs. Template-Heavy Supervision Factorial (ETSF)
- **Candidate ID:** `CAND_02_SUPERVISION_WEIGHTING_FACTORIAL`
- **Research Track:** Track 2 (Sensitivity and Causal Diagnosis)
- **Research Question:** Does masking template loss (supervising entity tokens only) resolve the M5.1 gradient dilution problem and increase prompt sensitivity?
- **Hypothesis ($H_1$):** Supervising only the target entity token will increase target token gradient magnitude by $>4\times$ and increase probability sensitivity delta $\Delta p$ beyond 0.05.
- **Null Hypothesis ($H_0$):** Entity-only supervision causes representation collapse or fails to improve target token sensitivity ($\Delta p \le 0.001$).
- **Experimental Design:** $2 \times 2$ Factorial Design:
  - *Arm 1:* Unmasked Loss $\times$ Full Sentence Context.
  - *Arm 2:* Unmasked Loss $\times$ Minimal Context.
  - *Arm 3:* Masked Template Loss $\times$ Full Sentence Context.
  - *Arm 4:* Masked Template Loss $\times$ Minimal Context.
- **Primary Metric:** Prompt probability sensitivity delta $\Delta p$ on target entity.
- **Secondary Metrics:** Validation loss on entity tokens, generation repetition rate.
- **Evaluator Requirements:** Logit probability extraction harness measuring top-10 next-token distribution.
- **Dataset Requirements:** DPCE-structured dataset with explicit token mask labels.
- **Parent Checkpoint:** Canonical Step 10,000.
- **Status:** **`NEEDS_DESIGN`** (Requires formal loss masking implementation and gradient telemetry logging).

---

### Candidate 3: Context-Position Sensitivity Probe (CPSP)
- **Candidate ID:** `CAND_03_CONTEXT_POSITION_SENSITIVITY`
- **Research Track:** Track 1 & Track 2 (Positional Dynamics)
- **Research Question:** Does ChakrMicro's copy fidelity degrade as the distance between context entity and target generation increases from 10 to 200 tokens?
- **Hypothesis ($H_1$):** Copy accuracy exhibits a monotonic decay as context distance increases, isolating rotary embedding frequency scaling as a causal constraint.
- **Null Hypothesis ($H_0$):** Copy accuracy is invariant to token distance (either uniformly high or uniformly zero).
- **Independent Variable:** Token offset distance between context entity and response prefix ($N \in \{10, 50, 100, 200\}$).
- **Control Condition:** Matched entity vocabulary evaluated at offset 10.
- **Primary Metric:** Copy accuracy stratified by token offset.
- **Secondary Metrics:** Attention weight magnitude between generation token and entity token.
- **Evaluator Requirements:** Distance-stratified evaluation suite.
- **Dataset Requirements:** Synthetic passages with controlled padding lengths.
- **Parent Checkpoint:** Candidate checkpoint from `CAND_01` (Prerequisite: `CAND_01` must achieve `PROMOTE`).
- **Status:** **`BLOCKED`** (Blocked pending successful completion of `CAND_01`).

---

### Candidate 4: Tokenizer Subword Segmentation Diagnostic (TSSD)
- **Candidate ID:** `CAND_04_TOKENIZER_SEGMENTATION_DIAGNOSTIC`
- **Research Track:** Track 2 (Sensitivity and Causal Attribution)
- **Research Question:** Does subword fragmentation (entities split into 2–4 tokens) explain the 0/200 extraction failure in Exp35 compared to single-token entities?
- **Hypothesis ($H_1$):** Single-token entities achieve significantly higher copy accuracy ($>30\%$ delta) than multi-token fragmented entities under identical training conditions.
- **Null Hypothesis ($H_0$):** Fragmentation level produces no significant difference in copy accuracy.
- **Independent Variable:** Entity tokenization length (1-token vs 2-token vs 3-token subwords).
- **Control Condition:** Single-token entities drawn from high-frequency BPE vocabulary.
- **Intervention Condition:** Multi-token entities requiring subword assembly.
- **Primary Metric:** Extraction accuracy stratified by entity subword count.
- **Secondary Metrics:** Token-level cross-entropy loss across fragmented tokens.
- **Evaluator Requirements:** Subword-aware token scoring script.
- **Dataset Requirements:** Vocabulary-audited entity lists partitioned by BPE token length.
- **Parent Checkpoint:** Canonical Step 10,000.
- **Status:** **`PROPOSED`**

---

### Candidate 5: Multi-Turn State Retention Pilot (MSRP)
- **Candidate ID:** `CAND_05_MULTI_TURN_STATE_RETENTION`
- **Research Track:** Track 3 (Multi-Turn State Retention)
- **Research Question:** Can ChakrMicro retrieve an entity declared in Turn 1 when queried in Turn 3 across an intervening dialogue turn?
- **Hypothesis ($H_1$):** Models trained with multi-turn conversation formatting maintain $>40\%$ recall on Turn 1 entities in 3-turn dialogues.
- **Null Hypothesis ($H_0$):** Model exhibits recency collapse, achieving $<5\%$ recall on Turn 1 entities.
- **Independent Variable:** Multi-turn dialogue history depth (1-turn vs. 3-turn vs. 5-turn).
- **Control Condition:** 1-turn direct question-answering.
- **Primary Metric:** Entity retrieval accuracy on Turn 1 facts at Turn 3.
- **Secondary Metrics:** Conversational coherence, repetition loops, EOS rate.
- **Evaluator Requirements:** Multi-turn conversational harness with dialogue history state tracking.
- **Dataset Requirements:** Curated multi-turn dialogues with verified state assertions.
- **Parent Checkpoint:** Checkpoint promoted from Track 1/Track 2.
- **Status:** **`BLOCKED`** (Blocked pending Track 1 promotion; running multi-turn without proven 1-hop copying is premature).

---

### Candidate 6: Unseen-Entity Generalization Benchmark (UEGB)
- **Candidate ID:** `CAND_06_UNSEEN_ENTITY_GENERALIZATION`
- **Research Track:** Track 4 (Generalization & OOD Transfer)
- **Research Question:** Does associative entity copying generalize to entity categories completely absent from training (e.g. training on personal names, testing on chemical compounds)?
- **Hypothesis ($H_1$):** Attention-based copy mechanisms generalize across semantic domains, achieving $<10\%$ performance degradation on OOD entity classes.
- **Null Hypothesis ($H_0$):** Model over-fits specific training entity embeddings, exhibiting $>80\%$ performance degradation on OOD classes.
- **Independent Variable:** Entity semantic domain (In-Distribution vs. Out-of-Distribution).
- **Control Condition:** In-Distribution entity test split.
- **Intervention Condition:** Out-of-Distribution semantic class test split.
- **Primary Metric:** Generalization retention ratio ($\text{Accuracy}_{\text{OOD}} / \text{Accuracy}_{\text{ID}}$).
- **Evaluator Requirements:** Lexically disjoint benchmark suite.
- **Dataset Requirements:** Bipartite dataset partitioned by semantic entity classes.
- **Parent Checkpoint:** Checkpoint promoted from `CAND_01` / `CAND_02`.
- **Status:** **`BLOCKED`** (Blocked pending successful demonstration of in-distribution copying).

