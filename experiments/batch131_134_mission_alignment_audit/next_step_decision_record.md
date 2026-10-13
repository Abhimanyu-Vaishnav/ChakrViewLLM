# Batch 131–134 — Next-Step Hypothesis & Decision Record

- **Date**: October 13, 2026
- **Status**: RATIFIED DECISION RECORD (Created BEFORE any new training)
- **Decision Choice**: **PATH B — Return to Neural-Core Learning Diagnostics (Relational Rule Induction Benchmark)**

---

## 1. Evaluation of Candidate Paths

In accordance with PART D of the batch mandate, three paths were evaluated:

### Path A: Continue Operand-Binding Research with a Tightly Controlled Experiment
- *Rationale for Rejection*:
  1. Over 36 batches (Batches 95–130) have studied operand binding on addition.
  2. The scientific findings are conclusive: under tiny parameter scale (3.44M parameters) and BPE subword fragmentation, two-digit argument copying reaches a persistent bottleneck ($\approx 20\%$ exact calls) that cannot be solved by small 200-step SFT tweaks or naive data expansion without substantial capacity and step-schedule scaling.
  3. Continuing to optimize a calculator tool-call string is an acute manifestation of task-focus drift that delays foundational progress on the sovereign intelligence core.

### Path B: Return to Neural-Core Learning Diagnostics (Selected)
- *Rationale for Selection*:
  1. ChakrView's core thesis requires an indigenous neural model that *learns generalizable rules*, not merely memorizing surface outputs.
  2. We must determine whether the 3.44M parameter architecture is capable of **conditional relational induction**: given a structured syntactic rule, can the model learn the abstract relational transformation and generalize to unseen symbols, or does it merely memorize seen input-output pairs?
  3. This diagnostic directly separates:
     - Training example memorization.
     - Generalization to novel symbol combinations.
     - Invariance to irrelevant distractors/framing.
     - Genuine conditional reasoning vs constant-output collapse.
  4. This provides a direct prerequisite for mathematics (symbolic manipulation), coding (AST transformations), and science (relational reasoning).

### Path C: Work on Another Foundational Capability (e.g., PPB or Federation)
- *Rationale for Rejection*:
  1. As established in the Mission-Alignment Audit, the cognitive scaffolding, PPB storage, and distributed federation layers are already implemented, tested, and far ahead of the neural brain. Adding more infrastructure without improving the brain's learning capability creates further imbalance.

---

## 2. Benchmark Specification: Relational Rule Induction (RRI-1)

To diagnose neural-core learning capacity without the confounds of calculator JSON schemas or arithmetic arithmetic lookup:

### 1. Task Definition
A clean, deterministic symbolic relation task: **Abstract Inversion and Relational Mapping (AIRM)**.
- **Rule Formulation**:
  A sequence of distinct symbolic entities $X \to Y$ under an explicit relation operator (e.g., `MAP`, `REVERSE`, `SELECT`).
  Specifically:
  - Task 1: **Conditional Reversal**: Given a sequence of $K$ distinct token entities `[e1, e2, e3]`, emit the reversed sequence `[e3, e2, e1]`.
  - Task 2: **Key-Value Associative Routing**: Given context bindings `K1: V1, K2: V2` and query `QUERY: K1`, emit `ANSWER: V1`.
- **Vocabulary Domain**:
  Use isolated, non-numeric linguistic tokens (e.g. single clean words from the ChakrView vocabulary like `alpha`, `beta`, `gamma`, `delta`, `omega`, `vector`, `matrix`, `tensor`, `scalar`) that tokenize into single unambiguous tokens.

### 2. Disjoint Splits & Leakage Resistance
- **Train Pool**: 16 entities. Total possible pairs/triplets strictly partitioned.
- **Test Partition 1 (Familiar Combinations)**: Combinations seen during training to measure memorization/fit capacity.
- **Test Partition 2 (Unseen Combinations of Known Entities)**: Novel pairings of entities seen in training to measure compositional generalization.
- **Test Partition 3 (Held-Out Entities)**: Entities completely withheld from training to measure true abstract rule induction (zero-shot entity transfer).
- **Test Partition 4 (Negative / Malformed Queries)**: Requests with missing keys or invalid instructions to verify fail-closed non-collapse.

### 3. Hypothesis & Success Criteria
- **Hypothesis $H_1$**: The 3.44M parameter `ChakrMicro` architecture possesses the representational capacity to learn an abstract permutation/routing rule and achieve $>80\%$ accuracy on unseen combinations of familiar entities.
- **Alternative Hypothesis $H_0$ (Memorization Ceiling)**: The model achieves $>90\%$ on familiar combinations (Partition 1) but collapses to $<20\%$ or repeats frequent tokens on unseen combinations (Partition 2), demonstrating that current SFT regimes induce shallow token-attractor memorization rather than relational attention circuits.

---

## 3. Experimental Controls

1. **Initialization**: Canonical `training_checkpoint_step_10000.pt` (SHA-256: `96357705a683...`) and Stage 1 copy-task `copy_task_sft_step150.pt`.
2. **Optimizer**: AdamW ($lr=3\times 10^{-4}$, weight decay 0.01, gradient clipping 1.0).
3. **Budget**: Standardized optimization steps with logged target-token exposure.
4. **Evaluation**: Free-running greedy decoding from cold-reloaded checkpoint; exact string and token matches logged per example.
5. **Denominators & Metrics**: Complete numerator/denominator reporting with binomial confidence intervals.
