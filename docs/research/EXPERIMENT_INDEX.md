# ChakrView Research Experiment Index & Evidence Registry

**Document Status**: Official Research Archive Index
**Scope**: Experimental Batches, Forensic Audits, and Scaling Investigations (Exp01 – Exp30)
**Retention Policy Reference**: [`docs/development/EXPERIMENT_ARCHIVING_POLICY.md`](../development/EXPERIMENT_ARCHIVING_POLICY.md)

---

## 1. Research Overview & Epistemic Demarcation

ChakrView separates **canonical production capabilities** ($\Delta W = 0$) from **exploratory research investigations**. This registry indexes all experimental runs from Exp01 to Exp30, explicitly demarcating:

1. **Reproduced vs Reported Evidence**: Independent replications (e.g. Exp29 auditing Exp28) vs initial pilot observations.
2. **In-Distribution Performance vs Held-Out Generalization**: Fixed-template copy saturation vs out-of-vocabulary evaluation.
3. **Experimental Mechanisms vs Production Runtime**: Experimental pointer networks are strictly quarantined from the canonical CPU inference pipeline.
4. **Hypothesis Verdict**: Clearly distinguishes validated narrow capabilities from rejected hypotheses and forensic audits.
5. **Storage Location**: Categorizes artifacts into versioned evidence (Git), local research archive (disk), and public release downloads.

---

## 2. Master Experiment Registry (Exp01 – Exp30)

| Experiment ID | Title & Hypothesis | Tested Mechanism | Key Empirical Outcome | Epistemic Verdict & Status | Evidence Location & Artifact Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp01–Exp05** | Single-Field & String Copy SFT | Autoregressive SFT with loss masking | High in-distribution copying (98%+); zero transfer to out-of-vocabulary entities. | **`VALIDATED IN-DISTRIBUTION`** (Fails Out-of-Distribution) | Versioned Git (`experiments/research_v08_json_sft/`) |
| **Exp06–Exp09** | Synthetic Copy Curriculum & Attention Overfit | Aggressive copy loss optimization | Single-token copying saturated; severe catastrophic forgetting on general pretraining text. | **`REJECTED HYPOTHESIS`** (Destabilizes General Language) | Versioned Git (`experiments/research_v08_json_sft/`) |
| **Exp10–Exp12** | Multi-Token Span Pointer & Constrained Hybrid | Prototype pointer attention heads | Successfully extracted multi-token spans on template text; degraded perplexity on prose. | **`QUARANTINED PROTOTYPE`** (Excluded from Production) | Versioned Git (`experiments/research_v08_json_sft/`) |
| **Exp16–Exp18** | Scaled Instruction SFT & Short Scientific Explanation | Out-of-distribution reasoning probes | **20 / 20 Human Rejections** (100% failure rate). Base model cannot synthesize ungrounded scientific explanations. | **`FORMAL REJECTION / NEGATIVE BOUND`** | Versioned Git (`release/chakrview-v0.8.0/`) & Local Archive (`v08_release_candidate_readiness_*`) |
| **Exp19–Exp21** | Debiased SFT Curriculum & Single-Task Binding | Balanced slot frequencies under long context | Solved prefix bias; uncovered greedy attractor collapse loops (mitigated via n-gram blocking). | **`DIAGNOSTIC SUCCESS`** (N-gram blocking integrated) | Versioned Git (Exp21 committed in `aa47d7c`); Local Archive (`exp19_debiased_*`, `exp20_*`) |
| **Exp22–Exp23** | Attention Copy Loss Gradient Diagnostic | Auxiliary copy loss gradients | Gradients from auxiliary copy loss destabilized early RoPE attention layers. | **`DIAGNOSTIC FAILURE`** (Explains Overfitting) | Versioned Git (Committed in commit `aa47d7c`); Local Checkpoints (`exp22_*`) |
| **Exp24–Exp25** | Copying vs Cognitive Routing & Vocab Generalization | Disjoint vocabulary generalization | Pointer copy drops to 12.5% on novel entities; cognitive router provides superior fallback. | **`ARCHITECTURAL PIVOT`** (Router favored over Pointer) | Versioned Git (Exp24 committed in `feb0939`); Local Archive (`exp25_*`) |
| **Exp26** | ChakrMicro-15M Architecture Scaling Hypothesis | Scaling hidden dimension (192 $\to$ 288, 8 layers) | Established parameter layout and CPU latency baseline for next-generation 15M core. | **`FUTURE ROADMAP HYPOTHESIS`** (Valid Configuration) | Versioned Git (Committed in `255537a` & `b771388`); Local Logs (`mission_10h_*`) |
| **Exp27** | Forensic Pointer Research & Gradient Diagnostic | Template leakage audit & gradient traces | Discovered pointer heads overfit fixed template delimiters; failed under distractor shift. | **`CRITICAL FORENSIC AUDIT`** (Leakage Risk Identified) | Local Archive (`exp27_forensic_pointer_research_*`) |
| **Exp28** | Fair Baseline vs Pointer Comparative Study | Arm A (Canonical LM) vs Arm B (Pointer LM) | Under randomized distractor positions, Arm B showed NO statistically significant gain over Arm A. | **`DECISIVE COMPARATIVE GATE`** (Justifies Quarantine) | Local Archive (`exp28_fair_baseline_vs_pointer_*`) |
| **Exp29** | Independent Replication of Exp28 Comparative Findings | Frozen independent test runner | 100% independent reproduction of Exp28; confirmed distractor permutation failure. | **`INDEPENDENTLY REPRODUCED`** | Local Archive (`exp29_independent_exp28_audit_*`) |
| **Exp30** | Human Review Closure & Master Release Gate Matrix | Pre-publication audit of 12 release gates | All 12 release gates verified; canonical checkpoint SHA-256 validated bit-exact for v0.8.0. | **`RELEASE GATE PASS`** (Authorized v0.8.0) | Local Archive (`exp30_human_review_closure_*`) |
| **Exp31** | Capability Generalization & Out-of-Vocabulary Audit | Canonical Zero-Shot vs SFT vs Pointer Head | Canonical LM fails held-out tokens (0.0%); pointer copies held-out tokens (20.0%, p < 0.001); 100% fail on abstention. | **`VALIDATED CAPABILITY / NARROW GENERALIZATION`** (Quarantined) | Versioned Git (`experiments/exp31_capability_generalization/`); Local Weights (`checkpoints/`) |
| **Exp32** | Pointer Retrieval, Distractor Resolution & Abstention | Balanced Mixture (Present + Absent Keys) | Solved 0% abstention trap (F1: 0.3514, false abstention 3.67%); revealed distractor selection as 99% of retrieval errors. | **`DIAGNOSTIC SUCCESS / ABSTENTION PROVED`** (Distractor Bottleneck) | Versioned Git (`experiments/exp32_pointer_abstention/`); Local Weights (`checkpoints/`) |
| **Exp33** | Relational Pointer Circuit & Key-to-Value Binding | Two-Stage Relational Cross-Attention Head | Slot permutation robustness improved (48.0%); distractor confusion persists (22.0% on 4-key distractors) due to 3.44M representation entanglement. | **`HYPOTHESIS DISPROOF / CAPABILITY GATE REJECT`** (Backbone Limit Identified) | Versioned Git (`experiments/exp33_relational_pointer/`); Local Weights (`checkpoints/`) |

---

## 3. Storage, Lineage & Reproduction Policy

- **Tier 1 (Versioned Evidence in Git)**: Formal reports, summary JSON manifests, reusable diagnostic harnesses, and documentation.
- **Tier 2 (Local Research Archive)**: Intermediate training checkpoints (`.pt`), raw JSONL generation traces, and full epoch logs preserved locally in `experiments/overnight_runs/`.
- **Tier 3 (Reproducible Scratch)**: Disposable packaging virtual environments (`test_clean_venv`) and temporary test directories.
- **Public Assets**: The canonical Step 10,000 checkpoint (`42,212,787` bytes, SHA-256 `96357705...`) is distributed exclusively via official GitHub Release Assets (`v0.8.0`).
