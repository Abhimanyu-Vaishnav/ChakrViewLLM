# ChakrView Exp19 Proposal — De-Biased Template Diversity & Extended SFT (`exp19_debiased_sft_cpu_600`)

**Proposal Status**: `READY_TO_PROPOSE` (Training will NOT start without explicit user authorization)  
**Experiment Identifier**: `exp19_debiased_sft_cpu_600`  
**Base Checkpoint**: Verified isolated copy of canonical `step_10000.pt`  
**Execution Budget**: 600 steps on CPU (~36 seconds compute time)  

---

## 1. Problem Statement & Empirical Justification

The Exp18 SFT diagnostic revealed:
1. **Proven Capability**:
   - The aligned prompt contract and causal loss-masking successfully enabled conversational turn-boundaries, increasing EOS termination rate from **0.62% to 45.0%** in just 300 steps.
2. **Identified Failure Mode (Prefix Monoculture Collapse)**:
   - **77.5% (124/160)** of generated responses collapsed into the single prefix template `"The difference..."` across factual QA, arithmetic, reference resolution, and abstention.
   - Analysis of the 1,200 training examples revealed that over 60% of response targets share rigid `"The <noun>..."` phrasing (e.g., `"The pronoun..."`, `"The difference..."`, `"The sum..."`), allowing the model to minimize training loss by memorizing a single opening pattern rather than attending to prompt semantic features.
3. **Identified Failure Mode (Under-Training / Step Budget)**:
   - At 300 steps with batch size 2, the model only sampled 600 examples, leaving half the 1,200-example dataset unseen.

---

## 2. Technical Experiment Design for Exp19

### A. Dataset Enhancement (In-Memory Only, No Official File Writes)
- **Diverse Response Openers**:
  - In `basic_arithmetic`: Introduce diverse opening syntaxes, e.g.:
    - `"The difference between X and Y is Z."`
    - `"X minus Y equals Z."`
    - `"Subtracting Y from X yields Z."`
    - `"Result: Z."`
  - In `factual_qa`: Vary direct answers vs sentence answers:
    - `"The capital of Ireland is Dublin."` $\leftrightarrow$ `"Dublin is Ireland's capital."`
  - In `reference_resolution`: Replace the repetitive `"The pronoun 'it' designates X"` with natural sentence references (`"'It' refers to X."`, `"In this log, 'it' denotes X."`).
- **Structured Formatting Prefix Anchoring**:
  - Prepend a strict JSON token anchor or schema directive in the system conditioning prompt to suppress the base model's doctest code-completion prior.

### B. Hyperparameter Configuration
- **Base Checkpoint**: Fresh isolated copy of canonical `training_checkpoint_step_10000.pt` (do not resume from corrupted weights).
- **Steps**: 600 optimizer steps (allows full 1-epoch coverage of the 1,200 training records at batch size 2).
- **Learning Rate**: $5 \times 10^{-5} \to 5 \times 10^{-6}$ (Cosine decay).
- **Batch Size**: 2.
- **Gradient Clipping**: 1.0.
- **Validation Interval**: 25 steps (saving best checkpoint based strictly on held-out validation loss).

---

## 3. Quantitative Acceptance Criteria (Gates)

To pass Exp19 and qualify as a candidate release candidate:
- **GATE-01 (Task Accuracy)**: $\ge 5.0\%$ scorable accuracy across scorable test records (up from 0.71%).
- **GATE-02 (EOS Completion Rate)**: $\ge 50.0\%$ clean EOS termination rate (up from 45.0%).
- **GATE-03 (Prefix Diversity)**: Less than **25.0%** of test outputs starting with any single 2-word prefix (down from 77.5%).
- **GATE-04 (Zero Protected Regressions)**: Bitwise match on all 9 canonical protected assets and 100% pass on repository regression test suite (126/126).

---

## 4. Next Step Authorization

This proposal is fully prepared, isolated, and tested. **No training will be launched without user confirmation.**
