# Step 147: Neural Reasoning Curriculum & Pattern Generalization

## 1. Overview
Step 147 introduces a controlled synthetic reasoning curriculum directly into the neural learning pipeline to determine whether ChakrMicro learns underlying patterns rather than memorizing exact surface strings.

## 2. Key Architecture Components

- `NeuralReasoningCurriculumTrainer`:
  - Synthesizes transitive relational tasks: `fact: A > B and B > C . therefore A > C`.
  - Train and held-out splits use disjoint symbol vocabularies (`alpha...zeta` vs. `phi...theta`) to prevent surface token memorization.
  - Measures structural pattern acquisition on unseen composition problems.
- `NeuralReasoningResult`: Tracks baseline vs candidate training and held-out reasoning accuracy.

## 3. Empirical Verification
- Verified generation of structurally diverse transitive pairs.
- Demonstrated gradient training loop on relational structures with frozen baseline preservation.
