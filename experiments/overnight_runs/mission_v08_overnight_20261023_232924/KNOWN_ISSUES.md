# Known Issues & Limitations — ChakrView v0.8 Candidate

## 1. Short Explanation Automatic Evaluation Limitation
- **Status**: By Design / Held Out.
- **Details**: 190 short-explanation items in the Exp18 candidate bank represent natural-language scientific explanations. Automated lexical or exact-match evaluation of short explanations introduces unacceptable false-positive and false-negative rates.
- **Mitigation**: These items are systematically flagged as `NEEDS_REVIEW` and excluded from automated scoring in `TaskEvaluator`. Human expert domain review is required before these items can be promoted to verified training data.

## 2. Low-Resource SFT Step Horizon & Generalization
- **Status**: Empirical Limitation.
- **Details**: While 120 steps of SFT significantly lowered validation cross-entropy loss (4.6159 -> 3.3562) and improved EOS stopping behavior (0.0% -> 8.75%), task-level factual and arithmetic accuracy on held-out test prompts remains low (1.43%).
- **Mitigation**: The canonical pretrained checkpoint `step_10000.pt` must remain the default checkpoint. Do not promote experimental SFT checkpoints without extended curriculum training and multi-step reasoning adapters.

## 3. Shared Refusal Templates in Abstention Tasks
- **Status**: Audit Finding.
- **Details**: In the uncertainty/abstention category, unanswerable queries appropriately produce standardized refusal phrases (e.g., `The flight log does not state the total fuel consumption.`). 15 distinct refusal templates appear across splits as intended standard refusals.
- **Mitigation**: Exact prompt texts are 100% disjoint across splits (0 prompt leaks). The shared response strings are verified standard refusal phrases rather than entity leakage.

## 4. Protected Runtime Compatibility
- **Status**: Enforced.
- **Details**: `chakrview/runtime/session.py` and `chakrview/brain/model.py` are strictly protected and remain unchanged bit-for-bit.
