# CR-EXP-17: v0.9.1 Release Report

## 1. Release Identification
- **Release Version:** v0.9.1
- **Release Codename:** Conversational Preview
- **Release Classification:** Experimental Research Preview
- **Target Platform:** Native CPU-first autonomous execution
- **Artifact Path:** `releases/v0.9.1_conversational_preview/`
- **Model Binary:** `releases/v0.9.1_conversational_preview/chakrview_v091_model.pt`
- **Model Checkpoint SHA-256:** `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10`

## 2. Release Objectives & Accomplishments
1. **Conversational Recovery:** Eradicated the refusal sink pathology (`"That information is not provided in the context."`) that compromised earlier SFT runs.
2. **Self-Identity Realization:** Grounded the model in its authentic persona as ChakrView, an independent first-principles indigenous neural model.
3. **Multilingual Grounding:** Added conversational alignment in Hindi and Hinglish.
4. **Interactive CLI Integration:** Delivered `python -m chakrview chat` with calibrated decoding (`repetition_penalty = 1.15`, `no_repeat_ngram_size = 3`).
5. **Frozen Benchmark Evaluation:** Evaluated across 200 non-overlapping probes in `CONVERSATION_BENCHMARK_V1.json` with zero training data leakage.
6. **Reproducibility Guarantee:** Maintained bit-exact architecture parity with the canonical ChakrMicro v0.1 specification ($3,443,136$ parameters).

## 3. Deployment Artifacts
- `chakrview_v091_model.pt`: Production release weights ($13.8\text{ MB}$).
- `config.json`: Complete hyperparameter and architectural specification.
- `MODEL_CARD.md`: Comprehensive capabilities, bounds, and ethical guidelines.
- `HOW_TO_RUN.md`: End-user CLI and programmatic inference instructions.
