# STEP 126: Neural / Cognitive Integration Boundary

## 1. Overview
Step 126 defines the precise boundary between ChakrMicro and the higher-level cognitive architecture:
$$\text{NEURAL BRAIN} \neq \text{KNOWLEDGE} \neq \text{MEMORY} \neq \text{TOOLS} \neq \text{COGNITION}$$

## 2. Architecture
- **`NeuralCognitiveBridge`**: Proxies representation queries and perplexity scoring.
- **Model Tiers**: `CANONICAL_BASELINE`, `CANDIDATE_CHECKPOINT`, `EVALUATION_CHECKPOINT`.
- **Governance**: Neural model has ZERO tool authority. Only `GovernedToolGate` authorizes actions.

## 3. Verification
Verified in `test_step126_neural_cognitive_boundary`. Canonical baseline SHA-256 remains bit-exact (`c5571c...`).
