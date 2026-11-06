# UPDATED CAPABILITY REGISTRY (POST CR-EXP-10)
**Last Updated:** 2026-11-07  
**Model Champion:** `cr_exp_10_arm_b_best_val.pt` (SHA-256: `8c15c2cb8b4753f24931fd0cf9a76d8b629d7738c9a4f1869ef65b509af0d488`)  

### 1. Conversational Understanding & Persona Preservation: **VALIDATED**
- **In-Distribution Score:** $77.33\%$ | **Strict OOD Score:** $79.00\%$ (119/150 in dedicated suite)
- **Status:** Fully validated. Consistently identities as ChakrView, engages politely, and refuses false biological embodiment.

### 2. Missing-Information Epistemic Abstention: **VALIDATED**
- **In-Distribution Score:** $46.67\%$ | **Strict OOD Score:** $100.00\%$ (100/100)
- **Status:** Fully validated. Accurately identifies when context lacks information regarding queried people or predicates.

### 3. Conditional Deductive Logic (Modus Ponens): **PARTIALLY VALIDATED**
- **In-Distribution Score:** $37.33\%$ | **Strict OOD Score:** $50.00\%$ (50/100)
- **Status:** Partially validated. Solves binary `If P then Q` logic problems at non-trivial baseline accuracy.

### 4. Elementary State Arithmetic: **EMERGING**
- **In-Distribution Score:** $44.00\%$ | **Strict OOD Score:** $11.00\%$
- **Status:** Emerging on familiar numeric ranges; weak zero-shot generalization on novel names.

### 5. Multi-Turn Working Memory & Entity State Tracking: **PARTIALLY VALIDATED (ID ONLY)**
- **In-Distribution Score:** $78.67\%$ | **Strict OOD Score:** $1.00\%$
- **Status:** Template-dependent. Highly reliable when familiar entity names are used ($78.67\%$), but fails on strictly novel OOD names due to limited embedding capacity in 3.44M parameters.

### 6. Multi-Entity Distractor Disambiguation: **NOT DEMONSTRATED ON OOD**
- **In-Distribution Score:** $42.67\%$ | **Strict OOD Score:** $3.00\%$
- **Status:** Primacy bias persists when contexts contain $\ge 3$ arbitrary novel entities.

### 7. Strict Negative Constraint Satisfaction: **NOT DEMONSTRATED**
- **In-Distribution Score:** $16.00\%$ | **Strict OOD Score:** $0.00\%$
- **Status:** Inhibitory token masking requires architectural attention head specialization or increased head count.
