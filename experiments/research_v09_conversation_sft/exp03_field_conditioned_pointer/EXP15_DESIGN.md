# ChakrView Experiment 15: Field-Conditioned Pointer Attention
**Directory**: `experiments/research_v09_conversation_sft/exp03_field_conditioned_pointer/`  
**Date**: October 22, 2026  
**Auditor / Researcher**: Antigravity AI Engine

---

## 1. Executive Objective & Research Hypothesis

In Experiments 11–14, research demonstrated:
1. Pure causal language modeling (ChakrMicro 3.44M) fails to bind dynamic entities on held-out test scenarios, memorizing surface training token strings (Exp13 & Exp14).
2. Unconditioned span pointers (Exp10 & Exp11) collapse into slot entanglement when multiple fields or dynamic context roles are queried, because the query state derived from the prefix hidden state is unconditioned on the requested field slot.

**The Field-Conditioned Pointer Hypothesis**:
> By decoupling the requested semantic slot from the general causal prefix and projecting an explicit **Field-Query Embedding** (`[SLOT_ENTITY]`, `[SLOT_VALUE]`, `[SLOT_TARGET]`, `[SLOT_ABSTAIN]`) into the query projections of the span pointer, the model can selectively extract dynamic context spans corresponding to the specific requested field without vocabulary hallucination or slot entanglement, while keeping the 3.44M causal LM backbone completely frozen.

---

## 2. Parameter Budget & Provenance Invariants

- **Frozen Native Backbone**:
  - Model: ChakrMicro (6 layers, $d_{\text{model}}=192$, 6 heads, $d_{\text{ff}}=512$, Pre-RMSNorm, RoPE, SwiGLU).
  - Weights loaded strictly from canonical `training_checkpoint_step_10000.pt` (SHA-256: `96357705...`).
  - **100% Frozen**: All $3,443,136$ parameters have `requires_grad = False`.
- **Trainable Pointer Head**:
  - `field_embedding`: Embedding table of size $[N_{\text{fields}}, 192]$ ($N_{\text{fields}}=10$).
  - `query_start`, `key_start`: Linear projections $[192, 192]$ (bias-free).
  - `query_end`, `key_end`: Linear projections $[192, 192]$ (bias-free).
  - Initialized from Exp10 pointer weights where applicable.
  - Total trainable parameters: $\approx 149,376$ parameters ($<4.3\%$ of model capacity).
- **Protected Files**:
  - `chakrview/runtime/session.py` is NEVER touched.
  - Canonical checkpoint and tokenizer files are NEVER overwritten.

---

## 3. Mathematical Architecture

Let the tokenized sequence be $X = (x_1, \dots, x_T)$ and the field slot index be $f \in \{0, \dots, N_{\text{fields}}-1\}$.

1. **Backbone Context Encoding**:
   $$H = \text{ChakrMicro}(X) \in \mathbb{R}^{T \times 192}$$
2. **Field-Conditioned Query Formulation**:
   $$E_f = \text{field\_embedding}(f) \in \mathbb{R}^{192}$$
   $$Q_{\text{start}} = E_f W_{q,\text{start}}, \quad Q_{\text{end}} = E_f W_{q,\text{end}}$$
3. **Key Formulations over Context**:
   $$K_{\text{start}} = H W_{k,\text{start}}, \quad K_{\text{end}} = H W_{k,\text{end}}$$
4. **Pointer Score Distributions**:
   $$S_{\text{start}}(i) = \frac{Q_{\text{start}} K_{\text{start}}(i)^T}{\sqrt{d_{\text{model}}}}, \quad S_{\text{end}}(j) = \frac{Q_{\text{end}} K_{\text{end}}(j)^T}{\sqrt{d_{\text{model}}}}$$
5. **Eligible Context Masking**:
   - Positions outside context turns (e.g. padding tokens, system tokens) are masked with $-\infty$.
   - Position `0` (`[BOS]`) is designated as the explicit **ABSTAIN / NO-ANSWER** position for missing-context scenarios.
6. **Joint Constrained Decoding**:
   $$(s^*, e^*) = \arg\max_{s \le e \le s + M} \left( S_{\text{start}}(s) + S_{\text{end}}(e) \right)$$
   Where $M = 12$ is the maximum span length. If $s^* = 0$, the model predicts `ABSTAIN / MISSING`.

---

## 4. Anti-Leakage & Grounding Label Protocol

- **Label Derivation**:
  - Labels $[s, e]$ are derived by finding the exact target substring within the context tokens of the prompt.
  - If the target is absent (e.g. in `missing_context_abstention` or `factual_uncertainty`), the label is strictly set to $[0, 0]$ (`ABSTAIN`).
- **Zero Evaluation Leakage**:
  - The model and evaluator receive **only** the input sequence $X$, the field slot ID $f$, and the eligible candidate mask.
  - The gold start/end labels are strictly hidden during inference.
- **Split Separation**:
  - Train (100 examples), Val (30 examples), and Held-Out Test (60 examples) from Exp14 are used with verified zero prompt overlap and zero entity leakage.

---

## 5. Comparative Arms & Pre-Registered Acceptance Criteria

We compare four distinct arms on identical held-out test examples:
1. **Arm A: Frozen Canonical Causal LM (Floor)**: Generates free-form text without pointer (Exp14 baseline: 0.0%).
2. **Arm B: Deterministic Extraction Reference**: Non-neural heuristic reference (Exp14: 100.0%).
3. **Arm C: Query-Agnostic Pointer Ablation**: Pointer head unconditioned on field identity (simulates Exp10/11 behavior).
4. **Arm D: Field-Conditioned Pointer Prototype**: The full proposed prototype.

### Pre-Registered Acceptance Criteria:
- **Criterion 1 (Primary)**: Field-conditioned pointer (Arm D) achieves $> 60.0\%$ Exact Span Accuracy on held-out Test (vs. 0.0% in Arm A).
- **Criterion 2 (Ablation Gain)**: Arm D improves significantly over Arm C, proving field conditioning is necessary to avoid slot ambiguity.
- **Criterion 3 (Abstention Handling)**: Arm D correctly selects position `0` (`ABSTAIN`) on $\ge 80.0\%$ of missing-context cases.
- **Criterion 4 (Artifact Preservation)**: Canonical checkpoint and tokenizer vocabulary remain bit-identical before and after training.
