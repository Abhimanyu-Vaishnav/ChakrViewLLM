# Field-Conditioned Pointer Attention: Research Design Specification
**Experiment Number**: Exp15 (Future Isolated Phase)  
**Document Date**: October 22, 2026  
**Auditor / Architect**: Antigravity AI Engine

---

## 1. Problem Statement & Theoretical Foundation

In Experiments 10–14, empirical results established two core realities:
1. **The 3.44M Causal LM Backbone Cannot Flexibly Bind Dynamic Prompt Variables**: When fine-tuned on conversational dialogue, the causal language model memorizes surface token sequences (e.g. reproducing `is_positive_train_0` from training data rather than binding `is_positive_test_0` from prompt context).
2. **Unconditioned Span Pointers Suffer Slot Entanglement**: When multiple slots are requested in a structured prompt (e.g. `entity`, `status`, `code`), the unconditioned pointer repeatedly extracts the first salient entity into every slot because the query representation lacks target slot conditioning.

**The Context-Grounding Hypothesis**:
> Introducing explicit field-conditioned query embeddings projected into the span-pointer attention head allows the model to selectively bind dynamic source entities from the context into specific requested conversational slots, without requiring open-vocabulary generative hallucination or model weight expansion.

---

## 2. Mathematical Architecture

```
                  [Prompt / Multi-Turn Context Tokens]
                                   |
                     +---------------------------+
                     | Frozen ChakrMicro 3.44M   |
                     | 6 Layers, d_model = 192   |
                     +---------------------------+
                                   |
                  Context Hidden States H in R^{L x 192}
                                   |
        +--------------------------+--------------------------+
        |                                                     |
        v                                                     v
[Slot Identity Token ID]                               [Key Projections]
        |                                           K_start = H * W_k_start
[Slot Embedding Table]                              K_end   = H * W_k_end
E_slot in R^{192}                                             |
        |                                                     |
[Query Projections]                                           |
Q_start = E_slot * W_q_start                                  |
Q_end   = E_slot * W_q_end                                    |
        |                                                     |
        +--------------------------+--------------------------+
                                   |
               Scaled Dot-Product Pointer Attention:
         Attn_start = softmax( (Q_start * K_start^T) / sqrt(d) )
         Attn_end   = softmax( (Q_end   * K_end^T)   / sqrt(d) )
                                   |
                     Predicted Span: [start_idx, end_idx]
```

### Parameter Allocation & Provenance:
- **Frozen Backbone**: The entire 6-layer ChakrMicro transformer ($3,443,136$ parameters) from the canonical Step-10000 checkpoint remains **100% frozen** ($\Delta W \equiv 0$).
- **Trainable Parameters**:
  - `slot_embeddings`: Embedding table of size $[N_{\text{slots}}, 192]$ (e.g. $N_{\text{slots}}=8$: `[SLOT_ENTITY]`, `[SLOT_VALUE]`, `[SLOT_STATUS]`, `[SLOT_FORMAT]`, `[SLOT_ACTION]`, etc.).
  - `W_q_start`, `W_k_start`, `W_q_end`, `W_k_end`: Linear projections of size $[192, 192]$.
  - Total trainable parameters: $\approx 149,000$ parameters ($<4.3\%$ of model capacity).

---

## 3. Grounding Labels & Anti-Leakage Protocol

1. **Source Span Label Generation**:
   - The dataset compiler searches for the exact target substring within the prompt context tokens.
   - It records the inclusive token boundaries: $[s_{\text{start}}, s_{\text{end}}]$.
   - If the target entity is absent (e.g. in `missing_context_abstention` or `factual_uncertainty`), the label is explicitly set to index `0` (the `[BOS]` or dedicated `[NONE]` token).
2. **Zero Evaluation Mask Leakage**:
   - As audited in Exp10.3, the pointer loss during training is CrossEntropyLoss on candidate start and end logits.
   - At evaluation time, candidate eligibility masks are strictly derived from prompt structure (e.g., candidate tokens must lie within context turns, not user query turns). Ground-truth start/end positions must **never** be used to filter candidates.

---

## 4. Controlled Baselines & Ablation Ladder

In Exp15, evaluation will compare:
1. **Arm 1: Frozen Causal LM**: Autoregressive generation without pointer (Floor).
2. **Arm 2: Unconditioned Span Pointer**: Exp11/12 architecture (measures slot entanglement baseline).
3. **Arm 3: Field-Conditioned Span Pointer**: The proposed architecture.
4. **Arm 4: Oracle Span Selection**: Ceiling performance when ground truth position is provided.
5. **Arm 5: Deterministic Regex Extractor**: Non-neural rule baseline.

---

## 5. Acceptance Criteria (Pre-Registered)

Before initiating Exp15 training, the experiment must meet these pre-registered acceptance criteria:
1. **Multi-Slot Disambiguation**: On multi-slot extraction queries, field-conditioned pointer achieves $> 70.0\%$ whole-object exact match (vs. $3.3\%$ in Exp12).
2. **Zero Surface Memorization**: On held-out entity test splits, exact entity binding achieves $> 65.0\%$ (vs. $0.0\%$ in Exp14 native LM).
3. **Canonical Hash Preservation**: Canonical checkpoint and tokenizer vocabulary remain bit-identical before and after training.
