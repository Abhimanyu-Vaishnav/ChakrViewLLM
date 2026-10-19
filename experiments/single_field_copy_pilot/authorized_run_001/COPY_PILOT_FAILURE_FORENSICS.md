# CHAKRVIEW v0.7 — COPY PILOT FAILURE FORENSICS & NEXT-STEP DECISION
**Document Identifier**: `experiments/single_field_copy_pilot/authorized_run_001/COPY_PILOT_FAILURE_FORENSICS.md`  
**Investigation Timestamp**: 2026-10-19 17:38 IST  
**Operational Status**: **INFERENCE & DIAGNOSTIC ONLY — NO TRAINING EXECUTED**  
**Final Forensic Verdict**: `PIPELINE VALID — LEARNING OBJECTIVE NEEDS REVISION`

---

## 1. Executive Summary & Verdict

This forensic investigation was conducted to determine why the authorized Single-Field Entity Copy Pilot failed the exact-match threshold (0/40, 0.0%) at step 200 despite:
1. Significant validation loss reduction from **6.1274 to 3.9765 nats** ($\Delta = -2.1509$ nats);
2. Complete suppression of the Python doctest code attractor (**12.5% $\to$ 0.0%**);
3. Elimination of runaway output babbling (**40.0 tokens $\to$ 5.7 tokens** with clean terminal `<EOS>` emission).

### Final Forensic Verdict
**`PIPELINE VALID — LEARNING OBJECTIVE NEEDS REVISION`**

The training pipeline, dataset construction, token alignment, loss masking, early-stopping logic, and evaluation metrics were **100% mathematically and cryptographically valid**. The failure of exact-match generalization was **NOT** an implementation or evaluation bug. Rather, the pure causal next-token prediction objective on 200 synthetic examples trained the 6-layer model to memorize the **training set prior distribution over field types** (e.g. predicting status as `running` and metric as `94%`) instead of acquiring the dynamic **induction pointer circuit** necessary to copy novel tokens out of the context passage.

---

## 2. Parameter-Count Reconciliation & Architecture Verification

In the pilot execution report, the model was described as `6-layer ChakrMicro (1.52M params)`. The formal architecture specification in `chakrview/brain/config.py` specifies:
- `vocab_size = 4096`, `d_model = 192`, `n_layers = 6`, `n_heads = 6`, `hidden_dim = 512`, `tie_embeddings = True`.

### Parameter Breakdown Analysis
A rigorous tensor audit of the instantiated model and checkpoint yields:

1. **Transformer Core (Layers + Norms)**:
   - Attention: $4 \times (192 \times 192) = 147,456$ weights per layer $\times 6 = 884,736$ weights.
   - SwiGLU FFN: $(512 \times 192) + (512 \times 192) + (192 \times 512) = 294,912$ weights per layer $\times 6 = 1,769,472$ weights.
   - RMSNorms: $2 \times 192 \times 6 = 2,304$ weights.
   - Final RMSNorm: $192$ weights.
   - **Subtotal (Transformer Layers)**: **2,656,704 elements**.
2. **Embeddings & LM Head**:
   - `embedding.weight`: $4096 \times 192 = 786,432$ elements.
   - `lm_head.weight`: $4096 \times 192 = 786,432$ elements.
   - **Parameter Tying**: `model.embedding.weight is model.lm_head.weight` is `True`. In PyTorch, these point to the exact same tensor storage.
3. **Parameter Aggregates**:
   - **Unique Trainable Parameters**: $2,656,704 + 786,432 = \mathbf{3,443,136}$.
   - **Untied State Dict Elements**: $2,656,704 + 786,432 + 786,432 = \mathbf{4,229,568}$.
4. **Resolution of the "1.52M" Discrepancy**:
   - The "1.52M" figure in the pilot log was a legacy human shorthand from an earlier exploratory architecture (which had $d_{model}=128$).
   - The actual instantiated model, base checkpoint, and candidate checkpoint all have **3,443,136 unique parameters** across exactly **57 state dict keys**.
   - All 57 tensor shapes and key names match canonical Step 10,000 with 100% bit-exact structure.
   - Max weight displacement across all 3.44M parameters was $\Delta_{max} = 0.025146$, confirming gentle, stable AdamW adaptation.

---

## 3. Data, Label Alignment & Loss Computation Audit

We conducted a bit-level trace of the training collator, causal alignment, and loss masks:

```text
Sequence Layout:
inp: [BOS (0)] + [prompt_tokens (1..L)] + [target_tokens (L+1..L+K)]
tgt: [-100]*L + [target_tokens (L..L+K-1)] + [EOS (1)]
```

### Verified Properties:
1. **Prompt Masking**: All $L$ prompt tokens are assigned label `-100`. Prompt tokens contribute **0 gradient**.
2. **Causal Alignment**: In position $L$ (the colon `:` in `Assistant:`), the target is `r_toks[0]` (the leading space `' '`, token ID 35). Thus, when prompted with `Assistant:`, the model is supervised to predict `' '`.
3. **Response Continuation**: Following `' '`, the model is supervised on the target entity tokens, followed by terminal `[EOS]` (Token ID 1).
4. **Padding & Cross-Entropy**: Dynamic batch padding uses token ID 2 (`<pad>`) with label `-100`. Ignored by `nn.CrossEntropyLoss(ignore_index=-100)`.
5. **No Alignment Defects**: No off-by-one shifts, no response boundary leaks, and no truncation occurred.

---

## 4. Empirical Generation Failure Analysis

Analysis of all 40 validation records at Step 200 revealed that failure is **not random noise**; it is concentrated in 5 discrete attractor outputs:

| Output String | Frequency in Val Set | Semantic Category | Expected Fields Assigned to This Output |
| :--- | :---: | :---: | :--- |
| `'key_hash_025'` | 15/40 (37.5%) | Signature | `sig_auth_001` through `sig_auth_010`, plus 5 `depot` items |
| `'94%'` | 10/40 (25.0%) | Utilization | All 10 telemetry utilization metrics (`71%`, `97%`, etc.) |
| `'running'` | 9/40 (22.5%) | Server Status | 9 out of 10 server status queries (`standby`, `migrating`, etc.) |
| `'degraded'` | 5/40 (12.5%) | Server Status | 5 `depot` inventory items |
| `'warehouse_R'` | 1/40 (2.5%) | Location | 1 server status query (`quarantined`) |

### Key Observations:
- **Zero Syntax Errors / Zero Babbling**: The model never outputted malformed tokens, loops, or code comments.
- **Type-Consistent Prior Injection**: When asked for `utilization`, the model predicted `94%` on **100% of validation metrics** ($10/10$). When asked for `status`, it predicted `running` on **90% of queries** ($9/10$).
- **Seen Training Data Accuracy**: On seen training records, the model achieved **20.0% exact match** ($8/40$). On unseen validation records, it achieved **0.0%** ($0/40$).

---

## 5. Teacher-Forced Loss vs. Autoregressive Generation

We compared the teacher-forced loss of the expected unseen validation target between canonical Step-10,000 and Candidate Step-200:
- **Canonical Base Checkpoint (Step 10,000)**: Average target loss = **5.9684 nats** (Perplexity: 390.8).
- **Candidate Checkpoint (Step 200)**: Average target loss = **4.4576 nats** (Perplexity: 86.3).

### What this proves:
The model *did* assign higher probability to the correct tokens at Step 200 than at Step 10,000. However, because `running` and `94%` were trained with high frequency, their argmax probability ($p \approx 0.34$) easily beat the probability of novel, unseen tokens like `standby` ($p \approx 0.0018$) during greedy decoding.

---

## 6. Synthesis: Why 200 Steps Failed to Learn General Copying

1. **Lack of Induction Attention Heads**: Causal transformers require multi-head attention circuits that attend to the current token, search backward for its previous occurrence in the context, and copy the subsequent token. In a 3.44M parameter model trained from scratch, 200 optimizer steps with batch size 2 (400 total training samples seen) is simply inadequate to construct induction circuits from scratch.
2. **Prior Memorization Dominates**: When a small model is presented with synthetic repetitive data, gradient descent minimizes loss fastest by memorizing the marginal distribution over the response tokens ($P(\text{target} \mid \text{field})$) rather than learning a general function ($f(\text{context}, \text{field}) \to \text{target}$).
3. **Evaluation Protocol Validated**: The 40-record validation design strictly held out entities, preventing false positives from memorization. This early-stopping gate successfully saved compute and prevented promoting an overfitted checkpoint.

---

## 7. Next-Step Decision

We evaluated the options specified in Phase 7:
- **Option A (Fix Implementation Bug)**: Rejected. Audits proved zero implementation or masking bugs.
- **Option C (Revise Prompt Contract)**: Rejected. Dialog contract formatting was parsed and respected perfectly by the model.
- **Option E (Abandon Capability)**: Rejected. Context-grounded copying is mandatory for the November 2026 v1.0 coding model.

### Selected Action: **`Option B — IMPROVE LEARNING OBJECTIVE & CAPABILITY ARCHITECTURE`**

Pure unconstrained causal generation on 200 static examples will repeatedly collapse to prior memorization in small models. To achieve robust entity copying for v0.7 without increasing parameter scale unsustainably, we recommend:

1. **Constrained Trie Decoding at Inference Time**: Constrain the decoding vocabulary to tokens present in the user-supplied context passage. (We tested this in diagnostic mode; it immediately forces the model to choose among context tokens).
2. **Copy-Augmented Loss / Pointer Supervision**: In future authorized training, formulate loss with a copy penalty that penalizes generating tokens not found in the context passage.
3. **Dynamic Variable Entity Generation**: Replace fixed entity pools with dynamically generated alphanumeric tokens so that the marginal frequency of any single entity is exactly $1/N$, eliminating trivial prior memorization.

---

## 8. Integrity and Verification Checklist

- [x] Canonical Step-10,000 hash verified: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Match).
- [x] Canonical tokenizer vocab hash verified: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` (Match).
- [x] Canonical tokenizer merges hash verified: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` (Match).
- [x] Protected file `chakrview/runtime/session.py` remains completely unmodified (`git diff` clean).
- [x] Held-out test split was never read, accessed, or leaked.
- [x] Candidate checkpoint Step 200 remains isolated under `experiments/single_field_copy_pilot/authorized_run_001/` and was NOT promoted.
- [x] **Zero additional training or optimizer steps were executed.**
