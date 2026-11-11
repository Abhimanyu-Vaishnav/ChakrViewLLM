# CR-16E-R2 Phase E: Controlled Gradient-Interference Experiment Design

### 1. Hypothesis and Theoretical Rationale
- **Research Question:** Does supervised loss on negative abstention examples (`Context ... Query -> [UNKNOWN]`) causally induce gradient interference that impairs arbitrary sequence copying?
- **Hypothesis:** When the model computes cross-entropy gradients on negative abstention examples, the gradient update suppresses continuation probabilities and introduces negative interference on attention heads, destabilizing arbitrary string replication.
- **Experimental Test:** A tightly controlled two-condition study:
  - **Arm G1 (Negative-Target Loss Enabled):** Identical to standard CR-16E training. All tokens (retrieval, copying, and negative abstention `[UNKNOWN]`) contribute to cross-entropy loss.
  - **Arm G2 (Negative-Target Loss Masked):** Exactly identical input examples, batch composition, batch ordering, optimizer states, parent initialization, and forward passes. However, labels for negative abstention examples are masked to `-100`, eliminating their gradient contribution while leaving copying and retrieval loss strictly identical.

---

### 2. Controlled Experimental Controls & Invariants

| Experimental Factor | Arm G1 (Enabled) | Arm G2 (Masked) | Verification Status |
|---|---|---|---|
| **Parent Checkpoint** | CR-16A Arm D (`9d609a77...`) | CR-16A Arm D (`9d609a77...`) | Identical SHA-256 |
| **Model Architecture** | Canonical ChakrMicro v0.1 ($3.44\text{M}$) | Canonical ChakrMicro v0.1 ($3.44\text{M}$) | Frozen, identical |
| **Tokenizer** | Canonical `vocab_4096` | Canonical `vocab_4096` | Frozen, identical |
| **Training Dataset** | `arm_d_shuffled_balancedspan_train.json` | `arm_d_shuffled_balancedspan_train.json` | Identical 4,000 examples |
| **Seeds Tested** | 2026, 3030, 4040 | 2026, 3030, 4040 | Paired seed-for-seed |
| **Batch Size & Ordering** | 16 with deterministic seed | 16 with deterministic seed | Bit-for-bit identical batch stream |
| **Training Steps** | Exactly 400 steps | Exactly 400 steps | Identical duration |
| **Learning Rate & Optimizer** | AdamW, lr=2e-4, weight_decay=0.01 | AdamW, lr=2e-4, weight_decay=0.01 | Identical optimizer config |
| **Negative Example Target Loss** | **Enabled (normal cross-entropy)** | **MASKED (-100 on abstention tokens)** | **Isolated Independent Variable** |

---

### 3. Masking Implementation & Synthetic Unit Verification
In `FactorialDataset`:
```python
# For Arm G2:
if is_masked_arm and item["task"] == "abstention":
    # Mask all label tokens to -100 so PyTorch CrossEntropyLoss ignores them completely
    labels = [-100] * len(labels)
```
- **Validation Check:**
  - Copying examples: Labels unchanged ($[l_1, l_2, \dots]$ where $i \ge \text{prompt\_len}-1$).
  - Retrieval examples: Labels unchanged.
  - Abstention examples in G1: Target tokens contribute loss on `[UNKNOWN]\n`.
  - Abstention examples in G2: Target tokens masked to `-100`; effective loss denominator excludes abstention tokens.

