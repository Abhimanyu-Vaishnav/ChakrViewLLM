# Batch 83 — Bounded Scratchpad Benchmark and Curriculum Specification

## 1. Hypothesis and Audit of Prior Work

In Batches 79–82, balanced arithmetic curriculum pretraining on direct addition equations broke global modal collapse, but revealed that ChakrMicro's causal attention routing conditioned almost exclusively on the **first operand $a$**, ignoring $b$.

### The Scratchpad Hypothesis:
Does explicitly training the model on intermediate counting sequences (`Start at a; count b more: ...; answer: c`) provide the transformer layers with the computational depth and autoregressive sequence positions needed to route information from both operands before generating the final answer?

---

## 2. Benchmark Split Provenance and Disjointness

Sprint Batch 83 reuses the exact 100-pair benchmark splits from Batch 79:
- **Train**: 35 unordered pairs $\to$ 64 ordered pairs (covers all 19 sum classes 0..18).
- **Validation**: 10 unordered pairs $\to$ 18 ordered pairs.
- **Held-Out Test**: 10 unordered pairs $\to$ 18 ordered pairs.
- **Disjointness Invariant**: Zero unordered, ordered, or reversed-pair overlap between training, validation, and held-out test splits.

### Prompt Formats:
1. **Direct Arithmetic**:
   ```
   <|im_start|>user\nAdd: {a} + {b} = ?<|im_end|>\n<|im_start|>assistant\n{c}<|im_end|>\n
   ```
2. **Scratchpad Arithmetic**:
   ```
   <|im_start|>user\nAdd with scratchpad: {a} + {b} = ?<|im_end|>\n<|im_start|>assistant\nStart at {a}; count {b} more: {counting_str}; answer: {c}<|im_end|>\n
   ```

---

## 3. Matched Training Token Budgets

From the 64 training pairs, two matched token streams were generated:
- **Condition A (Direct Training Tokens)**: Exactly **99,417 tokens** (SHA-256: `0dc9143984af47d6e0a13d478963bf4bcc35afd81dd5777d26b3b557a070542b`).
- **Condition B (Scratchpad Training Tokens)**: Exactly **99,417 tokens** (SHA-256: `aacc118bd25ea1caf61ba2f66a63056d07bc7f9901a567ff1a8ac74c9de13fb9`).
- Both conditions trained for 100 optimizer steps with batch size 4 and sequence length 64 ($100 \times 4 \times 64 = 25,600$ tokens) from the immutable canonical step-10,000 checkpoint (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
