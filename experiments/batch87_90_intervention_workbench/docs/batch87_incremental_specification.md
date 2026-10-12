# Batch 87 — Incremental Supervision Intervention and Data Specification

## 1. Hypothesis and Motivation

Across Batches 71–86, direct-answer learning produced constant modal attractors (predicting `6`, `7`, or `8`), while full-sequence scratchpad learning produced canned template prototype memorization (`Start at 0... answer: 4` or `Start at 9... answer: 15`).

### The Incremental Next-Step Hypothesis:
Does decomposing arithmetic into isolated single-step transitions ($k \to k + 1$ with loss applied specifically to predicting the next state given $(a, b, \text{curr})$) force the model to condition on its inputs, or does the small causal architecture still default to an attractor?

---

## 2. Benchmark Split Provenance and Disjointness

Sprint Batch 87 reuses the established 100-pair addition domain from Batches 79–86:
- **Train**: 35 unordered pairs $\to$ 64 ordered pairs (covers all 19 sum classes 0..18).
- **Validation**: 10 unordered pairs $\to$ 18 ordered pairs.
- **Held-Out Test**: 10 unordered pairs $\to$ 18 ordered pairs.
- **Disjointness Invariant**: Zero unordered, ordered, or reversed-pair overlap between training, validation, and held-out test splits.

### Transition Format:
For each pair $(a, b)$ with sum $c = a + b$:
```
Prompt: <|im_start|>user\nAdd step {k}: {a} + {b} from {curr}<|im_end|>\n<|im_start|>assistant\n
Response: {curr + 1}{" final" if k == b else ""}<|im_end|>\n
```

---

## 3. Matched Training Token Budgets

From the 64 training pairs, matched token streams of **124,289 tokens** were generated:
- **Condition A (Direct Training Tokens)**: 124,289 tokens (SHA-256: `139704d69a8c2ce4abb5ecb74513766a1316252d3f4a50b48f613bc7d64918d8`).
- **Condition B (Incremental Training Tokens)**: 124,289 tokens (SHA-256: `4ea85fcd4539139b3a62b02b6742eb0ad76527a862a9d1715937cda89d55aa12`).
- Both conditions trained for 100 optimizer steps with batch size 4 and sequence length 64 ($100 \times 4 \times 64 = 25,600$ tokens) from the canonical step-10,000 checkpoint (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
