# Batch 108 — Operand Copy Benchmark Design and Baseline

**Sprint**: ChakrView Master Sprint Batches 107–110  
**Module**: Minimal Operand Copy Benchmark Design  
**Author**: ML Research Engineer  
**Date**: 2026-10-12  

---

## 1. Motivation and Benchmark Philosophy

In Batches 95–106, the failure to generate dynamic arguments was confounded by several factors simultaneously:
1. Long JSON syntax formatting overhead (`TOOL_CALL: {"tool": "calculator", "op": "add", "a": X, "b": Y}`).
2. Multi-token digit ambiguities (digit 9 encoding as 2 tokens in some contexts).
3. Arithmetic prompt distractors and multi-turn conversational templates.

To isolate whether the transformer architecture itself can perform input-to-output variable transport, we designed a minimal, deterministic **Operand Copy Benchmark** that eliminates arithmetic and tool formatting.

---

## 2. Dataset Design & Specification

- **Task**: Pure input-to-output copying of two operands $a, b \in \{0, 1, \dots, 9\}$.
- **Prompt Format**:
  ```text
  <|im_start|>user
  Copy:
  A={a}
  B={b}<|im_end|>
  <|im_start|>assistant
  ```
- **Target Format**:
  ```text
  A={a}
  B={b}<|im_end|>
  ```

### Structural & Tokenization Invariants
- Under the canonical BPE tokenizer (`vocab_4096`):
  - Target format `A={a}\nB={b}<|im_end|>\n` is **strictly 16 tokens** for every digit pair $(a, b) \in [0, 9]^2$.
  - Target tokens for digits are strictly:
    - $0 \to 51$, $1 \to 52$, $2 \to 53$, $3 \to 54$, $4 \to 55$
    - $5 \to 56$, $6 \to 57$, $7 \to 58$, $8 \to 59$, $9 \to 60$
  - Digit token $a$ is strictly at token offset 2 in the completion.
  - Digit token $b$ is strictly at token offset 6 in the completion.
  - Prompt format is **strictly 43 tokens** for all pairs, with $a$ at token index 17 and $b$ at token index 21.

### Split Partitions & Zero Leakage Verification
- **Total Pairs**: 100 possible ordered pairs $(a, b) \in [0, 9]^2$.
- **Held-Out Test Set** ($N = 20$):
  - Repeated digits: $(0,0), (4,4), (9,9)$.
  - Symmetric reverse pairs: $(1,8), (8,1), (2,7), (7,2), (3,6), (6,3), (0,5), (5,0), (1,4), (4,1), (2,9), (9,2), (3,5), (5,3), (6,8), (8,6)$.
  - Asymmetric control: $(7,9)$.
  - **Digit coverage**: Both Operand A and Operand B cover all 10 digits $\{0..9\}$ in the test set.
- **Validation Set** ($N = 16$): Random sample of remaining pairs.
- **Training Set** ($N = 64$): All remaining 64 pairs.
- **Split Leakage**: $\text{Train} \cap \text{Test} = \emptyset$, $\text{Train} \cap \text{Val} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$ (strictly 0% leakage verified).

---

## 3. Canonical Checkpoint Baseline Performance

Before any fine-tuning on the copy task, the canonical step-10,000 checkpoint (`training_checkpoint_step_10000.pt`, SHA-256 `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`) was evaluated on the benchmark:

| Split | $N$ | Greedy Op A Acc | Greedy Op B Acc | Exact Pair Acc | Output Collapse | Mean P(A) [Rank] | Mean P(B) [Rank] |
|---|---|---|---|---|---|---|---|
| **Train** | 64 | **0.0%** | **0.0%** | **0.0%** | 100.0% | 0.71% [Rank 15.9] | 4.69% [Rank 7.0] |
| **Val** | 16 | **0.0%** | **0.0%** | **0.0%** | 100.0% | 0.65% [Rank 16.8] | 3.64% [Rank 8.7] |
| **Test** | 20 | **0.0%** | **0.0%** | **0.0%** | 95.0% | 0.69% [Rank 15.9] | 4.43% [Rank 7.2] |

- **Modal Prediction**: Pretraining Wikipedia markup (`|-\n|200\n|2000 || np.nd|| style`).
- **Baseline Takeaway**: The canonical base model has never seen this instruction format; its zero-shot copy accuracy is strictly 0.0%, providing a completely clean zero-point baseline.
