# Batch 99: Representation and Information-Flow Audit

**Sprint Date**: October 12, 2026  
**Repository Branch**: `dev/v0.6-v0.7-validation`  
**Starting HEAD**: `c2f191ae9261a1ea3ce1824d8d49daedc244354f`  
**Evaluation Model**: Canonical `ChakrMicro` (6 layers, $d_{model}=192$, $n_{heads}=6$, $hidden\_dim=512$, $vocab\_size=4096$)  
**Evaluated Checkpoints**:
- Canonical Step 10,000 (`training_checkpoint_step_10000.pt`, SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`)
- Condition B (Unstructured NL Tool Calling, SHA-256: `a78cbba985ef8e26db4f280a9775be351c31278f35cba4864ebf9494bf565b91`)
- Condition C (Explicit Slot-Binding SFT, SHA-256: `d7702580c5e586ed4ffa3a6498d4f35520886d92e8f3b581ab8df1c26d4cdaa2`)

---

## 1. Information-Flow Trace & Tokenizer Subword Behavior

### 1.1 Tokenizer Asymmetry on Operand Digits
A critical empirical discovery in this audit is that **the tokenizer does not treat single digits symmetrically**:
1. When tokenizing digits 0..9 following `": "` (e.g., `Operand A: {d}`):
   - For $d \in \{0, 2, 3, 4, 5, 6, 7, 8, 9\}$, the string `": "` encodes to token ID `324` (`': '`), and `{d}` encodes to single-digit token IDs ($51..60$).
   - For $d = 1$, the string `": 1"` is merged into a **single BPE subword token ID `2665`** (`': 1'`).
2. Consequently:
   - Sequence length and token offsets vary depending on whether Operand A or Operand B is `1`.
   - Any naive token-position lookup hardcoding fixed offsets (e.g. index 26 or index 32) fails whenever $a=1$ or $b=1$.
   - We implemented and verified `find_operand_token_positions()` which dynamically locates the token index based on the label tokens `Oper` (`2264`), `and ` (`354`), `A` (`68`), and `B` (`69`).

### 1.2 Tool Call Target Argument Encoding
In the target JSON payload `{"tool": "calculator", "op": "add", "a": {a}, "b": {b}}`:
- For Operand A: following `"a": ` (token `544`), digit $d$ is encoded as merged token `"{d}, "`:
  - `0, ` -> token `1290`
  - `1, ` -> token `628`
  - `2, ` -> token `892`
  - `3, ` -> token `660`
  - `4, ` -> token `1637`
  - `5, ` -> token `1298`
  - `6, ` -> token `1641`
  - `7, ` -> token `753`
  - `8, ` -> token `2225`
  - `9` -> token `60` followed by `, ` (`276`)
- For Operand B: following `"b": ` (token `544`), digit $d$ precedes the closing brace `}`:
  - `0}` -> token `1631`
  - `1}` -> token `585`
  - `2}` -> token `2202`
  - $3..9$ -> individual digit token ($54..60$) followed by `}` (`128`).
- In Condition B, the model collapsed to `1, ` (`628`) and `1}` (`585`).
- In Condition C, the model collapsed to `1, ` (`628`) and `8` (`59`).

---

## 2. Representation Probe Design & Split Isolation

To determine whether operand identities are recoverable from internal activations, we implemented a linear probe diagnostic:
- **Input Features**: Frozen representations of size 192 extracted from:
  - Token positions: `pos_a` (token representing Operand A), `pos_b` (token representing Operand B), and `prompt_end` (last token of the prompt prefix prior to generation).
  - Layers: Layer 0 (Embedding), Layer 1, Layer 3, Layer 5, Layer 7 (Post-Final RMSNorm).
- **Split Isolation**:
  - Training partition: 64 pairs from `train.jsonl` (35 unique unordered pairs).
  - Test partition: 18 pairs from `test.jsonl` (10 unique unordered pairs).
  - Pair / reverse-pair isolation: Zero unordered pair overlap between train and test. Neither $(a, b)$ nor $(b, a)$ from test appears in train.
- **Controls & Baselines**:
  - Majority-class baseline: Test accuracy of predicting the most frequent training class (11.1% on the balanced 10-class test set).
  - Permutation control: Labels randomly permuted on the training set with fixed random seeds.

---

## 3. Quantitative Probing Results (Condition C)

### 3.1 Probing Operand A
| Position | Layer | Train Acc | Test Acc | Majority Baseline | Shuffled Control Test |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`pos_a`** | Layer 0 (Emb) | 100.0% | **100.0%** | 11.1% | 38.9% |
| `pos_a` | Layer 3 | 100.0% | **100.0%** | 11.1% | 11.1% |
| `pos_a` | Layer 7 (Norm)| 100.0% | **100.0%** | 11.1% | 11.1% |
| **`pos_b`** | Layer 0 (Emb) | 15.6% | **0.0%** | 11.1% | 5.6% |
| `pos_b` | Layer 3 | 95.3% | **11.1%** | 11.1% | 11.1% |
| `pos_b` | Layer 7 (Norm)| 98.4% | **11.1%** | 11.1% | 0.0% |
| **`prompt_end`**| Layer 0 (Emb) | 12.5% | **11.1%** | 11.1% | 11.1% |
| `prompt_end` | Layer 1 | 31.2% | **22.2%** | 11.1% | 0.0% |
| `prompt_end` | Layer 3 | 50.0% | **33.3%** | 11.1% | 5.6% |
| `prompt_end` | Layer 5 | 53.1% | **44.4%** | 11.1% | 16.7% |
| `prompt_end` | Layer 7 (Norm)| 29.7% | **22.2%** | 11.1% | 11.1% |

### 3.2 Probing Operand B
| Position | Layer | Train Acc | Test Acc | Majority Baseline | Shuffled Control Test |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`pos_a`** | Layer 0 (Emb) | 15.6% | **0.0%** | 11.1% | 11.1% |
| `pos_a` | Layer 7 (Norm)| 15.6% | **0.0%** | 11.1% | 5.6% |
| **`pos_b`** | Layer 0 (Emb) | 100.0% | **100.0%** | 11.1% | 22.2% |
| `pos_b` | Layer 3 | 100.0% | **100.0%** | 11.1% | 0.0% |
| `pos_b` | Layer 7 (Norm)| 100.0% | **100.0%** | 11.1% | 22.2% |
| **`prompt_end`**| Layer 0 (Emb) | 12.5% | **11.1%** | 11.1% | 11.1% |
| `prompt_end` | Layer 1 | 23.4% | **22.2%** | 11.1% | 11.1% |
| `prompt_end` | Layer 3 | 59.4% | **50.0%** | 11.1% | 16.7% |
| `prompt_end` | Layer 5 | 60.9% | **55.6%** | 11.1% | 5.6% |
| `prompt_end` | Layer 7 (Norm)| 31.2% | **22.2%** | 11.1% | 27.8% |

---

## 4. Scientific Diagnosis & Interpretation

Distinguishing the competing hypotheses:
1. **Hypothesis 1 (Representation Loss)**: Is operand identity lost inside the network?
   - **Disproven**: At `pos_a` and `pos_b`, operand identity is decoded with 100.0% accuracy across held-out test examples at all layers (0 through 7). The model retains the local token identity perfectly in its residual stream.
2. **Hypothesis 2 (Routing Degradation at Sequence Tail)**:
   - At the `prompt_end` token position (the exact point where the model must begin generating), decodability drops from 100% to 22.2% - 55.6%.
   - Noticeable causal attention leakage appears on the training set (Layer 3 at `pos_b` reaches 95.3% train accuracy for Operand A, but only 11.1% on test), indicating memorized training set combinations rather than systematic forward routing.
3. **Hypothesis 3 (Readout / Generation Bias at Argument Slots)**:
   - When the teacher-forced prefix is extended all the way to `"a": `, the probe accuracy for Operand A recovers to **61.1% on held-out test data**.
   - However, during generation, the tied output projection assigns the highest probability to token `628` (`"1, "`) uniformly across all prompts (15.2% - 16.8%), regardless of whether Operand A was 0, 1, 4, 8, etc.
   - The correct operand token is consistently present among the top 10 candidates (probability 3.8% - 16.4%), but is overwhelmed by the static unigram prior acquired during training.

### Outcome Classification
The empirical findings establish **Outcome 2 with Routing Attenuation**:
Operand identity is fully recoverable at the operand positions ($100\%$) and partially routed to the argument generation position ($61\%$), but the generation policy collapses to a static argmax mode attractor ($100\%$ collapse to static arguments).
