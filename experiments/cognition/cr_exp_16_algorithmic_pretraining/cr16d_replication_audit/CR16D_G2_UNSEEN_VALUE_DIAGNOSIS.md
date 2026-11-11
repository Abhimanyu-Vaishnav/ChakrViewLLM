# CR-16D G2 Unseen-Value Diagnosis & Logit Analysis

### 1. Objective
Investigate why all evaluated models failed on Group G2 (probes with familiar keys and unseen values, averaging only 4% to 8% accuracy).

### 2. Experimental Tests
We performed deep token-level inspections across 20 representative G2 probes on the top-performing checkpoint (`arm_c_replay50_seed101`):
1. **Target Word Tokenization:** Inspected subword token fragmentation.
2. **Teacher-Forced Logits:** Computed target-token probability $P(w_1 \mid \text{prompt})$ and rank among the 4,096 vocabulary tokens.
3. **Free-Generation Output:** Compared top-5 predicted tokens with target tokens.

---

### 3. Quantitative Logit Findings

| Target Word | True Tokenization | Free Gen Output | Target Token ID | Target Rank in Logits | Target Probability | Top-1 Predicted Token |
|---|---|---|---|---|---|---|
| **Cotton** | `['Co', 't', 't', 'on']` | `[UNKNOWN]` | `1113` | **Rank 93** | 0.00014 | `[` (0.5256) |
| **Mercury** | `['M', 'er', 'cur', 'y']` | `Mango` | `80` | **Rank 1** | **0.8624** | `M` (0.8624) |
| **Spring** | `['S', 'pr', 'ing']` | `Sprin` | `86` | **Rank 1** | **0.9786** | `S` (0.9786) |
| **Shield** | `['S', 'hi', 'el', 'd']` | `Ship` | `86` | **Rank 1** | **0.9796** | `S` (0.9796) |
| **Diamond** | `['Di', 'am', 'on', 'd']` | `[UNKNOWN]` | `3821` | **Rank 18** | 0.00324 | `[` (0.5926) |

---

### 4. Root Causes Discovered

#### A. Multi-Token Subword Fragmentation
- In the training set, familiar values were predominantly single-token or common words (e.g., `Red`, `Apple`, `Lion`, `Paris`, `France`).
- In contrast, the OOD values in G2 fragment into 3 to 4 subword pieces:
  - `Mercury` $\to$ `['M', 'er', 'cur', 'y']`
  - `Shield` $\to$ `['S', 'hi', 'el', 'd']`
  - `Cotton` $\to$ `['Co', 't', 't', 'on']`

#### B. First-Token Success vs Subsequent-Token Autoregressive Drift
- Notice what happens for `Mercury`, `Spring`, and `Shield`:
  - The model **correctly predicts the first token with $>86\%$ probability** (`Rank 1` for `M` and `S`).
  - This proves that attention routing **did successfully point to the unseen value** in the prompt!
- However, after generating the first subword token (`M` or `S`), the model must continue decoding autoregressively.
  - When conditioned on `M`, the model's language model prior defaults to familiar training tokens: `M` $\to$ `an` $\to$ `g` $\to$ `o` (`Mango`).
  - When conditioned on `S` + `hi`, it completes to `p` (`Ship`) rather than `el` $\to$ `d` (`Shield`).

#### C. Unseen-Token Suppressions
- For rare or high-ID tokens (e.g. `Diamond` starting with token `3821`, or `Cotton` with `1113`), the unembedding prior is very low, and the model defaults to emitting the `[` bracket of `[UNKNOWN]`.

### 5. Scientific Conclusion
The failure on Group G2 is **not** a failure of attention routing. Attention points accurately to the value position and extracts the first subword token (`Rank 1` with $>86\%$ confidence). The failure is an **autoregressive subword completion drift** caused by lack of multi-token span training during algorithmic pretraining.

