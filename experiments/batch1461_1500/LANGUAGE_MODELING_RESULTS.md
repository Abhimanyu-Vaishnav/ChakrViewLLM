# Track A: Next-Token Prediction and Language Modeling Results

## Batches 1461–1500: Audit of Neural Model-Native Predictive Performance

### 1. Track Objective & Setup

Track A audits the fundamental predictive power of the indigenous ChakrMicro neural network—measuring what the model learned purely as a self-supervised autoregressive sequence predictor, without external tools, search indices, regexes, or prompt scaffolds.

- **Canonical Baseline:** `Step 10,000` pre-trained checkpoint (`training_checkpoint_step_10000.pt`, SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Domain-Adapted Candidate:** `Step 10,080` checkpoint (`chakrmicro_domain_adapted_experimental.pt`, SHA-256: `15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139`).
- **Benchmark Evaluation Corpus:** Frozen independent test corpus (30 documents, 7,741 target tokens) constructed and verified in Batch 1421–1460, spanning 5 distinct domains (Distributed & Consensus, Python & Concurrency, Explanatory Prose, Numerical Analysis, OS Architecture).

---

### 2. Quantitative Performance Metrics

Aggregate cross-entropy and perplexity are computed from the total accumulated negative log-likelihood across all evaluated target tokens ($\text{PPL} = \exp(\frac{1}{N}\sum \mathcal{L}_i)$) under standard causal self-attention masking:

| Model Checkpoint | Evaluated Target Tokens | Aggregate Cross-Entropy (nats) | Aggregate Perplexity (PPL) | Top-1 Accuracy (%) | Top-5 Accuracy (%) | Top-10 Accuracy (%) |
|---|---|---|---|---|---|---|
| **Canonical Base (Step 10,000)** | 7,741 | 5.3852 | 218.15 | 11.02% | 27.45% | 35.95% |
| **Adapted Candidate (Step 10,080)** | 7,741 | **4.8497** | **127.70** | **17.41%** | **33.12%** | **40.81%** |
| **Absolute Delta ($\Delta$)** | — | **-0.5355** | **-90.45** | **+6.39%** | **+5.67%** | **+4.86%** |
| **Relative Improvement** | — | **-9.94%** | **-41.46%** | **+57.98%** | **+20.65%** | **+13.52%** |

#### Per-Domain Breakdown:
| Domain Category | Tokens | Baseline PPL | Candidate PPL | PPL $\Delta$ | Relative PPL Drop |
|---|---|---|---|---|---|
| Distributed & Consensus | 1,534 | 195.80 | **111.41** | -84.39 | -43.10% |
| Python & Concurrency | 1,538 | 203.93 | **116.16** | -87.77 | -43.04% |
| Explanatory Prose | 1,571 | 217.46 | **127.34** | -90.12 | -41.44% |
| Numerical Analysis | 1,589 | 224.35 | **139.25** | -85.10 | -37.93% |
| OS Architecture | 1,509 | 253.73 | **147.68** | -106.05 | -41.80% |

- **Document Win Rate:** In **30 of 30 documents (100.0%)**, the candidate achieved strictly lower cross-entropy than the baseline.
- **Paired t-test:** $p < 10^{-12}$, confirming statistically significant distribution-wide token predictive improvement.

---

### 3. Critical Capability Audit Findings

1. **What is Demonstrated:**
   - **Genuine Statistical Language Modeling:** ChakrMicro successfully models the token distribution of structured technical English, code syntax, and markdown headings.
   - **Stable Domain Adaptation:** A lightweight 80-step adaptation meaningfully lowered perplexity across 30 completely unexposed documents without catastrophic forgetting.

2. **What is NOT Demonstrated:**
   - **Next-Token Prediction $\neq$ Reasoning or Task Execution:** While perplexity dropped from 218 to 127 and Top-1 accuracy reached 17.41%, this capability reflects shallow local syntactic and lexical continuation rather than semantic truth tracking, deduction, or problem-solving.
