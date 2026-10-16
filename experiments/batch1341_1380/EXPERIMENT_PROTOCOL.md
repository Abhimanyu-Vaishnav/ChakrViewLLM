# Experiment Protocol: Batches 1341–1380

## ChakrMicro Next-Token Prediction and Language Modeling Evaluation

### 1. Mission Mandate and Objective

The objective of Batches 1341–1380 is to evaluate the foundational language modeling capabilities of the canonical `ChakrMicro` neural core (`training_checkpoint_step_10000.pt`) on held-out technical documentation.

This mission strictly separates learned neural capabilities from the deterministic BM25 retriever, regex rules, and extractive evidence selection mechanisms. We answer the primary scientific question:

> **Can the existing canonical ChakrMicro checkpoint predict held-out next tokens better than statistical baselines (uniform random, empirical unigram) on a reproducible, leakage-controlled dataset?**

---

### 2. Experimental Protocol & Invariants

1. **Frozen Canonical Checkpoint:**
   - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
   - Canonical SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
   - Strict read-only access. No parameter modification or retraining allowed in this phase.

2. **Tokenizer and Architecture Alignment:**
   - Vocabulary: 4096 tokens (`release/chakrview-v0.5.0/data/experiments/vocab_4096`).
   - Architecture: `ChakrMicro` (d_model=192, n_layers=6, n_heads=6, hidden_dim=512, max_seq_len=512, tied embeddings).
   - Verified total parameter count: 4,229,568 (3,443,136 unique parameters when weight tying is applied).

3. **Strict Causal Masking & Shift Invariance:**
   - Causal sequence modeling requires predicting target $y_t = x_{t+1}$ given context $x_{1:t}$.
   - Input: $X = [x_0, x_1, \dots, x_{T-2}]$
   - Target: $Y = [x_1, x_2, \dots, x_{T-1}]$
   - Loss: Token-level Cross-Entropy: $\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log P(y_i \mid x_{\le i})$
   - Perplexity: $PPL = \exp(\mathcal{L})$.

4. **Comparative Baseline Suite:**
   - **Baseline A (Uniform Random):** Cross-Entropy = $\ln(4096) \approx 8.3178$, $PPL = 4096.0$, Top-1 Accuracy = $1/4096 \approx 0.0244\%$.
   - **Baseline B (Corpus Empirical Unigram):** Cross-entropy and top-k accuracy calculated from training-split token unigram distribution.
   - **ChakrMicro (Canonical Checkpoint):** Evaluated in strict `model.eval()` mode with zero dropout, no gradients.

5. **Split Design & Leakage Controls:**
   - Corpus: 52 technical documentation files (`experiments/batch1141_1180/data/corpus/*.md`).
   - Split: Split by document (26 train / calibration documents, 26 held-out test documents).
   - Zero document-level leakage.

6. **Safety & Protected Files:**
   - `chakrview/runtime/session.py` remains untracked and unstaged.
   - All experiment outputs written exclusively to `experiments/batch1341_1380/`.
