# PREVIOUS RESULT REPRODUCTION & AUDIT REPORT
## Batches 1861–1900: Independent Audit of Batches 1821–1860 Results & Controls

### 1. Executive Summary & Raw Metric Recomputation

In Batches 1821–1860, continued pre-training on the canonical ChakrMicro Step 10,000 checkpoint was reported as achieving a mean held-out cross-entropy reduction from **5.3578 to 5.1261** ($\Delta = -0.2317$) and perplexity reduction from **212.26 to 168.37** across three seeds (42, 123, 999).

In Work Package 1 of Batches 1861–1900, we independently audited the raw evaluation artifacts (`experiments/batch1821_1860/results/LANGUAGE_MODELING_EXPERIMENT_RESULTS.json`), the dataset manifest (`data/CORPUS_MANIFEST.json`), the tokenization pipeline, and the checkpoint hashes.

#### Independent Recomputation Results:
- **Canonical Baseline Loss**: Recomputed exactly as **5.3578** (NLL sum: 3793.31 over 708 evaluated tokens).
- **Candidate B Loss Across Seeds**:
  - Seed 42: **5.1251** (Perplexity: 168.19)
  - Seed 123: **5.1297** (Perplexity: 168.97)
  - Seed 999: **5.1236** (Perplexity: 167.95)
  - **Mean Loss**: **5.1261 ± 0.0026** (Delta: **-0.2317**).
- **Data-Matched Control C (Shuffled Doc Order)**: **5.1290** (Perplexity: 168.86).

All headline numbers from Batches 1821–1860 **reproduce bit-exact**.

---

### 2. Systematic Audit of Methodological Decisions & Limitations

Our audit identified four critical methodological constraints in the Batch 1821–1860 study:

1. **Small Held-Out Evaluation Budget (708 Tokens)**:
   - The held-out test suite comprised only 18 short documents totaling 708 target tokens.
   - While sufficient as a preflight sanity check, 708 tokens cannot provide statistical confidence against subtle data-distribution shifts or establish broad generalization.
2. **The Control C Dilemma (Shuffled Order vs. Content Control)**:
   - Control C simply shuffled the document sequence during training. Because the model was trained with AdamW on mini-batches across 1.5 epochs, shuffling document order did not alter the training token n-gram distribution.
   - Consequently, Control C achieved a virtually identical loss (5.1290 vs 5.1261). This proved that the model did not merely memorize a fixed sequence of documents, but left open whether the improvement was driven by domain-balanced curation or general vocabulary exposure.
3. **Target Masking & Document Boundary Consistency**:
   - The previous evaluation concatenated documents individually with strict cropping: inputs $x_{0 \dots T-1}$, targets $x_{1 \dots T}$.
   - No cross-document attention leakage occurred. Special padding tokens were properly excluded from NLL sums using `ignore_index=-100`.
4. **Seed Variance Scope**:
   - The reported standard deviation ($\pm 0.0026$) represented final loss variance across runs initialized from the same canonical checkpoint (`training_checkpoint_step_10000.pt`) with different data-sampling seeds.

---

### 3. Checkpoint Lineage & Hash Verification

| Checkpoint Identifier | File Location | Expected SHA-256 | Actual Calculated SHA-256 | Audit Verdict |
| :--- | :--- | :--- | :--- | :---: |
| **Canonical Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **BIT-EXACT MATCH** |
| **Domain-Adapted Checkpoint** | `experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt` | `15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139` | `15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139` | **BIT-EXACT MATCH** |
| **Batch 1821 Seed 42 Checkpoint** | `experiments/batch1821_1860/checkpoints/chakrmicro_cpt_seed_42.pt` | `ad828feef146ec69fa861461ffad20387532d0ff1b56360c70d4c82c3c52e460` | Verified | **CONFIRMED** |

---

### 4. Directives for Batches 1861–1900

To address the limitations identified above:
1. **Scale the Evaluation Corpus**: Expand the held-out test corpus by over an order of magnitude, targeting thousands of tokens across all six operational domains.
2. **Implement a Token-Budget-Matched Content Control**: Compare the balanced multi-domain corpus against a single-domain (or skewed distribution) corpus with identical token counts.
3. **Evaluate Training Token Budget Scaling**: Compare low-budget (120 steps) vs extended-budget (300 steps) continued pretraining to test for overfitting or saturation.
