# CR-16E-R2 Phase F: Independent Span-Generalization Validation

### 1. Verification of the Evaluation Distribution
A critical finding emerged from our forensic audit of **Benchmark B3 (Multi-Token Spans, $N=500$ probes)**:
- **Lexical Composition:** **100% (500 / 500)** of the target entity values tested in Benchmark B3 (e.g. `Copenhagen`, `Hippopotamus`, `Earthquake`, `Dandelion`, `Motorcycle`) were **present in the training vocabulary dictionary** of the balanced training datasets.
- There were **0 unseen entity values** in Benchmark B3.
- When tested on genuine unseen target values in the sealed CR-16C suite (`CAT7: familiar_keys_unseen_values`, $N=150$ probes), accuracy across all models was **strictly 0.0% (0 / 150)**.

---

### 2. Dissection of the Span Generation Capability

We must distinguish carefully between:
1. **First-Token Retrieval:** Locating the correct slot and emitting the initial subword.
2. **Conditional Continuation:** Continuing subword generation once the first token is retrieved.
3. **Full Exact-Match Generation:** Emitting the entire multi-token sequence and terminating correctly.

#### Table F1: Length-by-Length Generative Dissection (Balanced Models vs Standard Models)

| Length / Metric | Standard Models (Arms A & B) | Balanced Models (Arms C & D) | Arm D Seed 4040 (Peak) |
|---|---|---|---|
| **Length 1 (150 probes)** | | | |
| ├── First Token Match | 41.8% (62.7 / 150) | 22.7% (34.0 / 150) | 34.0% (51 / 150) |
| ├── Exact Match | **25.7% (38.5 / 150)** | 16.9% (25.3 / 150) | 28.0% (42 / 150) |
| └── Premature EOS / Drift | 38.5% | 25.5% | 17.6% |
| **Length 2 (150 probes)** | | | |
| ├── First Token Match | 23.0% (34.5 / 150) | 28.2% (42.3 / 150) | 30.0% (45 / 150) |
| ├── Exact Match | 12.3% (18.5 / 150) | **24.2% (36.3 / 150)** | **25.3% (38 / 150)** |
| └── Conditional Continuation | 53.5% | **85.8%** | **84.4%** |
| **Length 3 (100 probes)** | | | |
| ├── First Token Match | 14.7% (14.7 / 100) | 41.7% (41.7 / 100) | 53.0% (53 / 100) |
| ├── Exact Match | **0.0% (0 / 100)** | **37.3% (37.3 / 100)** | **49.0% (49 / 100)** |
| └── Conditional Continuation | **0.0% (Complete Drift)** | **89.4% (Completion)** | **92.5% (Completion)** |
| **Length 4 (100 probes)** | | | |
| ├── First Token Match | 8.2% (8.2 / 100) | 41.5% (41.5 / 100) | 58.0% (58 / 100) |
| ├── Exact Match | **0.0% (0 / 100)** | **40.2% (40.2 / 100)** | **57.0% (57 / 100)** |
| └── Conditional Continuation | **0.0% (Complete Drift)** | **96.9% (Completion)** | **98.3% (Completion)** |

---

### 3. Scientific Conclusions on Span Generalization

1. **What Was Validated:**
   - Balanced-span training definitively solved the **autoregressive continuation bottleneck** for familiar multi-token entity names.
   - When the model identifies the first token of a 3-token or 4-token word, it completes the full multi-token sequence with **$89.4\% - 96.9\%$ conditional fidelity**, proving that the 6-layer transformer does not suffer from intrinsic autoregressive drift on learned multi-token words.
2. **What Was NOT Validated (The Unseen Value Limitation):**
   - The capability does **not** generalize to unseen out-of-vocabulary entity strings ($0.0\%$ on CAT7).
   - The model learns to recall and complete multi-token sequences for entities represented in its parameter weights; it does not perform generic arbitrary subword span extraction on arbitrary novel words.
3. **The Length-1 Trade-off:**
   - Shifting the training prior toward longer spans slightly depressed 1-token precision ($25.7\% \to 16.9\%$), as the model occasionally appended extraneous continuation tokens.

