# Final Mission Report — Batches 1221–1260
## Training–Retrieval Gap Audit, Dataset Provenance and Evaluation Integrity

### 1. Executive Summary
Batches 1221–1260 conducted a diagnostic audit to investigate the training-to-retrieval performance gap of ChakrView's indigenous `ChakrBiEncoder`. We verified metric reproduction, audited dataset provenance, performed geometric embedding space analyses, and evaluated false-premise handling.

**Core Finding**: The contrastive test accuracy of 100% was not an evaluation shortcut or data leak, but the result of evaluating an easy pairwise binary task ($K=2$ classification) that failed to prepare the model for unconstrained corpus retrieval ($K=116$). Because training only used one negative per topic family and lacked in-batch negatives, generic vocabulary chunks in unrelated documents became "false-positive attractors", capturing $>0.95$ cosine similarity and collapsing pure dense retrieval to 5.0%.

### 2. Metric Reproduction Summary (100% Verified)
Every reported headline metric from Batches 1181–1220 was independently verified and reproduced:
- **BM25 Lexical Recall@1**: **98.33% (59/60)**, MRR = 0.9833, Accuracy = 84.29% (59/70), Median Latency = 1.04 ms.
- **Deterministic Hash Hybrid Recall@1**: **98.33% (59/60)**, MRR = 0.9833, Accuracy = 84.29% (59/70), Median Latency = 1.02 ms.
- **Untrained Bi-Encoder Recall@1**: **91.67% (55/60)**, MRR = 0.9556, Accuracy = 85.71% (60/70), Median Latency = 4.23 ms.
- **Domain-Adaptive Bi-Encoder Recall@1**: **88.33% (53/60)**, MRR = 0.9333, Accuracy = 84.29% (59/70), Median Latency = 3.90 ms.
- **Ablation (Pure Dense Vector Search)**: **5.00% (3/60)** Recall@1, 11.67% Recall@3.

### 3. Confirmed Defects and Hypotheses Tested
1. **Hypothesis: Data Leakage in Test Split?**
   - *Tested & Rejected*: Normalized hashes showed zero exact query or fact overlap between train and test splits.
2. **Hypothesis: Inference vs Training Architecture Mismatch?**
   - *Tested & Rejected*: Model architecture, tokenization (BPE 4,096 vocab), LayerNorm, and $L_2$ normalization are identical in training and runtime.
3. **Hypothesis: Single Same-Topic Negative Defect?**
   - *Tested & Confirmed*: Triplet loss against a single same-domain negative never penalized cross-domain high similarity. In-memory billing chunks reached $>0.95$ cosine similarity for compute queries, degrading hybrid retrieval rankings.
4. **Hypothesis: Paragraph Length Mismatch?**
   - *Tested & Confirmed*: Model trained on $\approx 15$-word sentences was tested on 400-char paragraph chunks, diluting mean-pooled token vectors.

### 4. False-Premise Evaluation Resolution
The 0% accuracy reported on the 5 false-premise questions was explained:
- The assistant correctly retrieved the true facts and answered with factual evidence (e.g. citing 6 attention heads instead of 128, and $49/mo instead of $9,000/mo).
- However, because the system outputted `decision = "ANSWER"` instead of explicitly setting `decision = "REFUTE"`, the evaluator recorded a decision mismatch. No hallucination occurred.

### 5. Invariant & Checkpoint Integrity
- **Canonical Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **Initial SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Final SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (**100% Bit-for-Bit Match**).
- **Protected Files**: `chakrview/runtime/session.py` was strictly preserved and unmodified.

### 6. Test Suite Status
- Total regression test suite: **22 passed, 0 failed** in 2.85s.
