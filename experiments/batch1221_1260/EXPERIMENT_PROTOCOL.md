# Experiment Protocol — Batches 1221–1260
## Training–Retrieval Gap Audit, Dataset Provenance and Evaluation Integrity

### 1. Mission Context and Objective
Batches 1221–1260 is an investigation and diagnostic milestone. Its objective is to answer the central research question:
*Why did the domain-adaptive ChakrBiEncoder report 100% accuracy on its contrastive test split, yet underperform BM25 on the held-out document-retrieval benchmark (88.33% vs 98.33% Recall@1, and 5% on pure dense vector search)?*

### 2. Methodological Standards
1. **Zero Fabrication**: All reported numbers are computed by independent Python test scripts executing over actual disk artifacts.
2. **Strict Invariant Enforcement**:
   - Canonical training checkpoint `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` must remain unmodified (`96357705a6...`).
   - Protected user file `chakrview/runtime/session.py` must not be overwritten or modified.
   - No remote git push or branch merge to `main`.
3. **Multi-Faceted Audit**:
   - Quantitative metric reproduction with explicit numerators and denominators.
   - Generalization and split leakage audit.
   - End-to-end trace from tokenization to vector similarity scoring.
   - Geometric analysis of embedding space and attractor chunks.
   - Evaluation integrity check of false-premise handling.
