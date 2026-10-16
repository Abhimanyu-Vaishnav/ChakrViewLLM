# Evidence-Based Next Milestone Decision — Batches 1221–1260
## Post-Audit Architectural Diagnosis and Decision for Batches 1261–1300

### 1. Final Outcome Selection
Under Milestone Objective 8, the evidence clearly dictates:
**OUTCOME A: TRAINING OBJECTIVE AND DATA SAMPLING DEFECT FOUND.**

### 2. Confirmed Defects and Evidence
1. **The Single Same-Topic Negative Defect**:
   `train_domain_adaptive.py` trained the encoder using triplet margin loss against a *single negative passage from the same topic family*. The model was never penalized for assigning high similarities to documents from *other* topics (e.g. billing, sop, storage). As a result, generic high-frequency tokens produced "attractor chunks" in billing documents that captured $>0.95$ cosine similarity for technical compute questions.
2. **The Contrastive-to-Corpus Mismatch**:
   Pairwise triple classification ($K=2$) tests whether a positive sentence beats one chosen negative. Corpus retrieval ($K=116$) requires a document chunk to beat *all 115 other chunks*. Achieving 100% on $K=2$ is mathematically trivial compared to $K=116$ ranking.
3. **Sequence Length and Aggregation Defect**:
   The model was trained on short sentences ($\approx 15$ words), but evaluated on 400-character multi-sentence chunks. Mean pooling over long chunks diluted specific entity embeddings.

### 3. Decisions & Actionable Roadmap for Batches 1261–1300
1. **Retain BM25 Production Baseline**:
   Do **NOT** replace or degrade BM25. BM25 remains the primary, verified retrieval engine ($98.33\%$ Recall@1, $1.05\text{ ms}$ latency).
2. **Do NOT Retrain Without Architectural & Objective Upgrades**:
   Further fine-tuning the 2-layer, 64-dim `ChakrBiEncoder` on simple sentence triples is unviable. Any future neural retrieval experiment MUST implement:
   - **In-Batch Negatives / InfoNCE Loss**: Every batch of size $B$ must treat all $B-1$ other documents as negatives, directly penalizing false-positive cross-domain attraction.
   - **Chunk-Level Training**: Positive passages must match the exact 400-character chunk sizes used in runtime retrieval.
   - **Higher Embedding Dimensionality**: Scale $d_{\text{embed}}$ from 64 to 128 or 256 to prevent dense geometric collapse.
3. **Explicit Refutation Logic for False Premises**:
   Enhance `EvidenceGovernedAnsweringGate` with premise contradiction detection so that queries containing demonstrably refuted assertions (e.g., "Why does it require 128 heads?") return `decision = "REFUTE"` rather than simply outputting the true fact under `decision = "ANSWER"`.
