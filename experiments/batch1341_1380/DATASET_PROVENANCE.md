# Dataset Provenance and Leakage Prevention Report

## Batches 1341–1380: Held-Out Technical Documentation Corpus

### 1. Data Origin and Corpus Scope

The technical documentation corpus is located in:
`experiments/batch1141_1180/data/corpus/`

The corpus comprises 52 Markdown documents covering 6 technical subsystems:
- **Billing (`billing_*.md`):** 8 documents (tiers, payment, enterprise discounts, refund policy, SLA commitments).
- **Compute (`compute_*.md`):** 10 documents (core topology, CPU SIMD, CUDA, Metal, quantization, scheduler).
- **Network (`network_*.md`):** 9 documents (mesh topology, TLS encryption, Raft consensus, socket tuning).
- **Security (`security_*.md`):** 9 documents (RBAC, auth tokens, threat model, incident response, isolation).
- **SOP (`sop_*.md`):** 8 documents (disaster recovery, log rotation, runtime environments, release checklist).
- **Storage (`storage_*.md`):** 8 documents (BM25 index, caching layer, SQLite schema, backup policy, compaction).

---

### 2. Strict Document-Level Split

To guarantee zero leakage and prevent chunk-overlap contamination, the corpus was partitioned at the **document boundary** rather than by splitting chunks or shuffling lines.

- **Split Protocol:** Balanced 50/50 division within each subsystem category:
  - **Training / Calibration Split:** 25 documents (3,212 tokens) used to fit the empirical unigram baseline.
  - **Held-Out Test Split:** 27 documents (3,456 target tokens) evaluated strictly out-of-sample.

Manifest stored in:
[experiments/batch1341_1380/data/corpus_lm_split_manifest.json](file:///D:/Project/ChakrView/experiments/batch1341_1380/data/corpus_lm_split_manifest.json)

---

### 3. Leakage and Overlap Audit

1. **Exact-Match & Shingle De-duplication:**
   - Evaluated 5-gram token overlap across training and test splits.
   - Common syntax headings (e.g. `# ChakrMicro`, `## 1. Specification`, `- Primary`) appear in both splits as standard markdown structure, but architectural values (e.g. 10485760 bytes vs 3600 seconds) are unique to specific documents.
2. **Prior Training History Check:**
   - Canonical checkpoint `training_checkpoint_step_10000.pt` was trained on synthetic pretraining shards (Batch 16–21) and Plan DSL sequences.
   - The corpus documents were authored in Batch 1141–1180 as reference documentation for the knowledge assistant retrieval tests.
   - Consequently, the technical documentation corpus represents **held-out domain text** that the neural core was never trained on during its initial 10,000 steps.
3. **No External Pretrained Models or APIs:**
   - No external pretrained weights, Hugging Face models, or remote LLM APIs were downloaded or queried.
