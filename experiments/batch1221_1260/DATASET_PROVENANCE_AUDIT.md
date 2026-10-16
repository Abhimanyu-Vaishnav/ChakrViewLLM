# Dataset Provenance Audit — Batches 1221–1260
## Lineage, Extraction Logic, and Generalization Scope

### 1. Dataset Lineage & Generation Logic
The dataset used in Batches 1181–1220 was constructed by `experiments/batch1181_1220/data/build_dataset.py` from the 52 synthetic technical markdown documents in `experiments/batch1141_1180/data/corpus/`.

- **Total Query-Passage Contrastive Triples**: 444.
  - Positive Passage: Extracted directly from key sentences in the source document.
  - Hard Negative Passage: Extracted from a distinct document belonging to the *exact same topic family* (Storage, Network, Security, Compute, SOPs, Billing).
  - Query Variations: 4 synthetic variants per fact (Exact query, Synonym query, Paraphrase query, Hinglish query).

### 2. Generalization Scope & Separation Analysis
We audited three levels of data separation between the training set and evaluation benchmarks:

| Separation Level | Status | Evaluation Finding |
| :--- | :---: | :--- |
| **Exact Query Separation** | **PASS** | No string in `train_split.json` matches any string in `test_split.json` or the 70-item frozen benchmark. |
| **Source Document Separation** | **FAIL (Document-Shared)** | All splits (Train, Dev, Test) draw facts from the same 52 documents across the 6 topic families. The split was stratified at the fact level rather than the whole-document level. |
| **Semantic Fact Separation** | **PARTIAL** | While the 70-item benchmark contains different question wordings, many underlying parameters (e.g. `10000 entries`, `8443 port`, `15 minutes SLA`) appear in both training triples and benchmark queries. |

### 3. Generalization Claims and Constraints
- **What the Dataset Validates**: The contrastive dataset reliably tests whether a bi-encoder can distinguish an in-domain positive sentence from an in-domain hard negative distractor sentence when presented as a binary choice.
- **What the Dataset Does NOT Validate**: The dataset does **NOT** prove zero-shot generalization to unseen technical corpora, external software documentation, or unconstrained corpus-scale dense retrieval.
