# Regression and Forgetting Report

## Batches 1421–1460: Auxiliary Evaluation and Retention Analysis

### 1. Multi-Benchmark Retention Matrix

We evaluated the candidate model (`Step 10,080`) across all available benchmarks to detect any regression or catastrophic forgetting:

| Benchmark / Evaluation Surface | Target Volume | Baseline (Step 10k) | Candidate (Step 10,080) | Outcome / Status |
|---|---|---|---|---|
| **Old Held-Out Test Set** | 27 docs (3,456 targets) | 158.07 PPL | **76.49 PPL** | **+51.6% Improvement** |
| **New Independent Technical Corpus** | 30 docs (7,741 targets) | 218.15 PPL | **127.70 PPL** | **+41.5% Improvement** |
| **Python Code Generation Probes** | Multi-token completions | Incomplete docstring | Valid method signature | **Preserved / Improved** |
| **Plan DSL Formatting Probes** | Plan token representations | Byte alignment intact | Byte alignment intact | **Zero Regression** |
| **Local Knowledge Assistant Regression** | 70 frozen benchmark queries | 98.6% accuracy | 98.6% accuracy | **100% Invariant** (BM25 retriever isolated) |

---

### 2. General-Language Retention Boundary

- **Finding:** ChakrView does not possess an external multi-gigabyte general web pretraining corpus (e.g. Common Crawl or Wikipedia) in tree.
- **Honest Boundary Statement:** While the candidate model demonstrably improves across 5 major technical disciplines (OS, Python, Distributed Systems, Math, and Architecture) and preserves Plan DSL syntax, **general-domain open-ended conversational English retention cannot be definitively established** due to the absence of a large-scale general-domain crawl corpus.
- **Policy Invariant:** In accordance with the project charter, the candidate checkpoint remains strictly an experimental research candidate and **is NOT promoted to overwrite the canonical pre-training checkpoint**.
