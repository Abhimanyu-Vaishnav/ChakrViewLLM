# EXPANDED CORPUS AND PROVENANCE REPORT
## Batches 1861–1900: Data Scaling, Domain Composition & Provenance Manifest

### 1. Scaling Rationale & Dataset Architecture

In Batches 1821–1860, the held-out language modeling evaluation set contained only 18 documents and 708 target prediction tokens. While sufficient to verify preflight pipeline execution, 708 tokens was too limited to establish statistical confidence across domain shifts.

In Work Package 2 of Batches 1861–1900, we compiled an **expanded multi-domain corpus** drawn from the project's indigenous technical and mathematical texts under `data/raw/`:
- **Training Set (`train_expanded_corpus.json`)**: 60 multi-paragraph documents (**17,381 tokens**).
- **Validation Set (`val_expanded_corpus.json`)**: 24 documents (**~6,500 tokens**).
- **Held-Out Test Set (`test_held_out_expanded_corpus.json`)**: 48 documents (**12,527 target prediction tokens**), representing an **17.7× expansion** in evaluation volume over Batch 1821 (708 tokens).
- **Control Train Set (`control_skewed_train_corpus.json`)**: 60 documents (**20,782 tokens**) skewed exclusively on code and mathematics to provide a content-distribution control.

---

### 2. Domain Composition & Provenance Mapping

All raw documents originate from project-authored synthetic specifications, algorithmic implementations, and CC0 classical texts indexed in `data/raw/manifest.json`. No external web scrapes or untracked proprietary data were used.

| Evaluation Domain | Source Files in `data/raw/` | Number of Held-Out Docs | Evaluated Tokens in Held-Out Set | Provenance Description |
| :--- | :--- | :---: | :---: | :--- |
| **General Prose** | `english_foundation.txt`, `english_computer_science.txt` | 8 | 1,842 | System design principles, cognitive architecture theory. |
| **Technical Docs** | `stage_b_systems.txt`, `english_systems.txt` | 8 | 1,914 | Operating systems, kernel namespaces, cgroups, virtual memory. |
| **Mathematics** | `stage_b_derivations.txt`, `math_foundation.txt` | 8 | 2,108 | Linear algebra, calculus, eigenvalues, probability theory. |
| **Python Code** | `stage_b_algorithms.txt`, `code_foundation.txt` | 8 | 2,345 | Data structures, algorithms, sorting, matrix operations. |
| **Propositions** | `stage_b_procedures.txt` | 8 | 1,760 | Formal logic, governance rules, execution policies. |
| **Hinglish** | `stage_b_dialogues.txt`, `hinglish_foundation.txt` | 8 | 2,558 | Technical conversations in Romanized Hindi. |
| **Total Held-Out** | **6 Distinct Domains** | **48** | **12,527** | Sealed in `EXPANDED_CORPUS_MANIFEST.json` |

---

### 3. Leakage & Overlap Audit

1. **Document-Level Decoupling**: Paragraph chunks for training, validation, and testing were sampled from disjoint source document offsets, ensuring zero identical paragraph repeats between splits.
2. **N-Gram Overlap Check**: Exact 10-gram match between `train_expanded_corpus.json` and `test_held_out_expanded_corpus.json` was audited and confirmed at **0.0%**.
3. **Immutability**: The held-out dataset was sealed prior to training and remained strictly excluded from all training loops, validation checks, and optimizer updates.
