# CHAKRVIEW — Batches 1901–1940
## Data Leakage, Template Duplication & Provenance Audit Report

### Executive Summary

An exhaustive, multi-tier forensic audit was conducted on the Batch 1861–1900 dataset creation pipeline (`build_expanded_corpus.py`), manifest (`EXPANDED_CORPUS_MANIFEST.json`), training corpus (`train_expanded_corpus.json`), and held-out test corpus (`test_held_out_expanded_corpus.json`).

The previous report claimed that continued pretraining reduced held-out cross-entropy from **5.7817** to **4.0067** (150 steps) and **3.0989** (300 steps). However, our audit reveals critical structural flaws in the test corpus construction:
1. **100% of the held-out Prose evaluation set (8 out of 8 documents) consisted of identical repeated boilerplate placeholder sentences.**
2. **5 out of 6 domains in the training and test corpora were sampled from identical underlying synthetic files** (`stage_b_systems.txt`, `stage_b_derivations.txt`, `stage_b_algorithms.txt`, `stage_b_procedures.txt`, `stage_b_dialogues.txt`).
3. While exact 10-gram character string overlap across train and test splits was avoided by splitting contiguous blocks, the **syntactic and semantic templates** (e.g. arithmetic equations, workflow assertions, logging strings) were identical between training and test chunks, appearing hundreds of times in both.
4. **Condition D (300-step scaling)** was evaluated in-memory only and omitted from disk serialization in Batch 1861, rendering it an unpersisted single-run observation.

---

### 1. The Prose Domain Placeholder Duplication Flaw

In `build_expanded_corpus.py`, chunk extraction from `english_foundation.txt` and `english_computer_science.txt` yielded fewer than the required 22 chunks for the prose domain. The script implemented the following backfill fallback:

```python
while len(dom_chunks) < 22:
  dom_chunks.append(
      f"Domain {dom} indigenous system text specifications verifying"
      " architecture standards for ChakrView milestone batches."
  )
```

#### Forensic Finding:
- Documents `exp_doc_prose_0013` and `exp_doc_prose_0014` in the validation set, and **all 8 documents in the held-out test set** (`exp_doc_prose_0015` through `exp_doc_prose_0022`), were exact 116-character string duplicates of this boilerplate phrase.
- Each document evaluated to exactly 31 target tokens, with identical cross-entropy (5.7352 canonical; 3.0989 in Condition D).
- **Audit Verdict:** The held-out prose test set measured memorization and repetition of a single synthetic phrase, rather than natural language generalization.

---

### 2. Intra-Source Proximity Leakage Across Domains

The Batch 1861 corpus creation partitioned the corpus into 6 domains, but 5 of those domains were populated by slicing single large synthetic files generated in Stage B:

| Domain | Primary Source File | File Size (Bytes) | Train Chunks | Test Chunks | Overlap Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Technical** | `stage_b_systems.txt` | 338,537 | 10 | 8 | Direct slice from same file |
| **Mathematics** | `stage_b_derivations.txt` | 149,071 | 10 | 8 | Direct slice from same file |
| **Code** | `stage_b_algorithms.txt` | 211,699 | 10 | 8 | Direct slice from same file |
| **Propositions** | `stage_b_procedures.txt` | 71,220 | 10 | 8 | Direct slice from same file |
| **Hinglish** | `stage_b_dialogues.txt` | 139,850 | 10 | 8 | Direct slice from same file |

#### Concrete Leakage Examples:

#### A. Hinglish Domain Template Duplication
- **Train Example (`exp_doc_hinglish_0111`):**
  ```text
  arenge taaki matrix multiplication further speed up ho sake.
  Dev Log #0374 - Q: Is pipeline mein batch size 374 par CPU cache miss rate kitna measure hua?
  Dev Log #0374 - A: Profiler output ke mutabiq L1 cache hit rate 96.4% raha, jo low-resource setting ke liye kaafi solid performance hai.
  ```
- **Test Example (`exp_doc_hinglish_0125`):**
  ```text
  taaki matrix multiplication further speed up ho sake.
  Dev Log #0285 - Q: Is pipeline mein batch size 285 par CPU cache miss rate kitna measure hua?
  Dev Log #0285 - A: Profiler output ke mutabiq L1 cache hit rate 96.4% raha, jo low-resource setting ke liye kaafi solid performance hai.
  ```
- **Template Frequency:** The exact sentence substring `"L1 cache hit rate 96.4%"` occurs **379 times** in `stage_b_dialogues.txt`. The only variation between train and test chunks is the integer identifier (e.g., `#0374` vs `#0285`).

#### B. Code Domain Template Duplication
- **Train Example (`exp_doc_code_0067`):**
  ```python
  assert evaluate_tensor_kernel_v0361()['qkv'] > 0, (
      'Kernel parameter bounds violated for v0361'
  )


  def evaluate_tensor_kernel_v0362(
      dim: int = 554, heads: int = 8
  ) -> dict[str, int]:
    """Compute parameter sizing for neural transformer layer variant 362."""
```
- **Test Example (`exp_doc_code_0081`):**
  ```python
  assert evaluate_tensor_kernel_v0053()['qkv'] > 0, (
      'Kernel parameter bounds violated for v0053'
  )


  def evaluate_tensor_kernel_v0054(
      dim: int = 246, heads: int = 6
  ) -> dict[str, int]:
    """Compute parameter sizing for neural transformer layer variant 54."""
```
- **Template Frequency:** The exact pattern `assert evaluate_tensor_kernel_vXXXX()['qkv'] > 0, 'Kernel parameter bounds violated for vXXXX'` occurs **379 times** in `stage_b_algorithms.txt`.

#### C. Mathematics Domain Template Duplication
- **Train Example (`exp_doc_mathematics_0045`):**
  ```text
  The characteristic equation is det(lambda * I - M_66) = lambda^2 - 132 * lambda + 4355 = 0.
  Verification of Frobenius norm ||M_66||_F^2 = Tr(M_66^T * M_66) confirms computational consistency.
  ```
- **Test Example (`exp_doc_mathematics_0059`):**
  ```text
  The characteristic equation is det(lambda * I - M_8) = lambda^2 - 16 * lambda + 65 = 0.
  Verification of Frobenius norm ||M_7||_F^2 = Tr(M_7^T * M_7) confirms computational consistency.
  ```
- **Template Frequency:** The exact phrase `"confirms computational consistency"` occurs **239 times** across `stage_b_derivations.txt`.

#### D. Propositions Domain Template Duplication
- **Train Example (`exp_doc_propositions_0089`):**
  ```text
  Diagnostic Workflow #033 - Phase Gamma: Confirm that RMSNorm denominator epsilon prevents division by zero under zero inputs.
  ```
- **Test Example (`exp_doc_propositions_0103`):**
  ```text
  Diagnostic Workflow #116 - Phase Gamma: Confirm that RMSNorm denominator epsilon prevents division by zero under zero inputs.
  ```
- **Template Frequency:** This exact sentence occurs **139 times** in `stage_b_procedures.txt`.

---

### 3. Summary of Flaws & Severity

1. **Synthetic Template Overfitting:** The dramatic drop in cross-entropy reported in Batch 1861 (from 5.7817 to 3.0989) was primarily driven by the model memorizing fixed syntactic skeletons whose parameters (numbers, identifiers) merely varied slightly between splits.
2. **Boilerplate Prose Leakage:** The prose test split was completely invalidated by containing 8 identical copies of an unvarying system string.
3. **Absence of Independent Provenance:** Because train and test sets shared common source generators, the previous test set did not constitute an independent generalization benchmark.

### 4. Recommendation for Phase 4 Benchmark
An independent evaluation benchmark must draw exclusively from unshared raw sources (e.g. `hindi_foundation.txt`, `sanskrit_classics.txt`, `math_formulas.txt`, `code_python.txt`, `mixed_foundation.txt`, `numbers_foundation.txt`) that were **never used** in Batch 1861 training, guaranteeing source-disjointness and template independence.
