# Data Leakage & Split Integrity Audit — Batch 2981–3020
## Balanced SFT Mixture (`experiments/batch2941_2980/`)

### 1. Executive Summary

This audit independently inspects the 400-example balanced SFT dataset generated in Batch 2941–2980 (`experiments/batch2941_2980/BALANCED_SFT_DATASET.jsonl`), its split files (`train.jsonl`, `val.jsonl`, `test.jsonl`), manifest, and builder script (`build_balanced_dataset.py`).

We audit 5 distinct categories of data leakage:
1. **Exact Duplicate Leakage:** Exact string matches across train, validation, and test splits.
2. **Normalized Duplicate Leakage:** Whitespace, casing, and punctuation normalized matches across splits.
3. **Same-Source-Document Leakage:** Examples derived from the same underlying raw document crossing split boundaries.
4. **Template-Level Leakage:** Shared prompt structure or syntactic boilerplate where only variables are substituted.
5. **Semantic & Factual Overlap / Benchmark Contamination:** Overlap between training data and held-out historical benchmark suites.

---

### 2. Exact and Normalized Duplicate Audit

- **Total Dataset Size:** 400 records
  - Instruction: 280 records (70.0%) across 7 categories (40 each).
  - Replay / Base-continuation: 120 records (30.0%) across raw pretraining corpora.
- **Split Distribution:**
  - `train`: 320 records (80.0%) — 224 instruction, 96 replay.
  - `val`: 40 records (10.0%) — 28 instruction, 12 replay.
  - `test`: 40 records (10.0%) — 28 instruction, 12 replay.

#### Audit Findings:
- **Exact duplicate prompts across splits:** **0** (Disjoint sets verified).
- **Normalized duplicate prompts across splits:** **0** (Lowercased and whitespace-collapsed prompts are 100% unique across all 400 records).
- **Prompt/Target cross-contamination:** Verified zero examples where an instruction target appears as a prompt in another split.

---

### 3. Source-Document Leakage Audit (Replay Data)

In `experiments/batch2941_2980/build_balanced_dataset.py`, replay examples were extracted sequentially from 6 raw corpus files:
- `data/raw/english/english_computer_science.txt`
- `data/raw/english/english_systems.txt`
- `data/raw/english/english_foundation.txt`
- `data/raw/code/code_python.txt`
- `data/raw/code/code_foundation.txt`
- `data/raw/mathematics/math_foundation.txt`

Because the 120 replay items were partitioned by index:
- Indices 0..95 -> `train` (96 items)
- Indices 96..107 -> `val` (12 items)
- Indices 108..119 -> `test` (12 items)

#### Document Boundary Crossings Identified:
1. **`corpus_english_foundation`**:
   - Total eligible lines: 73
   - Assigned to `train`: 69 examples
   - Assigned to `val`: 4 examples
   - **Finding:** The boundary of `corpus_english_foundation.txt` crossed between `train` and `val`. While the individual text passages are distinct lines from different paragraphs, they originate from the **same source file**.
2. **`corpus_code_foundation`**:
   - Total eligible lines: 25
   - Assigned to `val`: 2 examples
   - Assigned to `test`: 9 examples
   - **Finding:** Examples from `corpus_code_foundation.txt` crossed between `val` and `test`.

**Audit Conclusion:** While exact passage leakage is zero, **same-document leakage** occurred in the replay portion because split partitioning was applied *after* concatenating corpus lines rather than partitioning documents *before* sampling windows.

---

### 4. Template-Level & Semantic Leakage Audit (Instruction Data)

The instruction set consists of 7 categories (40 items each).
- Within each category, prompts use structured prefixes:
  - `context_extraction`: `"Instruction: Extract the requested detail from the text:\n{context}\nResponse:"`
  - `context_grounded_qa`: `"Instruction: Answer the question using only the provided context.\n{context}\nResponse:"`
  - `structured_output`: `"Instruction: Output a JSON object with keys {keys} representing {entity}.\nResponse:"`
  - `short_summarization`: `"Instruction: Summarize the following passage in a concise sentence:\n{passage}\nResponse:"`
  - `bounded_text_transformation`: `"Instruction: Convert the string '{str}' to uppercase/lowercase...\nResponse:"`
  - `uncertainty_insufficient_info`: `"Instruction: Answer the question based on the provided text. If the text does not contain the answer, reply that the information is not provided.\n{context}\nResponse:"`

#### Findings:
- **Template Sharing:** High. The 40 items in each category deliberately share identical framing to teach the micro-model the syntactic contract.
- **Factual Independence:** The underlying factual entities (e.g. server names, timestamps, technical definitions) are unique per record.
- **Semantic Classification:** Semantic leakage cannot be claimed to be zero. A model exposed to 32 instances of JSON key-formatting in training is evaluating on 4 instances of similar JSON formatting in validation/test. This tests in-distribution syntactic execution rather than out-of-distribution reasoning.

---

### 5. Benchmark Suite Contamination Audit

We cross-referenced the 400 dataset examples against the two held-out historical benchmarks:
1. **28-Case Generation Benchmark** (`experiments/batch2781_2820/GENERATION_BENCHMARK.json`):
   - Overlap with `train`: **0 cases (0.0%)**
   - Overlap with `val`: **0 cases (0.0%)**
   - Overlap with `test`: **0 cases (0.0%)**
   - Overlap with raw pretraining corpus: **0 cases**
2. **24-Case SFT Benchmark** (`experiments/batch2821_2860/SFT_EVALUATION_SET.json`):
   - Overlap with `test.jsonl`: **0 cases (0.0%)**
   - Overlap with `val.jsonl`: **0 cases (0.0%)**
   - Overlap with `train.jsonl`: **21 cases**
   - **Crucial Finding:** In Batch 2821–2860, the 240-example dataset had cases 28..30 of each category designated as held-out test. In Batch 2941–2980, `build_balanced_dataset.py` imported all examples from Batch 2821–2860 and took indices 0..31 as `train`, moving cases 28..30 into the training split!
   - **Implication:** The 24-case historical benchmark in `experiments/batch2821_2860/SFT_EVALUATION_SET.json` **cannot be used as an independent test set** for models trained on Batch 2941–2980 `train.jsonl`, because 21 of its cases are present in `train.jsonl`.
   - The Batch 2941–2980 40-case `test.jsonl` remains genuinely disjoint and uncontaminated. The old 24-case suite must be labeled strictly as a **historical regression suite**, not an independent held-out benchmark.
