# ChakrView Dataset Provenance Register

**Repository:** `ChakrView`  
**Location:** `docs/data/DATASET_PROVENANCE_REGISTER.md`  
**Purpose:** End-to-end lineage, source file mapping, transformation history, and forensic attribution evidence.  
**Version:** 1.0.0  
**Effective Date:** 2026-10-30  
**Status:** Living Governance Document  

---

## 1. Provenance Verification Standards

Forensic verification requires documenting the precise origin of every token, distinguishing between verified local procedural generation, local raw text ingestions, and untracked generative artifacts.

Evidence confidence ratings:
- `VERIFIED`: Complete, auditable script code and exact local inputs exist in repository history.
- `REPORTED`: Declared in metadata, manifests, or commit messages, but missing formal bibliographic chain of title.
- `INCOMPLETE`: Critical generation components or source files are absent from the active workspace.

---

## 2. Dataset Lineage & Forensic Evidence

### 2.1 Exp34 Instruction Mixture V1 (`DS-EXP34-V1`)
- **File Path:** `data/sft/exp34_instruction_mixture_train.jsonl`
- **Total Records:** 1,000 | **File Size:** 73,105 bytes
- **Full SHA-256:** `56571858de253bb0aeca700e584a483eb5dfdba33eb2910ca1b3f7b2d0aee532`
- **Compiler Script:** `scripts/compile_exp34_dataset.py`
- **Component Generators:**
  - `scripts/generate_full_1000_dataset.py` (Category 1: Technical Explanations, 100 records)
  - `scripts/generate_dataset_categories_2_to_5.py` (Categories 2–5: Python Programming, Code Debugging, Algorithmic Reasoning, Arithmetic Reasoning; 100 records each)
  - `scripts/generate_dataset_categories_6_to_10.py` (Categories 6–10: Passage Summarization, Constrained Formatting, Conversational Clarification, Hindi Comprehension, Bilingual Hinglish; 100 records each)
- **Input Sources:** Indigenous synthetic prompt templates and rule-based algorithmic generators.
- **Transformations:** Fixed string prefix formatting, automated question-answer pair construction.
- **Forensic Findings:** 1,000 unique prompts (100%), but only 917 unique responses (83 duplicates across Categories 6, 9, 10).
- **Unresolved Evidence:** While authored by project engineers, individual category scripts lack standalone bibliographic notes confirming all code solutions were written from scratch rather than adapted from standard textbook problems.

---

### 2.2 Exp34 Candidate V2 (`DS-EXP34-V2`)
- **File Path:** `data/sft/exp34_candidate_v2/exp34_instruction_mixture_train_v2.jsonl`
- **Total Records:** 1,000 | **File Size:** 74,374 bytes
- **Full SHA-256:** `6836ff38e16d012e8577c77171d8853bc59e44cb5cb1d8a14af4dbf217dd2010`
- **Compiler Script:** `scripts/compile_exp34_candidate_v2.py`
- **Component Generators:**
  - `scripts/generate_candidate_v2_category2.py`
  - `scripts/generate_candidate_v2_categories3_to_5.py`
  - `scripts/generate_candidate_v2_categories6_to_10.py`
- **Direct Source Dependencies:**
  - **Category 6 (Passage Summarization - 100 lines):** Ingests lines from `data/raw/english/english_foundation.txt`, `data/raw/english/english_computer_science.txt`, and `data/raw/english/english_systems.txt`.
  - **Category 9 (Hindi Comprehension - 100 lines):** Ingests lines from `data/raw/hindi/hindi_foundation.txt`, `data/raw/hindi/hindi_computing.txt`, and `data/raw/hindi/hindi_science_culture.txt`.
  - **Category 10 (Bilingual Hinglish - 100 lines):** Ingests lines from `data/raw/hinglish/hinglish_foundation.txt` and `data/raw/hinglish/hinglish_tech.txt`.
- **Transformations:** Deterministic prompt wrapping (e.g. `Summarize the core technical takeaway...`, `Hinglish engineering discussion #{i}...`) and response synthesis referencing the ingested line.
- **Forensic Findings:** Successfully achieved 1,000 unique prompts (100%) and 1,000 unique responses (100%).
- **Unresolved Evidence:** `data/raw/manifest.json` asserts that the raw corpus contains "indigenous synthesis & public domain classical texts". However, the specific historical editions, translators, and copyright expiration dates for the Hindi and Sanskrit classical materials are not explicitly cited.

---

### 2.3 Milestone 3 Conversational Mixture Suite (`DS-M3-*`)
- **Files:**
  - Full Mixture: `data/sft/m3_conversational/m3_conversational_mixture_600.jsonl` (SHA-256: `1706aa1db077f4a73e9360553188f0bae5c4fa1c516bf89c7ebe8daecb4385fb`)
  - Train Split (540): `data/sft/m3_conversational/splits/m3_train_split_540.jsonl` (SHA-256: `a5f2c4813a52fa0a2ce0bc2d18f6d3c42ffbc6c62705eb07913f7841f7212573`)
  - Validation Split (60): `data/sft/m3_conversational/splits/m3_val_split_60.jsonl` (SHA-256: `53ac3009d9cea2c8a37cb3a808661e78de6dea16e493622f7eecbdd4376d8c58`)
- **Total Records:** 600 total (540 train / 60 val)
- **Compiler Script:** `scripts/compile_m3_dataset.py`
- **Input Sources:** 15 conversational category definitions hardcoded directly within the compiler script body.
- **Generation & Splitting Methodology:**
  - For each category, 10 base dialogue seeds were manually authored.
  - An expansion loop scaled each category to exactly 40 records by appending session identifiers (e.g., `[Session {i+1}]`, `(Ref #{i+1})`, `(Rule {i+1})`).
  - A deterministic stratified 90/10 split was applied using `seed=42`.
- **Forensic Findings:** 
  - There are exactly **125 unique response strings** across all 600 records (20.8% response diversity).
  - Train split has 121 unique responses; validation split has 50 unique responses.
  - Zero exact prompt string leakage between train and validation. However, validation prompts are indexed variations of the training seed templates.
- **Unresolved Evidence:** The compiler script declares `"source_type": "human_authored_conversational"` and `"license": "CC0-1.0"`. Formal contributor copyright waiver documentation has not been filed in repository governance.

---

### 2.4 Milestone 5 / Exp35 DPCE Suite (`DS-EXP35-*`)
- **Files & Actual Workspace Locations:**
  - Full Mixture: `experiments/exp35_dpce/data/exp35_full_dataset.jsonl` (SHA-256: `ac247d1e9a61b945640b094d3dda63a355fba704d362820d2225ebdac20b8c8d`)
  - Train Split (900): `experiments/exp35_dpce/data/exp35_train.jsonl` (SHA-256: `5b10b7d58e891e3cadb1213f436417dd34701b6ed3e0573ba2d534a786e7ad1d`)
  - Validation Split (100): `experiments/exp35_dpce/data/exp35_validation.jsonl` (SHA-256: `a30d633f8006ae1175cda8d3277c7b94f123665dfaf555f54263deace68f1a6a`)
  - Benchmark (200 tasks): `experiments/exp35_dpce/benchmark/exp35_benchmark_200.json` (SHA-256: `29c2739ce0effd164a7d884fea04ce20f22287c73705829b9d2671705f016e00`)
- **Total Records:** 1,000 dataset records + 200 benchmark tasks.
- **Compiler Script:** *Unavailable on disk.* The procedural script referenced in earlier planning was executed transiently during the M5 research session.
- **Input Sources:** Synthetically controlled fictional entity pools (names: *Zoya, Kabir, Meera, Nisha, Rohan, Tara*; stations: *101–105*; items: *papaya, mango*).
- **Forensic Findings:**
  - 100% unique prompt strings (1,000/1,000) and 100% unique response strings (1,000/1,000).
  - Clean boundary isolation: Zero prompt or target overlap between training split and the 200 benchmark tasks.
- **Unresolved Evidence:** 
  - The generator source code is missing, preventing independent regeneration.
  - The JSONL records omit the `"license"` attribute entirely (`UNDECLARED`), conflicting with the external markdown quality report asserting `CC0-1.0`.

