# CBR-SFT-05 Dataset Quality & Composition Report

## 1. Overview & Research Objectives

The **CBR-SFT-05** dataset comprises **400 meticulously engineered supervised fine-tuning examples** explicitly architected to overcome the exact entity copying and multi-turn context retention bottlenecks exposed during CBR-SFT-04:
1. **Exact Context-to-Output Copying:** Teaching the causal transformer to bind query keys to specific context values rather than outputting generic plausible completions.
2. **Key-Value & Variable Binding:** Instilling explicit associative bindings (`PROJECT=ChakrView`, `PORT=8421`) and handling variable updates.
3. **Multi-Turn Context Retention:** Advancing systematically from 2-turn fact recalls to 3-turn dialogues and 4-turn dialogues with distractor turns.
4. **Natural Conversational Transfer:** Embedding structured retrieval capabilities into natural technical and operational dialogues.

---

## 2. Dataset Composition & Stratification

| Track | Category Description | Total Records | Train Split (90%) | Val Split (10%) | Stratification Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Track A** | Exact Copy & Entity Extraction | 100 | 90 | 10 | 90.0% / 10.0% |
| *-- A1* | *IP Addresses (15)* | *15* | *13* | *2* | *86.7% / 13.3%* |
| *-- A2* | *Port Numbers (15)* | *15* | *14* | *1* | *93.3% / 6.7%* |
| *-- A3* | *Commit Hashes (15)* | *15* | *13* | *2* | *86.7% / 13.3%* |
| *-- A4* | *Paths & Filenames (15)* | *15* | *14* | *1* | *93.3% / 6.7%* |
| *-- A5* | *Technical Codes & IDs (15)* | *15* | *13* | *2* | *86.7% / 13.3%* |
| *-- A6* | *Punctuation Boundaries (15)* | *15* | *14* | *1* | *93.3% / 6.7%* |
| *-- A7* | *Candidate Selection (10)* | *10* | *9* | *1* | *90.0% / 10.0%* |
| **Track B** | Key-Value & Variable Binding | 100 | 90 | 10 | 90.0% / 10.0% |
| *-- B1* | *Single Key-Value Binding (30)* | *30* | *27* | *3* | *90.0% / 10.0%* |
| *-- B2* | *Multi-Key Binding Blocks (30)* | *30* | *27* | *3* | *90.0% / 10.0%* |
| *-- B3* | *Variable Updates & Recency (20)* | *20* | *18* | *2* | *90.0% / 10.0%* |
| *-- B4* | *Unknown Keys & Abstention (20)* | *20* | *18* | *2* | *90.0% / 10.0%* |
| **Track C** | Multi-Turn Context Retention | 100 | 90 | 10 | 90.0% / 10.0% |
| *-- C1* | *Two-Turn Context Recall (30)* | *30* | *27* | *3* | *90.0% / 10.0%* |
| *-- C2* | *Three-Turn Context Recall (30)* | *30* | *27* | *3* | *90.0% / 10.0%* |
| *-- C3* | *Four-Turn with Distractors (40)* | *40* | *36* | *4* | *90.0% / 10.0%* |
| **Track D** | Natural Conversational Transfer | 100 | 90 | 10 | 90.0% / 10.0% |
| *-- D1* | *Natural Dialogue Topics (90)* | *90* | *81* | *9* | *90.0% / 10.0%* |
| *-- D2* | *Rich Multi-Turn Scenarios (10)* | *10* | *9* | *1* | *90.0% / 10.0%* |
| **TOTAL** | **Full CBR-SFT-05 Dataset** | **400** | **360** | **40** | **90.0% / 10.0%** |

---

## 3. Serialization Contract & Sequence Lengths

- **Serialization Format:**
  `User: {u1}\nAssistant: {a1}\nUser: {u2}\nAssistant: {a2}...`
- **Response Format:** Begins with leading space `" "` matching subword SentencePiece byte tokenization conventions.
- **Sequence Length Statistics (Tokens):**
  - Minimum tokens: **51**
  - Maximum tokens: **182**
  - Mean tokens: **91.7**
  - Canonical Context Limit: **512 tokens** (all sequences strictly fit with 330+ token headroom).

---

## 4. Leakage & Duplication Verification

Audited programmatically via `scripts/build_cbr_sft_05_dataset.py`:
- **Unique Prompts in Dataset:** 400 / 400 (0 exact duplicates)
- **Train / Validation Leakage:** 0 collisions
- **Train / Evaluation Leakage:** 0 collisions across all 120 held-out evaluation tasks
- **Validation / Evaluation Leakage:** 0 collisions
- **Status:** `VERIFIED_ZERO_LEAKAGE`

