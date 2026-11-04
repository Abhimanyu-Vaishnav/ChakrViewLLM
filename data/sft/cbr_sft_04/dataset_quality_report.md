# CBR-SFT-04 Dataset Quality & Composition Report

## 1. Overview & Objectives

The **CBR-SFT-04** dataset is an authored, highly curated 160-example supervised fine-tuning dataset engineered to address the specific failure modes identified in CBR-SFT-01 through CBR-SFT-03B:
1. **Temporal Bias Remediation:** Elimination of the skewed morning bias seen in earlier SFT iterations via an exact 1:1:1:1 balance among Morning, Afternoon, Evening, and Generic greetings.
2. **Entity Grounding & Copying:** Explicit inclusion of 50 technical, factual, and abstention entity extraction examples to train the model to copy exact tokens (IP addresses, port numbers, hashes, personal names, and metrics) rather than hallucinate plausible generic text.
3. **Multi-Turn Context Retention:** Authoring of 40 indivisible two-turn dialogue pairs serialized with proper conversation framing (`User: ...\nAssistant: ...\nUser: ...\nAssistant: ...`) to steer the model towards retaining context from prior turns.
4. **Boundary & Identity Precision:** Rigorous negative constraints (refuting claims of being ChatGPT/LLaMA/Mistral, confirming local CPU-first indigenous architecture, and stating epistemic uncertainty).

---

## 2. Dataset Partition & Stratification

The dataset is partitioned into an exact 90% / 10% stratified split:
- **Total Examples:** 160
- **Training Partition (`train.jsonl` / `cbr_sft_04_train_144.jsonl`):** 144 examples
- **Validation Partition (`validation.jsonl` / `cbr_sft_04_val_16.jsonl`):** 16 examples

### Stratified Track Breakdown

| Track | Category Description | Total Count | Train Count (90%) | Val Count (10%) | Stratification Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Track A** | Temporal Greetings | 40 | 36 | 4 | 90.0% / 10.0% |
| *-- A1* | *Morning Greetings* | *10* | *9* | *1* | *90.0% / 10.0%* |
| *-- A2* | *Afternoon Greetings* | *10* | *9* | *1* | *90.0% / 10.0%* |
| *-- A3* | *Evening Greetings* | *10* | *9* | *1* | *90.0% / 10.0%* |
| *-- A4* | *Generic / Timeless* | *10* | *9* | *1* | *90.0% / 10.0%* |
| **Track B** | Identity, Role, Limits & Abstention | 30 | 27 | 3 | 90.0% / 10.0% |
| **Track C** | Entity Grounding, Copying & Abstention | 50 | 45 | 5 | 90.0% / 10.0% |
| **Track D** | Multi-Turn Dialogue Retention Pairs | 40 | 36 | 4 | 90.0% / 10.0% |
| **TOTAL** | **Full Dataset** | **160** | **144** | **16** | **90.0% / 10.0%** |

---

## 3. Serialization Contract

All examples adhere strictly to the ChakrMicro prompt serialization schema:
- Single-turn prompts end with `\nAssistant:`.
- Multi-turn prompts serialize full historical turns:
  `User: {u1}\nAssistant: {a1}\nUser: {u2}\nAssistant:`
- Responses begin with a leading space ` ` matching SentencePiece byte-pair tokenization conventions (e.g. `" Good morning!..."`), ensuring smooth token concatenation with no duplicated punctuation or spacing anomalies.
- Training serialization appends the special end-of-sequence token `<|eos|>`.

---

## 4. Leakage & Duplication Audit Results

The deterministic audit (`duplicate_audit.json`) confirmed:
- **Total records:** 160
- **Unique prompts:** 160 (0 exact duplicates, 0 templated collisions)
- **Train-Val Leakage:** 0 collisions
- **Train-Eval Leakage:** 0 collisions across all 80 held-out evaluation tasks
- **Val-Eval Leakage:** 0 collisions
- **Audit Verification Status:** `CLEAN_ZERO_LEAKAGE`

