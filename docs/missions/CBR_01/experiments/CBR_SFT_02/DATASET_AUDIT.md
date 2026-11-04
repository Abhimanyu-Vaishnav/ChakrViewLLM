# CBR-SFT-02 — DATASET AUDIT REPORT

---

## 1. Composition & Category Stratification

The CBR-SFT-02 dataset contains exactly 600 records constructed to resolve the chat turn-taking defect while reinforcing multi-turn context and dynamic entity copying:

| Category | Total Examples | Train Split (90%) | Val Split (10%) | Percentage | Function |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Chat-scaffolded Instruction** | 300 | 270 | 30 | 50.0% | General technical, reasoning, and programming tasks from Exp34 v3 wrapped in `User: ...\nAssistant:` |
| **Greetings & Role Interaction**| 50 | 45 | 5 | 8.33% | Conversational greetings, identity acknowledgment, and polite interaction |
| **Multi-turn Context Retention**| 100 | 90 | 10 | 16.67%| Verified two-turn conversations where Turn 2 strictly requires Turn 1 entities |
| **Dynamic Entity-Copying** | 150 | 135 | 15 | 25.0% | Span copying of dynamic entities, departments, equipment, and locations |
| **TOTAL** | **600** | **540** | **60** | **100.0%** | **Balanced Conversational Curriculum** |

---

## 2. Cryptographic Hashes & Partition Integrity
- **Train File:** `data/sft/cbr_sft_02/splits/cbr_sft_02_train_540.jsonl`
  - SHA-256: `82214c95c47ea44350c00635074fef201873cea253eafc5d2d8541dbb85bbbf0`
- **Val File:** `data/sft/cbr_sft_02/splits/cbr_sft_02_val_60.jsonl`
  - SHA-256: `6f044dd2a12bbe56d1409243f442881b4fdc708ab24aaa05cafb0eeac0cf6af2`
- **Train/Val Overlap:** Exactly **0** prompt collisions, **0** record ID collisions.
- **Response Diversity:** **86.7%** unique response strings (468 unique across 540 train records).

---

## 3. Benchmark Contamination & Leakage Audit
- **Overlap with M3 Conversational Suite (60 Tasks):** Exactly **0** collisions.
- **Overlap with Held-out Dynamic Entity Grounding (100 Tasks):** Exactly **0** collisions.
- **Overlap with Held-out Multi-Turn Context (50 Tasks):** Exactly **0** collisions.
- **Provenance & Review Status:** Derived exclusively from verified Exp34 v3 instructions and synthetic role/context templates under internal research governance.

