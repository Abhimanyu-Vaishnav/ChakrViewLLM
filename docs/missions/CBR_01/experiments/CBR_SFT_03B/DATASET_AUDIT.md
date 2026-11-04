# CBR-SFT-03B — DATASET AUDIT REPORT

---

## 1. Focused Track A Dataset Composition (50 Examples)

The CBR-SFT-03B dataset is tightly focused on conversational turn-taking, role identities, and short Q&A interactions without any entity-copying tasks:

| Category | Count | Percentage | Function & Rational |
| :--- | :---: | :---: | :--- |
| **Greetings & Openings** | 20 | 40.0% | Core polite greetings (`Good morning`, `Namaste`, `Hello`) to establish generation initiation |
| **Identity & Role** | 15 | 30.0% | Name, origins, CPU-native architecture, and independent identity |
| **Short Conversational QA** | 15 | 30.0% | Helping acknowledgments, uncertainty admissions, and turn closures |
| **TOTAL** | **50** | **100.0%** | **Focused Conversational Steering Core** |

---

## 2. Partition & Cryptographic Integrity
- **Train File (45 records):** `data/sft/cbr_sft_03b/splits/cbr_sft_03b_train_45.jsonl`
  - SHA-256: `87f89babf5ea6c4975f6bea8f39f56d218268917e34e83fb44bf9ef0b21f738c`
- **Val File (5 records):** `data/sft/cbr_sft_03b/splits/cbr_sft_03b_val_5.jsonl`
  - SHA-256: `254f88365f5a10f97a8cc5f6e7bef785b039f031d66069f2ad1e3229c67ada75`
- **Held-out Benchmark Suite (70 tasks):** `evaluation/cbr_sft_03b_suite/held_out_greeting_identity_70.json`
  - SHA-256: `0900f6867c178b681e38c9924eb7ec72bb9ae17783b49aff84005c18b1d1eb87`
- **Prompt Collisions:** Exactly **0** collisions between training and held-out evaluation prompts.
- **Response Diversity:** 100.0% unique response strings across all 50 examples.

