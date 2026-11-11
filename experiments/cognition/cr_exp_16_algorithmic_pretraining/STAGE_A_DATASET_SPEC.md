# CR-EXP-16 Stage A Dataset Specification
## Algorithmic Sequence Copying Dataset

**Mission Identifier:** CR-EXP-16  
**Auditor / Engineer:** Autonomous Scientific Research & Engineering Team  
**Date:** November 2026  

---

## 1. Design Objectives & Token Vocabulary

The Stage A dataset is synthesized to isolate pure sequence reproduction without semantic or conversational priors:
- **Lexical Pool:** 80 distinct natural symbols and numbers (colors, animals, elements, directions, digits) selected to provide rich token representation across vocabulary space.
- **Sequence Lengths:** Dynamically varied from 1 token to 8 tokens.
- **Token Repetitions:** 20% of sequences contain internal repeated tokens (e.g. `red red blue red`).
- **Prompt Variety:** 5 distinct structural framing templates (`Echo sequence:`, `Repeat:`, `Copy:`, `Input: ... Copy:`, `Sequence: ... -> ...`) to decouple the copy operation from any single textual trigger.

---

## 2. Dataset Quantities and Hashes

- **Training Dataset:** `experiments/cognition/cr_exp_16_algorithmic_pretraining/data/stage_a_train.json`
  - Total Samples: 1,000
  - Unique Sequences: 933
  - SHA-256: `a1bd50eea3c030442c14f26d62fafa3812fd3b8e40e9bb9ddb2a101aee361deb`
- **Validation Dataset:** `experiments/cognition/cr_exp_16_algorithmic_pretraining/data/stage_a_val.json`
  - Total Samples: 150
  - SHA-256: `82862e151e4ee576b2c1490120a6719dba3d603cb4223446cb26763d7a10dcde`

