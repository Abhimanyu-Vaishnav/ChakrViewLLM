# NR-045: Neural Generation Failure Forensic Investigation Report

## 1. Investigation Metadata
- **Mission ID:** `NR-045`
- **Focus:** Forensic Investigation of ChakrView 1.0 RC1 Generation Incoherence
- **Repository:** `D:\Project\ChakrView`
- **Active Branch:** `exp/cbr-01-conversational-breakthrough`
- **Audited Checkpoint:** `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` (SHA-256 `CF585EBF...`)
- **Canonical Baseline:** `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` (SHA-256 `2C21632E...`)
- **Tokenizer Artifacts:** `data/experiments/vocab_4096/` (Vocab SHA-256 `DB4D6119...`)

---

## 2. Technical Findings & Root Causes Ranked by Severity

Through systematic, read-only empirical testing across all 10 mandated phases, the exact technical causes of generation incoherence have been identified and classified:

### Ranked Root Causes:

1. **Root Cause 1: Duplicate Prompt Formatting in 82.8% of Training Data (Severity: CRITICAL)**
   - *Finding:* Historical samples in `master_corpus_train.jsonl` already contained `"User: ...\nAssistant: "` in their text. The training loader wrapped them again in `<user>\n...\n<assistant>\n`.
   - *Impact:* The model was trained on `<user>\nUser: ...\nAssistant: \n<assistant>\n`. At inference time, the runtime passed `<user>\n...\n<assistant>\n`. This prompt template desynchronization immediately forced the model into an out-of-distribution state at token 0 of generation.
2. **Root Cause 2: Unmasked Prompt Loss During SFT (Severity: CRITICAL)**
   - *Finding:* In `train_conversational_sft.py`, prompt tokens were NOT masked with `-100`. Loss was calculated across the entire sequence.
   - *Impact:* Over 60% of optimization gradients were consumed predicting the user's prompt tokens rather than learning how to generate coherent responses. This produced an artificially low training loss (0.9076, PPL 2.48) while leaving the generative response heads undertrained.
3. **Root Cause 3: High Density of Single-Word Token Copy Tasks & Boilerplate (Severity: HIGH)**
   - *Finding:* 1,126 out of 5,826 training samples (19.3%) consisted of trivial 1-word or 2-word echo prompts (e.g. *"Copy token exactly: EPSILON-752-Q"*). The average response length was only 12.5 words.
   - *Impact:* The 17M model collapsed its probability mass onto short, repetitive token clusters (`"is the sum of"`, `"API development"`).
4. **Root Cause 4: Greedy Argmax Decoding at Inference (Severity: HIGH)**
   - *Finding:* Previous evaluation scripts ran greedy argmax ($T=0.0$) without repetition penalties.
   - *Impact:* Controlled generation testing proved that greedy decoding gets trapped in low-entropy loops, whereas setting $T=0.7$ with top-p sampling immediately recovers coherent factual answers, Python code (`add(a, b)`), and logical conclusions.
5. **Root Cause 5: RoPE Static Sequence Length Caching (Severity: MEDIUM)**
   - *Finding:* Static RoPE precomputation at 256 caused dimension mismatches when multi-turn dialogue history exceeded 256 tokens. Resolved via dynamic length interpolation.
6. **Cleared Component: The Tokenizer (Severity: NONE / CLEARED)**
   - *Finding:* The canonical Byte-Level BPE 4096 tokenizer achieved **100.0% round-trip fidelity** across English, Hindi Devanagari, Hinglish, Python code, and Unicode. The tokenizer is completely exonerated.

---

## 3. Core Deliverable Reports

All investigation findings are documented in full detail:
- [NR045_TOKENIZER_FORENSICS.md](file:///d:/Project/ChakrView/reports/NR045_TOKENIZER_FORENSICS.md) — 100% round-trip benchmark results.
- [NR045_DATASET_FORENSICS.md](file:///d:/Project/ChakrView/reports/NR045_DATASET_FORENSICS.md) — Template duplication and short response analysis.
- [NR045_TRAINING_OBJECTIVE_AUDIT.md](file:///d:/Project/ChakrView/reports/NR045_TRAINING_OBJECTIVE_AUDIT.md) — Prompt masking loss audit.
- [NR045_INFERENCE_PIPELINE_AUDIT.md](file:///d:/Project/ChakrView/reports/NR045_INFERENCE_PIPELINE_AUDIT.md) — Runtime prompt tracing and RoPE analysis.
- [NR045_CONTROLLED_GENERATION_RESULTS.md](file:///d:/Project/ChakrView/reports/NR045_CONTROLLED_GENERATION_RESULTS.md) — Temperature comparisons and teacher-forced divergence.
- [NR045_ROOT_CAUSE_MATRIX.md](file:///d:/Project/ChakrView/reports/NR045_ROOT_CAUSE_MATRIX.md) — Classified root cause table.
- [NR045_RECOMMENDED_REMEDIATION_PLAN.md](file:///d:/Project/ChakrView/reports/NR045_RECOMMENDED_REMEDIATION_PLAN.md) — Actionable NR-046 blueprint.

---

## 4. Post-Investigation Artifact Hash Integrity Check

All original checkpoints and tokenizer artifacts were re-hashed to verify that zero modifications occurred during the investigation:

| Checkpoint / Artifact | Verified SHA-256 | Match Baseline |
| :--- | :--- | :---: |
| `release/v1.0.0-rc1/chakr_1_0_rc1_best.pt` | `CF585EBF9AA70F630516EEC9C45E85F127E29ED91D6B6AE6A26D44FD20FCCE4E` | **MATCH (UNCHANGED)** |
| `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt` | `2C21632EBAE4F0B05B9FF7091797ECC956AC0AD00EF6EC934381223BC273B269` | **MATCH (UNCHANGED)** |
| `data/experiments/vocab_4096/vocab.json` | `DB4D6119FEDDBCC08D0CA8C580965BFC2B6D9729AED741E83044181307E7218A` | **MATCH (UNCHANGED)** |
| `data/experiments/vocab_4096/merges.json` | `ADA4B1DCFBB3D962A2E60203089881872A5063DE73B2FEF9EEFF81CE0016EFA7` | **MATCH (UNCHANGED)** |

---

## 5. Mission Verdict
`NR-045 COMPLETE: ROOT CAUSES OF RC1 GENERATION DEFECTS FULLY DIAGNOSED, PROVEN EMPIRICALLY, AND DOCUMENTED. CHECKPOINT REMAINS PARTIALLY SALVAGEABLE VIA SAMPLING, BUT HIGH-QUALITY CONVERSATIONAL CAPABILITY REQUIRES PROMPT-STRIPPED, RESPONSE-ONLY MASKED SFT (NR-046). ALL CANONICAL ARTIFACTS REMAIN 100% UNTOUCHED.`
