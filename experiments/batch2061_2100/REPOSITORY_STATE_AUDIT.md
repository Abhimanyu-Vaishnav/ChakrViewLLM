# ChakrView Repository State Audit — Batches 2061–2100
**Milestone:** Batches 2061–2100  
**Audit Type:** Independent Verification, Checkpoint Lineage & Working Tree Audit  
**Date:** October 2026  
**Auditor:** Independent Release Gate Agent  

---

## 1. Git Repository State

| Attribute | Recorded Value | Integrity Verification |
|:---|:---|:---:|
| **Target Branch** | `dev/v0.6-v0.7-validation` | Verified active branch |
| **HEAD Commit** | `b006c21` (`b006c21cae15c43048b3450348863b601d284ed5`) | Confirmed HEAD |
| **Milestone Ancestry** | Preceding commit `b006c21` is exact HEAD | Verified ancestor |
| **Commit Subject** | `feat(batch2021-2060): natural corpus expansion, multilingual tokenizer audit, leakage-safe evaluation and cpt readiness` | Verified |
| **Preceding Commit** | `a0f8150` (`feat(batch1981-2020): independent statistical reproduction...`) | Confirmed parent |

### Working-Tree Status
Prior to mission start, the working tree contained pre-existing user modifications in:
- `chakrview/runtime/session.py` (Modified: updated dialogue formatting to canonical ChatML / `ConversationTemplate`).
- Multiple untracked batch sprint documents and diagnostic reports (`docs/sprints/batch103_...`, `experiments/batch83_86_...`, etc.).

**Preservation Policy:** Under constraint 3.2 ("Preserve all pre-existing user changes, especially `chakrview/runtime/session.py`"), zero pre-existing modifications were reset, overwritten, or modified.

---

## 2. Protected Checkpoint & Tokenizer Cryptographic Verification

All canonical and experimental checkpoints were independently hashed using SHA-256 over their binary payloads:

| Artifact | Relative Path | Expected SHA-256 | Actual Measured SHA-256 | Status |
|:---|:---|:---|:---|:---:|
| **Canonical Checkpoint (Step 10,000)** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **MATCH (Frozen)** |
| **CPT 150-Step Candidate** | `experiments/batch1861_1900/checkpoints/cpt_balanced_seed_42.pt` | `85753315bb55853f0e8bad91a916c3846c848333a1b8e3b9b02a5256b7aa7f1c` | `85753315bb55853f0e8bad91a916c3846c848333a1b8e3b9b02a5256b7aa7f1c` | **MATCH (Frozen)** |
| **CPT 300-Step Candidate** | `experiments/batch1901_1940/checkpoints/cpt_balanced_seed_42_300steps.pt` | `2f7751511b623e1dd06bbe2d721df71e679da34d48d4b443f092600d768e7628` | `2f7751511b623e1dd06bbe2d721df71e679da34d48d4b443f092600d768e7628` | **MATCH (Frozen)** |
| **Canonical Tokenizer Config** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/config.json` | `971ca6c8db89717d2da50eec3357564d6db5c8227b233a1c6a282f9ef04d2bfb` | `971ca6c8db89717d2da50eec3357564d6db5c8227b233a1c6a282f9ef04d2bfb` | **MATCH (Frozen)** |
| **Canonical Tokenizer Vocab** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/vocab.json` | `9ee34c2b87fecf2eec253457aee2f1c8a140f7574349603071bb219803138b32` | `9ee34c2b87fecf2eec253457aee2f1c8a140f7574349603071bb219803138b32` | **MATCH (Frozen)** |
| **Canonical Tokenizer Merges** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/merges.json` | `cfaad02c1a01799276d337f711be8ae9427b2ce8f041ff636e0d37e0eb0092c6` | `cfaad02c1a01799276d337f711be8ae9427b2ce8f041ff636e0d37e0eb0092c6` | **MATCH (Frozen)** |

### Tokenizer Model File Path Clarification
The mission prompt listed:
- `release/chakrview-v0.5.0/data/experiments/vocab_4096/chakrview_tokenizer.model`
- `release/chakrview-v0.5.0/data/experiments/vocab_4096/tokenizer.json`

**Finding:** The indigenous ChakrView BPE tokenizer format is byte-level BPE consisting of `vocab.json`, `merges.json`, and `config.json`. These 3 files exist and are loaded deterministically by `chakrview.tokenizer.serialization.load_tokenizer_artifacts()`. SentencePiece `.model` / HuggingFace `tokenizer.json` files were never part of the v0.5.0 release artifact schema. This difference is formally noted and resolved.

---

## 3. Inventory of Batch 2021–2060 Deliverables

The 10 deliverable files and 5 raw source files committed in `experiments/batch2021_2060/` were inventoried:

1. `CONTROLLED_CPT_EXPERIMENT_PLAN.md` (7,151 bytes) — Pre-registered experimental protocol.
2. `CORPUS_V2_MANIFEST.json` (1,346,406 bytes, SHA-256: `4c7fc6ed...`) — Full chunked Corpus v2.
3. `DATA_LEAKAGE_AUDIT.md` (5,518 bytes) — Split isolation and n-gram analysis.
4. `DIRECT_CPT_COMPARISON.md` (4,364 bytes) — 150-step vs 300-step analysis.
5. `MULTILINGUAL_TOKENIZER_AUDIT.md` (5,404 bytes) — Tokenizer fertility evaluation.
6. `NATURAL_CORPUS_EXPANSION_REPORT.md` (10,291 bytes) — Source expansion report.
7. `SOURCE_PROVENANCE_MANIFEST.json` (6,822 bytes, SHA-256: `a6d579b5...`) — 13 cataloged sources.
8. `SPLIT_V2_MANIFEST.json` (45,759 bytes, SHA-256: `00748cbe...`) — Train/val/test assignment.
9. `TRAINING_READINESS_DECISION.md` (4,847 bytes) — Preceding milestone readiness report.
10. `corpus_ingestion_runner.py` (7,353 bytes) — Deterministic manifest generation script.
11. `data/raw_sources/authentic_sanskrit_gita.txt` (90,662 bytes)
12. `data/raw_sources/google_hinglish_top_validation.txt` (62,092 bytes)
13. `data/raw_sources/wikipedia_en_information_theory.txt` (60,705 bytes)
14. `data/raw_sources/wikipedia_en_turing_machine.txt` (45,324 bytes)
15. `data/raw_sources/wikipedia_hi_chemistry.txt` (71,477 bytes)
16. `tests/test_batch2021_2060_corpus_and_tokenizer.py` (6,016 bytes) — Existing regression suite (9 tests passing).

**Verdict on Phase A:** PASSED. Repository baseline and checkpoint lineage are completely verified and unmodified.
