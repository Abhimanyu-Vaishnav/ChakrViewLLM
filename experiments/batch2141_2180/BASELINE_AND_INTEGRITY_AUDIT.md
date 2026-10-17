# ChakrView Baseline and Experiment Integrity Audit — Batches 2141–2180
**Milestone:** Batches 2141–2180  
**Audit Phase:** Workstream A — Repository Audit and Protected Asset Recovery  
**Date:** October 2026  
**Auditor:** Lead ML Research & Release Auditor Agent  

---

## 1. Git Repository State & Working Tree Audit

| Attribute | Measured Repository Value | Status |
|:---|:---|:---:|
| **Repository Path** | `D:\Project\ChakrView` | Confirmed active root |
| **Active Branch** | `dev/v0.6-v0.7-validation` | Verified target branch |
| **Current HEAD Commit** | `6955858` (`6955858e3ce68571da268e0e7a17537b0181519b`) | Confirmed HEAD |
| **Ancestry Check** | `18fad0f` is direct parent of `6955858` | Verified ancestry |
| **Working-Tree Modifications** | `chakrview/runtime/session.py` contains pre-existing user modifications (ChatML format). Preserved untouched. | **PROTECTED & PRESERVED** |
| **Untracked Directories** | `experiments/batch103_...` through `batch119_...`, sprint docs. Untouched. | **PROTECTED & PRESERVED** |

---

## 2. Protected Baseline & Cryptographic Hash Verification

All canonical model checkpoints, historical candidates, tokenizer files, and manifests were hashed independently via SHA-256:

| Artifact Name | Path | Expected Hash | Actual Measured SHA-256 | Status |
|:---|:---|:---|:---|:---:|
| **Canonical Pretraining Checkpoint (Step 10,000)** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | `96357705a683...` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED (Immutable)** |
| **Historical 150-Step Candidate** | `experiments/batch1861_1900/checkpoints/cpt_balanced_seed_42.pt` | `85753315bb55...` | `85753315bb55853f0e8bad91a916c3846c848333a1b8e3b9b02a5256b7aa7f1c` | **VERIFIED (Immutable)** |
| **Historical 300-Step Candidate** | `experiments/batch1901_1940/checkpoints/cpt_balanced_seed_42_300steps.pt` | `2f7751511b62...` | `2f7751511b623e1dd06bbe2d721df71e679da34d48d4b443f092600d768e7628` | **VERIFIED (Immutable)** |
| **Tokenizer Config** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/config.json` | `0fca72b6a70e...` | `0fca72b6a70e9ae812589199c661ef6be20cb9973f06353db125a1ec7264e589` | **VERIFIED (Immutable)** |
| **Tokenizer Vocab** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/vocab.json` | `db4d6119fedd...` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **VERIFIED (Immutable)** |
| **Tokenizer Merges** | `release/chakrview-v0.5.0/data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3...` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **VERIFIED (Immutable)** |
| **Cleaned Corpus v2 Manifest** | `experiments/batch2101_2140/CORPUS_V2_CLEANED_MANIFEST.json` | `e3db033c83f3...` | `e3db033c83f356f653b10d7eb00994da31d6e0738ab4177faddb133e5b8da380` | **VERIFIED (Immutable)** |
| **Split v2 Manifest** | `experiments/batch2021_2060/SPLIT_V2_MANIFEST.json` | `00748cbec81c...` | `00748cbec81c6eb52880f0b74756f2e672d65358143822dde466c9cec4124a31` | **VERIFIED (Immutable)** |
| **CPT Preregistration Manifest**| `experiments/batch2101_2140/CPT_PREREGISTRATION.json` | `b5307dd0807d...` | `b5307dd0807d2987f072925efd6659ece1a4f1972c08f8cc21c90a04ab02342d` | **VERIFIED (Frozen)** |
| **CPT Preflight Gate** | `experiments/batch2101_2140/CPT_PREFLIGHT_GATE.json` | `cfa1e567ce38...` | `cfa1e567ce3843ea09d430e470a174ece42daac28d67318d4386e1281ab8be53` | **VERIFIED (Frozen)** |

---

## 3. Architecture & Parameter Count Audit

The canonical checkpoint configuration was loaded and verified directly in PyTorch:
- `vocab_size`: `4096`
- `d_model` (width): `192`
- `n_layers`: `6`
- `n_heads`: `6`
- `hidden_dim` (FFN): `512`
- `max_seq_len`: `512`
- `tie_embeddings`: `True`
- **Total Parameters:** `3,443,136`
- **Trainable Parameters:** `3,443,136`

---

## 4. Preflight & Data Split Disjointness Verification

1. **Partition Allocation:**
   - **Train:** 782 documents, 147,476 tokens across 6 source families (`cpython_algorithms_heapq`, `gutenberg_tagore_gitanjali`, `ietf_rfc793_tcp`, `wikipedia_en_turing_machine`, `wikipedia_hi_chemistry`, `wikipedia_hi_premchand`).
   - **Validation:** 186 documents, 32,617 tokens across 3 source families (`cpython_algorithms_bisect`, `google_hinglish_top_dialogues`, `wikipedia_hi_ganit`).
   - **Test (Held-Out Benchmark):** 290 documents, 65,059 tokens across 4 source families (`gutenberg_loney_coordinate_geometry`, `wikipedia_en_information_theory`, `wikipedia_hi_sanskrit_sahitya`, `wikipedia_sa_bhagavad_gita`).
2. **Disjointness Guarantee:**
   - Source-family level: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
   - Document SHA-256 level: 0 duplicate hashes across partitions.
   - Text level: Whitespace 8-gram cross-split overlap is 0.
3. **Historical Pretraining Contamination Limitation:**
   - Canonical step 10,000 checkpoint was trained on synthetic algorithmic shards (`data/tokenized/stage_c/`).
   - Cleaned Corpus v2 uses natural text (Project Gutenberg, Wikipedia, RFC 793, CPython stdlib).
   - Python code standard library syntax shares common idiomatic constructs with earlier coding benchmarks, but no verbatim overlap with natural training data exists.

**Audit Conclusion:** Workstream A passed without deviation. The repository is in an authentic, verified, and uncorrupted state.
