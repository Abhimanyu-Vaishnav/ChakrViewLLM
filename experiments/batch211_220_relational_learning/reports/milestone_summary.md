# Batch 211–220: Milestone Summary Report
## Relational Knowledge Learning, Exact-Value Binding, and Generalization Validation

**Branch**: `dev/v0.6-v0.7-validation`  
**Delivered Milestone**: Batches 211–220  
**Device Context**: CPU (Intel/AMD x86_64, Windows, Torch 2.7.0+cpu, no CUDA device)  
**Canonical Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`  
**Canonical SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified Intact)  
**Canonical Unique Parameters**: 3,443,136 across 6 layers, $d_{model}=192$, vocabulary 4096, context window 512.

---

## 1. Accomplishments by Batch

### Batch 211 — Repository & Prior-Result Audit
- Verified canonical checkpoint hash, architecture invariants, and parameters.
- Re-executed prior runtime test suite (`tests/test_batch201_210_runtime_integration.py` passed 4/4).
- Audited the reported 8/8 integrated score from Batches 201–210: identified that 5/8 turns were resolved deterministically by `EvidenceExtractor`, 1 by session memory, 1 by neural core fallback, and only 1 required Trie pointer resolution.
- Authored `reports/initial_audit.md`.

### Batch 212 — Relational Learning Task Contract
- Authored machine-readable `reports/task_contract.md` defining JSON schema for relational tuples across Mathematics, Coding, and Science.
- Formalized 6 discrimination error categories:
  1. Correct entity and correct relation (Target Success).
  2. Correct entity, wrong relation (Semantic Confusion).
  3. Correct relation, wrong entity (Entity Intrusion).
  4. Numerically plausible but unsupported answer (Hallucination/Distractor Copy).
  5. Missing fact (Ungrounded / Safe Abstention).
  6. Contradictory evidence (Conflict Rejection).

### Batch 213 — Deterministic Synthetic Relational Corpus (10,000 Records)
- Implemented `data/generate_relational_corpus.py` producing 10,000 synthetic records with traceable random seeds and template families.
- Formats include single-fact, multi-fact with distractors, and adversarial unanswerable cases in English and Hindi (Devanagari).

### Batch 214 — Leakage-Resistant Split & Audit
- Partitioned by 14 underlying entity clusters (Train: 9 clusters / 6,360 records, Val: 2 clusters / 1,097 records, Test: 3 clusters / 2,543 records).
- Audited train vs test overlap: verified 0% entity leakage (zero test entity tokens appear in training contexts).
- Authored `reports/data_and_leakage_audit.md` and saved `data/relational_corpus_manifest.json`.

### Batch 215 — Controlled Baselines Establishment
- Evaluated 6 frozen baseline arms over 100 stratified held-out test examples:
  1. Canonical Vocab-Only Generation: 1.0% (1/100)
  2. Standard Pointer (Unconstrained Greedy): 7.0% (7/100)
  3. Query-Aware Pointer (Unconstrained Greedy): 7.0% (7/100)
  4. TokenTrie Constrained Candidate Selection: 32.0% (32/100)
  5. Lexical BM25 Document Retrieval: 100.0% (100/100 top-1 recall)
  6. Deterministic Slot Extractor: 48.0% (48/100 exact match)
- Saved `reports/baseline_results.json`.

### Batch 216 — Isolated Relational Pointer Head Training
- Trained an isolated `ChakrPointerHead` (`d_model=192`, `d_proj=64`, query-aware) on factual tuples over 3 epochs (200 records).
- Loss converged from 9.5091 to 4.3136 in ~7.8 seconds on CPU.
- Preserved all canonical model weights without modification.
- Evaluated trained head on test split:
  - Trained Pointer Greedy EM: 7.0% (7/100)
  - Trained Pointer + TokenTrie EM: **47.0% (47/100)** (+15.0% absolute improvement over baseline 32.0%).
- Saved checkpoint `checkpoints/relational_pointer_head_step1000.pt` and `reports/training_results.json`.

### Batch 217 & 218 — Exact-Value Binding & Granular Generalization Slices
- Evaluated 5 targeted generalization slices (225 total examples):
  1. `in_distribution_seen`: 78.0% (39/50) — 7 wrong distractor copies.
  2. `held_out_entity`: 48.0% (24/50) — 12 wrong distractor copies.
  3. `numeric_distractors`: 62.0% (31/50) — 14 wrong distractor copies.
  4. `multilingual_devanagari`: 38.0% (19/50) — 15 wrong distractor copies.
  5. `unanswerable_adversarial`: 100.0% (25/25) — 0 wrong distractor copies (100% correct abstention).
- Saved `reports/generalization_results.json`.

### Batch 219 — Integrated Runtime Regression & Robustness
- Verified `GroundedSessionAdapter` with the new relational pointer head checkpoint across 8 edge-case and regression scenarios.
- Added context window safety truncation in `_generate_neural` to prevent sequence length overflows.
- Result: 8/8 passed (100.0%).
- Saved `reports/runtime_regression_results.json`.

### Batch 220 — Consolidation, Manifest & Safe Delivery
- Created comprehensive test suite `tests/test_batch211_220_relational_learning.py`.
- Verified all regression test suites pass.
- Produced `experiment_manifest.json` recording hashes, file sizes, and configurations.
- Preserved protected user changes in `chakrview/runtime/session.py`.

---

## 2. Key Metrics Summary Table

| Evaluation Arm / Slice | Total Evaluated | Exact Match / Pass Count | Accuracy (%) | Component Origin |
|---|---|---|---|---|
| **Baselines (Held-Out Test)** | | | | |
| Canonical Vocab-Only | 100 | 1 | 1.0% | Neural Core |
| Baseline Pointer Greedy | 100 | 7 | 7.0% | Neural Pointer |
| Baseline TokenTrie Constrained | 100 | 32 | 32.0% | Constrained Trie |
| BM25 Lexical Retrieval | 100 | 100 | 100.0% | Lexical BM25 |
| Deterministic Extractor | 100 | 48 | 48.0% | Regex / Slot Extractor |
| **Trained Pointer Head** | | | | |
| Trained Pointer + TokenTrie | 100 | 47 | 47.0% | Constrained Trie |
| **Generalization Slices** | | | | |
| In-Distribution Seen | 50 | 39 | 78.0% | Constrained Trie |
| Held-Out Entity (0% Leakage) | 50 | 24 | 48.0% | Constrained Trie |
| Numeric Distractors (Adversarial) | 50 | 31 | 62.0% | Constrained Trie |
| Multilingual (Devanagari/Hindi) | 50 | 19 | 38.0% | Constrained Trie |
| Adversarial Unanswerable | 25 | 25 | 100.0% | Deterministic Abstention |
| **Runtime Integration** | 8 | 8 | 100.0% | Multi-Tier Integrated |

---

## 3. Recommended Next Milestones
1. **Subword BPE Normalization for Indic Languages**: Address the 38.0% drop on Hindi by aligning subword token boundaries before TokenTrie generation.
2. **Relational Schema Biasing in Pointer Projections**: Inject explicit relation type embeddings into the pointer query projection to reduce the 28.0% numeric distractor confusion rate.
3. **Multi-Hop Compositional Evidence Linking**: Extend the pointer mechanism from single-hop slot retrieval to multi-fact chaining across documents.
