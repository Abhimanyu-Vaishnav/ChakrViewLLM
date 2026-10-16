# Dataset Provenance & Split Architecture (Batches 1621–1660)

## 1. Frozen Artifacts & Cryptographic Verification

All dataset partitions were constructed, verified, and sealed before model evaluation:
- **Manifest:** `experiments/batch1621_1660/data/DATASET_MANIFEST.json`
- **Training Partition:** `experiments/batch1621_1660/data/train.jsonl` (120 rule-governed examples)
- **Validation Partition:** `experiments/batch1621_1660/data/val.jsonl` (10 in-distribution examples)
- **Held-Out Test Suite:** `experiments/batch1621_1660/data/test_held_out.json` (150 held-out tasks)
- **Held-Out Test SHA-256:** `4def0f16cdfe2f41eaec46ed14f98be395caa6505047675c15becf20f40258d1`

---

## 2. Partition Rules & Contamination Prevention

| Partition | Scope & Size | Generation Rule & Invariants |
|:---|:---:|:---|
| **Training (`train.jsonl`)** | 120 samples | Teaches 6 specific rules: (1) Service ports (`auth` $\rightarrow$ `8001`, etc.), (2) Tier slot allocation (`nano` $\rightarrow$ `2`, etc.), (3) Forwarding Hop 1 (`Cluster X connects to bridge Y`), (4) Forwarding Hop 2 (`Bridge Y relays to archive Z`), (5) Arithmetic doubles for integers 1–20, (6) Standard Python function headers. |
| **Validation (`val.jsonl`)** | 10 samples | In-distribution generalization check with held-out service names (`registry`, `metrics`) and arithmetic integers (22, 24, 26). |
| **Held-Out Test (`test_held_out.json`)** | 150 tasks | **Zero lexical overlap** with training set: (1) Unseen distributed cluster syntax, (2) Arithmetic doubles on integers 25–54 (untrained range), (3) Unseen code completion headers, (4) Transitive compositional reasoning ($X \rightarrow Z$), (5) Counterfactual unseen service queries. |

---

## 3. Contamination Controls
- **No Direct Answer Leakage:** Unseen composition queries ($X \rightarrow Z$) never appear as complete sentences anywhere in the training or validation sets.
- **Arithmetic Domain Shift:** The training data only demonstrates doubles up to 20; the test suite evaluates 25 to 54 to distinguish true algebraic rule generalization from rote memory.
