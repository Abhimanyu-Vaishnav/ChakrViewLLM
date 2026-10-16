# Previous Experiment Audit

## Batches 1421–1460: Verification of Batch 1381–1420 Adaptation Harness and Metrics

### 1. Verification of Code and Execution Integrity

We audited [run_domain_adaptation.py](file:///D:/Project/ChakrView/experiments/batch1381_1420/run_domain_adaptation.py) and [run_lm_evaluation.py](file:///D:/Project/ChakrView/experiments/batch1341_1380/run_lm_evaluation.py):

| Dimension | Verification Item | Audit Finding | Status |
|---|---|---|---|
| **Tokenizer Identity** | Same tokenizer used across baseline and candidate | Both use canonical vocab 4096 from `release/chakrview-v0.5.0/data/experiments/vocab_4096`. | **CONFIRMED** |
| **Model Configuration** | Identical structural hyperparameters | Both use `ModelConfig(vocab_size=4096, d_model=192, n_layers=6, n_heads=6, hidden_dim=512, max_seq_len=512, tie_embeddings=True)`. | **CONFIRMED** |
| **Weight Initialization** | Candidate initialized from canonical weights | Verified: Candidate initialized by loading `state_dict` from `training_checkpoint_step_10000.pt`. | **CONFIRMED** |
| **Data Partitioning** | Train/Val/Test isolation | 21 adaptation train docs, 4 dev val docs, 27 test docs. Test docs were strictly uncalled during optimizer steps. | **CONFIRMED** |
| **Loss & PPL Calculation** | Aggregate NLL divided by total target tokens | Computed globally as $\exp(\sum \text{NLL} / \sum N_{\text{targets}})$, avoiding arithmetic batch averaging. | **CONFIRMED** |
| **Target Population** | Identical targets evaluated | Both baseline and candidate evaluated across identical 3,456 targets ($N-1$ per document). | **CONFIRMED** |

---

### 2. Metric Reproduction on Old Held-Out Test Set (27 Documents)

Re-evaluating the saved candidate checkpoint against the baseline on the 27 old test documents yields exact reproduction:

- **Baseline Step 10,000:**
  - Loss: **5.0630** nats
  - Perplexity: **158.07**
  - Top-1 Accuracy: **12.85%**
  - Top-10 Accuracy: **40.39%**
- **Candidate Step 10,080:**
  - Loss: **4.3371** nats
  - Perplexity: **76.49**
  - Top-1 Accuracy: **23.52%**
  - Top-10 Accuracy: **49.33%**
- **Discrepancies:** **0.0000** (Zero discrepancy).

---

### 3. Classification of the Old Test Set

> **Important Methodological Note:**
> Because the 27 test documents from `experiments/batch1141_1180/data/corpus/` were evaluated in both Batch 1341–1380 and Batch 1381–1420 to benchmark candidate progress, they are formally classified as a **Previously Inspected In-Domain Benchmark**.
> To make an unassailable scientific claim of generalization, an **independent, uninspected evaluation corpus** must be constructed and evaluated.
