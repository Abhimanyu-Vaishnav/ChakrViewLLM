# CV-GENESIS-NR-008 Data Quality and Balance Report

## 1. Executive Summary

Mission **CV-GENESIS-NR-008 Phase 3** addressed the primary root cause of low factual accuracy in Hindi and weak multi-turn coherence identified in NR-006: **severe linguistic and domain imbalance** (historical training corpora contained only 11.4% Hindi data).

A newly curated, cryptographically validated, leakage-filtered SFT mixture was generated at `experiments/genesis_balanced_sft_008/data/`.

---

## 2. Dataset Synthesis & Composition

- **Total Curated Examples:** 4,401
- **Train Split (85%):** 3,740 examples
- **Validation Split (10%):** 440 examples
- **Test Split (5%):** 221 examples
- **Target Vocabulary:** 4,096 BPE (`db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)

### Linguistic Distribution (Train Split):
| Language | Count | Historical Share (NR-005) | **NR-008 Balanced Share** | Strategic Justification |
| :--- | :--- | :--- | :--- | :--- |
| **English (`en`)** | 1,881 | 76.0% | **50.3%** | Anchors technical explanations, coding syntax, and logical QA. |
| **Devanagari Hindi (`hi`)** | 1,558 | 11.4% | **41.7%** | **~4x expansion**. Ingested high-quality factual explanations and domain knowledge from UCIP14R. |
| **Hinglish (`hinglish`)** | 301 | 12.6% | **8.0%** | Retains natural colloquial Hindi-English conversational code-switching. |

---

## 3. Quality Assurance and Anti-Contamination Gates

1. **Blind Benchmark Anti-Leakage Filter:**
   - 300 sealed blind evaluation prompt hashes from NR-006 were loaded into an in-memory hash set.
   - Any candidate training example matching a blind prompt under normalized hash matching was strictly filtered out.
   - **Contamination Rate:** **0.0%**.
2. **Exact Deduplication:**
   - Evaluated normalized prompt hashes to prevent prompt overfitting.
   - 0 duplicate prompts exist in the final mixture.
3. **Format Standard:**
   - All examples follow canonical `User: <input>\nAssistant: <target>` sequencing with strict supervision of the assistant target and `<eos>` delimiter.
