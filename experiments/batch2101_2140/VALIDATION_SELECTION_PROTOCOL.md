# ChakrView Validation Selection Protocol — Batches 2101–2140
**Milestone:** Batches 2101–2140  
**Phase:** Phase D — Validation Selection Protocol  
**Date:** October 2026  
**Auditor:** Independent Preflight Release Agent  

---

## 1. Executive Summary & Verification of Validation Token Imbalance

In Batches 2061–2100, the CPT preparation audit reported that the validation partition in `SPLIT_V2_MANIFEST.json` was heavily skewed toward conversational Hinglish dialogues.

We audited `experiments/batch2021_2060/CORPUS_V2_MANIFEST.json` and verified this ratio directly from the manifest:

| Validation Source Family | Domain | Language | Document Count | Token Count | Token Share (%) | License |
|:---|:---|:---|:---:|:---:|:---:|:---|
| `google_hinglish_top_dialogues` | Hinglish | `hi-en` | 159 | 26,607 | **81.57%** | CC BY-SA 4.0 |
| `wikipedia_hi_ganit` | Math | `hi` | 19 | 4,741 | **14.54%** | CC BY-SA 4.0 |
| `cpython_algorithms_bisect` | Code | `code` | 8 | 1,269 | **3.89%** | PSF-2.0 |
| **Total Validation Split** | - | - | **186** | **32,617** | **100.00%** | - |

### The Critical Optimization Hazard:
Under raw token-weighted cross-entropy:
$$\mathcal{L}_{\text{val}}^{\text{weighted}} = 0.8157 \cdot \mathcal{L}_{\text{hinglish}} + 0.1454 \cdot \mathcal{L}_{\text{ganit}} + 0.0389 \cdot \mathcal{L}_{\text{bisect}}$$
If early stopping or checkpoint selection were guided by $\mathcal{L}_{\text{val}}^{\text{weighted}}$, the training controller would effectively optimize solely for Hinglish query continuation, ignoring code degradation or Hindi regression.

---

## 2. Specification of the Standalone Validation Evaluator

To eliminate this bias, we implemented a tested, standalone evaluator in `experiments/batch2101_2140/validation_evaluator.py`:
- **File:** `experiments/batch2101_2140/validation_evaluator.py`
- **Method:** `evaluate_validation_split(model, tokenizer, manifest_path, device="cpu")`
- **Isolation Guarantee:** Explicitly filters `d['split'] == 'validation'`. Zero access to test data is possible.

### Baseline Benchmark on Canonical Model (Step 10,000):
Evaluating the canonical pretraining checkpoint (`training_checkpoint_step_10000.pt`) across all 186 validation documents yields:
- **`cpython_algorithms_bisect` (Code):** $\text{CE} = 4.3686, \text{PPL} = 78.94$ (1,261 eval tokens)
- **`wikipedia_hi_ganit` (Hindi Math):** $\text{CE} = 4.0745, \text{PPL} = 58.82$ (4,722 eval tokens)
- **`google_hinglish_top_dialogues` (Hinglish):** $\text{CE} = 6.6526, \text{PPL} = 774.81$ (26,448 eval tokens)
- **Raw Token-Weighted Loss:** $\text{CE} = \mathbf{6.1884}, \text{PPL} = 487.08$
- **Equal-Weighted Macro-Source Loss:**
  $$\mathcal{L}_{\text{val}}^{\text{macro}} = \frac{4.3686 + 4.0745 + 6.6526}{3} = \mathbf{5.0319}, \text{PPL} = 153.22$$

---

## 3. Predeclared Checkpoint-Selection & Early-Stopping Metric

For all future CPT runs, checkpoint selection $\theta^*$ and early-stopping patience will be governed strictly by:
$$\mathcal{M}^* = \mathcal{L}_{\text{val}}^{\text{macro}} = \frac{1}{K} \sum_{k=1}^K \mathcal{L}_{\text{val}}^{(k)}$$
where $K=3$ represents the 3 independent validation source families (`google_hinglish`, `hi_ganit`, `cpython_bisect`).

### Non-Regression Constraint:
A checkpoint will only be eligible for selection if, in addition to minimizing $\mathcal{L}_{\text{val}}^{\text{macro}}$:
$$\mathcal{L}_{\text{bisect}} \le \mathcal{L}_{\text{bisect}}^{\text{canonical}} + 0.10 \quad \text{and} \quad \mathcal{L}_{\text{ganit}} \le \mathcal{L}_{\text{ganit}}^{\text{canonical}} + 0.10$$
This prevents Hinglish optimization from sacrificing code or Hindi mathematical capability.

---

## 4. Assessment of Validation Source Family Sufficiency

### Current Status: 3 Independent Source Families
- **Hinglish (`google_hinglish_top_dialogues`):** Covers code-switched conversational requests.
- **Hindi (`wikipedia_hi_ganit`):** Covers formal Devanagari mathematics exposition.
- **Code (`cpython_algorithms_bisect`):** Covers formal Python standard library algorithms.

### Sufficiency Assessment:
1. **Sufficiency for Controlled CPT Preflight:** The 3 source families provide adequate coverage across three orthogonal linguistic axes (`hi-en`, `hi`, `code`) to gate against catastrophic forgetting.
2. **Limitations for Broad Domain Generalization Claims:** Having only 3 validation sources is insufficient to claim omni-domain generalization (e.g., biological sciences, formal legal prose, or multi-turn conversational agents). For future major releases (v0.8+), additional independent validation sources in English technical prose and formal logic should be staged.
3. **Integrity Rule:** We reject manufacturing "artificial sources" by splitting existing files. True independent source expansion will be staged as distinct repositories in future milestones.
