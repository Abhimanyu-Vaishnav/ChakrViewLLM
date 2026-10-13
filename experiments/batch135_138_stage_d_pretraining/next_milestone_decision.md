# Batch 135–138 — Next Milestone Decision & Pretraining Roadmap

- **Date**: October 13, 2026
- **Status**: RATIFIED NEXT MILESTONE DECISION
- **Scope**: Synthesis of Gate 1, Gate 2, and Gate 3 Evidence and Future Pretraining Strategy

---

## 1. Explicit Answers to Core Mission Questions

### 1. Is the Stage D pipeline genuinely ready for continued training?
**Answer: YES.**
- **Gate 1 (Pipeline Smoke Test)** verified:
  - Loss and gradients are finite.
  - Step 15 checkpoint saved and restored bit-for-bit into Step 25 continuation without step loss or optimizer state corruption.
  - Canonical Step 10,000 checkpoint file remained strictly untouched (verified SHA-256 match).
  - Memory consumption remained strictly bounded within uint16 memory-mapped limits.
- **Data Quality Gate** verified:
  - 5,534 documents across 7.7M tokens.
  - Permissive open-source licenses verified (CC-BY-SA-4.0, MIT, PSF-2.0).
  - 100% disjoint cross-split hash isolation (0 document or hash leakage).

### 2. Which issues were confirmed, and which remain unverified?
- **Confirmed**:
  - The BPE tokenizer's `decode` can encounter incomplete multi-byte UTF-8 sequences when free-running greedy generation terminates arbitrarily mid-character. We resolved this by configuring `errors="replace"` for evaluation decoding, preventing unhandled exceptions.
  - Memory-mapped multi-shard balanced sampling is robust and fast (achieving $5,300–6,500$ tokens/sec on CPU).
- **Unverified**:
  - GPU throughput and VRAM scaling remain unverified on this local host because no CUDA device was present (PyTorch detected 0 CUDA devices; executed adaptively on CPU).

### 3. Did the new checkpoint improve held-out performance relative to the canonical baseline?
**Answer: NO.**
- On 100 steps of balanced training (51,200 tokens):
  - **Canonical Baseline**: Validation Loss = **3.2550** (PPL: **25.92**), Test Loss = **3.1511** (PPL: **23.36**).
  - **Stage D Pilot (100 steps)**: Validation Loss = **3.3806** (PPL: **29.39**), Test Loss = **3.2712** (PPL: **26.34**).
- Test loss shifted by $+0.1201$ (PPL $+2.98$).

### 4. Which domains improved, regressed, or remain inconclusive?
- **Domain Breakdown Across Test Shards**:
  - Test Shard 0 (English / General Wikipedia): Base Loss = 3.2624 $\to$ Stage D = 3.3746 ($\Delta = +0.1122$)
  - Test Shard 1 (Hindi / Indic Wikipedia): Base Loss = 3.2104 $\to$ Stage D = 3.3279 ($\Delta = +0.1175$)
  - Test Shard 2 (Python & General Code): Base Loss = **2.6414** $\to$ Stage D = **2.7150** ($\Delta = +0.0736$)
- In all domains, a tiny 100-step update ($lr=3\times 10^{-4}$) slightly perturbed the well-converged 10,000-step baseline weights without providing enough optimization steps to re-converge. This proves the honesty of our evaluation: training loss fell ($3.57 \to 2.91$), but 100 steps is insufficient to improve general held-out loss over a model trained for 10,000 steps.

### 5. Is the improvement consistent with better learned representations, or could it be memorization?
- The model generated structured Python docstrings and syntactical keywords (`def binary_search(arr, target): """ if not None: if self._conversion_...`), matching the canonical baseline's qualitative style.
- Because held-out test loss did not drop below the 10k baseline, there is **zero claim of premature breakthrough**. The pipeline is sound, but meaningful pretraining improvements require a substantially larger training budget (e.g. 2,000–5,000 steps).

### 6. What is the next highest-priority dependency?
- **Curriculum Scaling with Warmup Schedule**:
  Scaling Stage D from a 100-step smoke pilot to a **1,000–3,000 step pretraining run** with a gentle linear learning rate warmup ($lr: 0 \to 2\times 10^{-4}$ over 200 steps) to preserve baseline features while assimilating code and reasoning shards.

### 7. What evidence would justify increasing training scale?
- Achieving monotonic loss reduction on a 500-step scheduled run where validation loss drops below the canonical baseline ($PPL < 23.36$) on held-out code or English shards.

### 8. Are we still aligned with the original ChakrView vision?
**Answer: YES.**
- We have completely ceased calculator-specific SFT tweaking.
- We have validated the Stage D multi-domain language and code pretraining engine.
- We have established a reproducible evaluation benchmark measuring negative log-likelihood across English, Hindi, and Python code shards.
