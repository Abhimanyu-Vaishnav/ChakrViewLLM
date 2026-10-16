# CHAKRVIEW — Batches 1981–2020
## Neural Evaluation Readiness Gate Decision

### Mission Mandate

Determine the formal training-readiness status for ChakrView's neural core based on:
1. Independent reproduction of statistical evaluation results.
2. Source-cluster uncertainty and group-level statistical significance.
3. Corpus quality audit, split concentration analysis, and language representation.
4. Provenance and legal licensing verification.

Pre-declared decision options:
- `READY_FOR_CONTROLLED_CPT`
- `READY_WITH_LIMITATIONS`
- `NOT_READY`

---

### Formal Verdict

**DECISION: READY_WITH_LIMITATIONS**

ChakrView's statistical and measurement framework is fully verified, reproducible, and internally consistent. The legal provenance of all candidate sources is approved without exception. However, **the staged natural corpus cannot be used for training or evaluation in its current split allocation due to extreme test-set source concentration (98.6% of test tokens from a single source) and domain gaps in Hinglish and pure Sanskrit.**

---

### 1. Readiness Gate Audit Matrix

| Evaluation Dimension | Standard Required | Audit Finding | Verdict |
| :--- | :--- | :--- | :--- |
| **Statistical Reproduction** | Reported loss deltas and bootstrap CIs must reproduce bit-exact from raw predictions. | Token-weighted (-0.2012) and macro document (-0.3088) reproduced exactly. Point estimates lie strictly within intervals. | **PASS** |
| **Source-Clustered Uncertainty** | Evaluate clustering across 18 source groups. | 150s CPT maintains significant negative interval (`[-0.4160, -0.0506]`). 300s CPT spans zero (`[-0.2714, +0.1140]`). | **PASS** |
| **Provenance & Licensing** | All ingested sources must have verified public domain or open licenses. | All 8 sources verified under Public Domain, PSF-2.0, or CC BY-SA 4.0. Recorded SHA256 hashes immutable. | **PASS** |
| **Split Source Isolation** | No shared source files across train/validation/test splits. | Verified. Every source is assigned strictly to a single split. Zero cross-split file leakage. | **PASS** |
| **Test Split Balance & Representativeness** | Test set must provide balanced, multi-domain evaluation without single-source dominance. | **FAILED (Concentration Bottleneck):** S. L. Loney Coordinate Geometry accounts for **99.5% of test documents and 98.6% of test tokens**. | **LIMITATION** |
| **Language Coverage Quality** | Meaningful coverage of Hindi, Sanskrit, Hinglish, Code, and Math. | High-quality Hindi, English, Code, and Math. **Gaps identified:** Sanskrit source is actually Hindi commentary; natural Hinglish absent. | **LIMITATION** |

---

### 2. Mandatory Remediation Directives Prior to Launching Pretraining

To transition from `READY_WITH_LIMITATIONS` to full `READY_FOR_CONTROLLED_CPT`:

1. **Rebalance the Natural Test Set:**
   - Sub-sample `gutenberg_loney_coordinate_geometry` to at most 150–200 documents (~35,000 tokens) in the test split.
   - Introduce held-out source groups from English prose, Hindi literature, technical systems, and Python code into the test split to ensure multi-domain evaluation balance.
2. **Expand the Validation Split:**
   - Expand validation data beyond the current 26 documents (5,948 tokens) to at least 80–100 documents (~15,000–20,000 tokens) across multiple independent sources.
3. **Ingest Authentic Sanskrit & Natural Hinglish:**
   - Add authentic Sanskrit verses (e.g. classical Panchatantra or Bhagavad Gita Devanagari text) to replace the Hindi Wikipedia article currently mislabeled as Sanskrit.
   - Ingest genuine conversational Hinglish from verified open-source datasets (e.g. L3Cube MahaHinglish) under CC BY-SA.
4. **Preserve Architectural Authority Constraints:**
   - Under no circumstances shall the neural core be granted autonomous tool execution authority or bypass the `SafeCalculator` and `GovernedToolRouter`.
