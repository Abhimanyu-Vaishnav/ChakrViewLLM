# Batch 102: Synthesis, Reproducibility Manifest, and Engineering Handoff

**Sprint Date**: October 12, 2026  
**Repository Branch**: `dev/v0.6-v0.7-validation`  
**Starting HEAD**: `c2f191ae9261a1ea3ce1824d8d49daedc244354f`  
**Execution Environment**: Windows 11, Python 3.14.7, PyTorch 2.14.0+cpu, pytest 9.1.1  

---

## 1. Verified Scientific Observations

1. **Tokenizer Asymmetry & Subword Mergers**:
   - The indigenous BPE tokenizer merges single-digit `1` with the label colon-space prefix `": 1"` into a single token ID `2665`.
   - Digits 0, 2..9 are encoded as two tokens: `": "` (token `324`) followed by the digit token ($51..60$).
   - In tool argument positions, single digits are similarly merged with delimiters (`"1, "` as token `628`, `"1}"` as token `585`, `"8"` as token `59` followed by `128` for `}`).
   - Dynamic token-offset resolution is required to trace operands across prompts.

2. **Residual Stream Preserves Local Operand Identity (100.0% Decodability)**:
   - At token positions `pos_a` and `pos_b`, linear probes trained on 64 training pairs decode held-out test operand identities with **100.0% accuracy** (against a chance baseline of 11.1% and a shuffled-label control of 11.1%–22.2%).
   - This holds across all internal network depths: Embedding (Layer 0), intermediate blocks (Layer 3), and post-final RMSNorm (Layer 7).
   - Therefore, the model's failure to generate correct arguments is **not** caused by operand erasure or loss of local representation.

3. **Causal Information-Routing Attenuates at Sequence Tail**:
   - At the `prompt_end` generation boundary, decodability drops from 100% to **22.2%–55.6%**.
   - SFT (both Condition B and Condition C) increases the probability assigned to the ground-truth operand tokens at the argument generation position from $<1\%$ to $\sim 8.7\% - 9.8\%$, elevating them into the **top-6 candidates**.
   - However, the mode attractor token (`"1, "` for A, `"8"` for B) maintains an invariant unigram prior of $15.6\% - 17.2\%$.
   - Under greedy argmax decoding, this leads to **100% constant-call collapse** across both natural language prompting (`{"a": 1, "b": 1}`) and explicit slot prompting (`{"a": 1, "b": 8}`).

---

## 2. Experimental Results Summary

| Model / Checkpoint | Valid Syntax | Collapse Rate | Most Common Call | Probe Acc (`pos_a`) | Probe Acc (`prompt_end`) | $P(\text{tgt}) / P(\text{gen})$ (Arg A) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Step 10k** | 0.0% | 51.6%–72.2% | Wiki text table | 100.0% | 22.2%–50.0% | 0.021 (Rank 76) |
| **Condition B (NL Tool)** | 100.0% | 100.0% | `{"a": 1, "b": 1}` | 100.0% | 38.9%–44.4% | 0.348 (Rank 5.5) |
| **Condition C (Slot-Binding)**| 100.0% | 100.0% | `{"a": 1, "b": 8}` | 100.0% | 22.2%–33.3% | 0.556 (Rank 5.9) |

---

## 3. Deliverables & Artifacts Produced

1. **Representation Probe Diagnostic** (`experiments/batch99_102_representation_audit/representation_probe.py`):
   - Dynamic BPE-aware token alignment.
   - Leakage-free probe splits (0 unordered pair overlap).
   - Shuffled-label controls and confusion matrices.
   - Summary results persisted to `representation_probe_summary.json`.

2. **Controlled Evaluation Suite** (`experiments/batch99_102_representation_audit/controlled_diagnosis.py`):
   - Computes teacher-forced argument token probability gaps ($P(\text{target})$ vs $P(\text{gen})$).
   - Evaluates syntax, execution, and spurious clarification handling.
   - Summary results persisted to `controlled_diagnosis_summary.json`.

3. **Interactive Token Diagnostics Workbench** (`experiments/batch99_102_representation_audit/workbench/app.py`):
   - Interactive token-chip UI with instant token inspector drawer.
   - Top-5 candidate distributions and log-probabilities.
   - Divergence point highlighting.
   - Offline report reloading and legacy backward compatibility.
   - Preserved CLI interface (`inspect`, `diagnose`, `list-reports`, `serve`).

4. **Regression Test Suite** (`tests/test_batch99_102_operand_diagnostics.py`):
   - 10 new tests covering BPE mergers, probe split isolation, deterministic seeding, candidate distributions, report persistence, and security allowlist checks.
   - Combined test suite (`test_batch87_90` + `test_batch91_94` + `test_batch95_98` + `test_batch99_102`): **30 passed in 4.03s**.

5. **Technical Documentation**:
   - `docs/batch99_representation_audit.md`
   - `docs/batch100_controlled_diagnostics.md`
   - `docs/batch101_token_inspector.md`
   - `docs/batch102_synthesis_and_handoff.md`

---

## 4. Reproducibility Manifest

### Checkpoint Hashes
- **Canonical Step 10,000**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Condition B (NL Tool)**: `65eb913bbed9359a343b6efb46cfa3f1b4c9197931327151e3604f3261a8ef11`
- **Condition C (Slot-Binding)**: `d7702580c5e586ed4ffa3a6498d4f35520886d92e8f3b581ab8df1c26d4cdaa2`

### Reproduction Commands
```bash
# 1. Run Representation Probe Audit
python experiments/batch99_102_representation_audit/representation_probe.py

# 2. Run Controlled Diagnosis Experiment
python experiments/batch99_102_representation_audit/controlled_diagnosis.py

# 3. Run Focused Test Suites
python -m pytest tests/test_batch91_94_tool_calling_comparator.py \
                 tests/test_batch95_98_slot_binding_diagnostics.py \
                 tests/test_batch99_102_operand_diagnostics.py -v

# 4. Launch Interactive Token Diagnostics Workbench
python experiments/batch99_102_representation_audit/workbench/app.py serve --port 8501
```

---

## 5. Separation of Findings

- **Verified Observations**:
  - The model retains full linear decodability of operand identity at its source token positions ($100\%$).
  - Decodability attenuates as information travels through causal self-attention to the prompt boundary ($22\%–55\%$).
  - Target argument probabilities rise under SFT into the top-6 candidate tokens ($\sim 9\%$), but are overcome by the static unigram prior ($\sim 16\%$) at argmax decoding.
- **Hypotheses (Unproven)**:
  - An explicit cross-attention pointer mechanism or copy-head objective would overcome the static unigram bias without destabilizing the decoder.
- **Recommended Next Experiment**:
  - Implement a bounded **Direct Argument Loss Weighting or Dynamic Copy Projection** that specifically penalizes static unigram collapse at the argument slot positions while preserving tool syntax macros.
