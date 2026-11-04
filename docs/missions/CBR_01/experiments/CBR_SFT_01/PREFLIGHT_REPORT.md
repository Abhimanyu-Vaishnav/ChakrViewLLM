# CBR-SFT-01 — PREFLIGHT VERIFICATION REPORT

**Experiment ID:** `CBR-SFT-01`  
**Parent Model:** Canonical Step 10,000 (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`)  
**Parent Branch:** `exp/cbr-01-conversational-breakthrough`  
**Execution Date:** 2026-11-04  
**Preflight Status:** **`PASS`** (All 5 Mandatory Preflight Verification Gates Cleared)  

---

## 1. Protected Artifacts & Hash Invariance
All protected baseline files remain 100% bit-exact:
- `chakrview/brain/model.py`: `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` (MATCH)
- `chakrview/runtime/session.py`: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` (MATCH)
- `training_checkpoint_step_10000.pt`: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (MATCH)
- `.gitignore`: `700f3d8b6d8c5c216884f1808f5d5c7da17282ce4d69f6e5d563f2ee49e4fda6` (MATCH)
- Extraction decision doc: `e5b433adebb53b45e8250f07cae38282e1a06839bed15fa788d16d4db5192f00` (MATCH)
- `experiments/overnight_runs/`: Untouched and unaccessed.

---

## 2. Dataset Verification (Exp34 v3 Splits)
- **Train Split:** `data/sft/exp34_candidate_v3/splits/exp34_train_split_900.jsonl` (SHA-256: `d42c56d1c77a8d6cb8b75216909cc34e49f4346724b6aa276ba4b1d298f8a104`)
- **Val Split:** `data/sft/exp34_candidate_v3/splits/exp34_val_split_100.jsonl` (SHA-256: `98803283f26788f26940027c7a467e6101b1d85616b9d2da861857f3387e56ca`)
- **Record Integrity:** 900 train records, 100 val records, exactly 0 ID overlap, exactly 0 prompt overlap.
- **Response Diversity:** 100.0% unique response strings (900 unique out of 900).
- **Category Balance:** Perfectly stratified across 10 categories (90 records per category).
- **M3 Test Contamination:** Zero prompt overlap with the 60-task M3 evaluation suite.
- **Trust Gate & Permissions:** All records validated under `VERIFIED_TRAINING` review status.

---

## 3. Tokenizer & Prompt Contract Parity
- **BOS/EOS IDs:** `bos_id=0`, `eos_id=1`, `pad_id=2`, `ignore_index=-100`.
- **Delimiter Contract:** `strip_prompt_delimiters=False` confirmed as canonical to ensure whitespace parity between training sequence headers (`Assistant:`) and runtime session prefixes.
- **Vocabulary:** 4,096 BPE tokens verified.

---

## 4. Training Mechanics & Causal Alignment
Synthetic tiny-batch gradient check verified:
1. Target sequence is `[-100]*len(p) + response_tokens + [EOS]`.
2. Input sequence is `[BOS] + p_toks + response_tokens`.
3. Prompt tokens produce zero loss and zero target gradients.
4. Active target count equals $len(\text{response}) + 1$ (including EOS).
5. Padding tokens with `ignore_index=-100` produce zero loss.
6. Backward pass yields finite gradients across all parameter tensors with zero NaN/Inf values.

---

## 5. Baseline Evaluation Reproducibility
- Re-executed the 60-task M3 conversational benchmark against the canonical Step 10,000 checkpoint on native CPU.
- **Result:** Exactly **0/60 passed (0.00%)**, bit-for-bit replicating the canonical evaluation report.
- The 0.0% baseline floor is fully verified and stable.

