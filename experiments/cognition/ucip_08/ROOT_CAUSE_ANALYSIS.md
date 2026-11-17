# UCIP-08: Root-Cause Decision Gate & Issue Classification

Before undertaking any recovery training or architectural changes, all discovered phenomena across UCIP-05, 06, and 07 were classified into formal diagnostic gates:

---

## Issue 1: Flawed Substring Arithmetic Matching in Evaluator
- **Classification**: **CONFIRMED EVALUATION DEFECT**
- **Evidence**: On probe `ARITH_054` (`12 - 5 = 7`), UCIP-07 generated `"You now have 17 coins in total."`. Evaluator 800 executed `if kw_clean in norm_out: success = True`. Because `'7' in '17'`, the wrong calculation was scored as `PASSED`.
- **Root Cause**: Bypassing regex word-boundary anchors via `or kw_clean in norm_out`.
- **Impact**: Inflated arithmetic performance on benchmark from true ~1-2% to 10.0%.
- **Correction**: Require strict regex `\b{kw}\b` and integer set comparison against target numeric values.
- **Validation**: Calibration suite confirms `CALIB_01` (`17` instead of `7`) is rejected.

---

## Issue 2: Single-Keyword Acceptance on Unrelated Word Salad
- **Classification**: **CONFIRMED EVALUATION DEFECT**
- **Evidence**: On `ENG_EXPL_021` (`What is photosynthesis?`), model generated `"Water eving liscose ner, water do to create oxyological eving slowly"`. Because `"water"` appeared, it was graded `PASSED`.
- **Root Cause**: Evaluator condition `if len(kw_hits) >= 1: success = True` across expansive keyword bags.
- **Impact**: Shifted benchmark passing threshold from genuine comprehension to emitting high-frequency unigrams.
- **Correction**: Enforce multi-keyword threshold ($\ge 2$), minimum response length ($\ge 4$ words), and automatic disqualification upon repetitive looping.
- **Validation**: Verified on `sealed_owner_reality_100.json`, where UCIP-07's score immediately dropped from 31.37% to its honest level of 12.0%.

---

## Issue 3: Template Permutation Memorization & Dataset Overlap
- **Classification**: **CONFIRMED DATA DEFECT**
- **Evidence**: 47 of the 80 core concepts in the 800-probe benchmark were exact string matches in `ucip06_train.jsonl`, multiplied 10x with artificial prompt prefixes (`"Please compute: "`, `"Hey! "`).
- **Root Cause**: Benchmark synthesis script drew prompts directly from SFT training files instead of held-out distributions.
- **Impact**: The model learned local prefix-dependent n-gram triggers rather than generalized reasoning. When evaluated on the un-prefixed base prompt `"If I have 5 apples..."`, it output `5 apples`, but with `"Please compute: "` it recalled `3 apples`.
- **Correction**: Seal a completely independent, held-out 100-probe benchmark (`sealed_owner_reality_100.json`, SHA-256: `7607eaf4...`) containing 0% overlap with training data.

---

## Issue 4: Severe Repetition Suppression Artifacts in CLI
- **Classification**: **CONFIRMED INFERENCE DEFECT**
- **Evidence**: On `"What does a CPU do?"`, the raw model produces `"A converts program instructions, copatience of a of a..."`. But when passed through `ConversationalChatCLI` with `repetition_penalty=1.15` and `no_repeat_ngram=3`, the repetition blocker suppressed `"of a"`, causing the model to emit bizarre lexical artifacts: `"A converts program instructions, copatience of a magnet and use lives the dogance to whitation."`
- **Root Cause**: Relying on post-hoc decoding penalties to patch over an undertrained language model whose token-level entropy is poorly calibrated.
- **Impact**: Degrades conversational fluency and introduces confusing hallucinations during owner testing.
- **Correction**: Calibrate decoding temperature and penalties to natural bounds (`rep_penalty=1.05`, `no_repeat_ngram=0`), and address repetition at the training loss level via label smoothing and unigram entropy regularization.

---

## Issue 5: Exposure Bias & Early Rollout Drift
- **Classification**: **CONFIRMED TRAINING DEFECT**
- **Evidence**: Layer-by-layer logit tracing demonstrated that while Step 0 (first token) has high accuracy ($p > 0.60$), Step 1 frequently drifts off-manifold. Under pure teacher forcing, the model is never trained on its own previous mistakes, causing irreversible divergence into attractor loops.
- **Correction**: Implement mixed-context / scheduled sampling augmentation and sequence-level anti-attractor loss during fine-tuning.
