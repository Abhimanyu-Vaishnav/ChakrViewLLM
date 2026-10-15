# Starting Point Audit: Batches 581–620
## Neural Argument Extraction, Pointer Learning & Reliable Tool Grounding

**Audit Date:** 2026-10-15  
**Repository:** `D:\Project\ChakrView`  
**Current Branch:** `dev/v0.6-v0.7-validation`  
**Head Commit:** `1cc9a1f` — `feat(batch541-580): train structured intent candidate, validate zero-leakage contract, and evaluate tool dispatch`  
**Previous Milestone:** `1cc9a1f` (Verified clean lineage from Batches 541–580)

---

### 1. Verification of Prior Artifacts & Reproducibility Audit

#### 1.1 Checkpoint SHA-256 Verifications
- **Canonical Step 10,000 Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (**Verified Immutable**)
- **Candidate 501–540 (Compositional Rule):** `experiments/batch501_540/checkpoints/candidate_step200_compositional_rule.pt`
  - SHA-256: `b18e69ebf4f0b396465af3c323f38bcfa84d5a38eabbd4a4220a10d6bfafee98` (**Verified Match**)
- **Candidate 541–580 (Intent Understanding):** `experiments/batch541_580/checkpoints/candidate_step250_intent_understanding.pt`
  - SHA-256: `9c3204906e8e94f5e80c5db9f6ffb0fa6903c6fe3159206fb164e44d52b8357e` (**Verified Match**)

#### 1.2 Reproduction of 59-Item Benchmark on `candidate_step250_intent_understanding.pt`
We re-executed `experiments/batch541_580/run_eval_clean.py` under identical test conditions:
- **Total Test Items:** 59
- **Valid JSON Rate:** 52/59 (**88.1%**, exact reproduction)
- **Overall Intent Accuracy:** 22/59 (**37.3%**, exact reproduction)
  - `calculator`: 19/29 (65.5%)
  - `knowledge_lookup`: 1/4 (25.0%)
  - `concept_definition`: 1/5 (20.0%)
  - `direct_chat`: 1/5 (20.0%)
  - `text_utility`: 0/6 (0.0%)
  - `clarification_needed`: 0/5 (0.0%)
  - `unsupported`: 0/5 (0.0%)
- **Exact Argument Copy Accuracy:** 2/59 (**3.4%**, exact reproduction)
- **Clean True EOS Termination:** 52/59 (**88.1%**, exact reproduction)

---

### 2. Diagnosis of Failure Root Causes

Why does the 3.4M autoregressive model achieve **65.5% on calculator intent** but only **3.4% on exact argument extraction**?

1. **Associative Token Prior vs Dynamic Register:**
   - In standard autoregressive next-token prediction, the neural core memorizes frequent sequences. When prompted with `User: Evaluate expression: 94 - 62\nIntent:`, the model outputs `{"schema_version":"v1.0","intent":"calculator","tool":"calculator","args":{"expression":"22 * 15"},"status":"ready"}`.
   - The expression `22 * 15` was heavily present during SFT rehearsal, and the small 3.4M model defaults to the associative prior rather than copying `94 - 62` from the prompt.
2. **Lack of an Explicit Span / Pointer Mechanism:**
   - The user query already contains the exact string to be evaluated or transformed (`94 - 62` or `'indigenous neural core'`).
   - Forcing a 3.4M parameter model to reconstruct this character-by-character through 30+ autoregressive decoding steps creates a compounding probability bottleneck ($P(\text{all tokens correct}) = \prod p_t$).
3. **Existing Repository Component Opportunities:**
   - In `chakrview/brain/model.py`, `ChakrMicro.forward()` already supports `return_hidden_states=True`, returning representations `h` of shape `[B, T, 192]`.
   - In `chakrview/runtime/pointer_mechanism.py`, `ChakrPointerHead` provides a pre-norm attention dot-product over input positions.
   - By predicting the **span boundary (start_token, end_token)** or extracting the exact substring directly via neural attention / span prediction, the model does not need to guess digits.

---

### 3. Plan for Batches 581–620

1. **Work Package B:** Formulate a formal **Span & Intent Contract** (`schema_version: "v2.0"`) supporting span coordinates `[start_char, end_char]` or `[start_tok, end_tok]` mapping to original user input.
2. **Work Package C:** Build a clean, leakage-free dataset generator producing token-level and character-level span annotations for all supported intents.
3. **Work Package D:** Implement and compare:
   - Architecture A: Pure Autoregressive baseline.
   - Architecture B: Multi-Head Span & Intent Classifier (neural span prediction head attached to `ChakrMicro` hidden states).
   - Architecture C: Pointer-grounded structured generator.
4. **Work Package E & F:** Train and evaluate on frozen held-out test suite with exact argument copy metrics.
5. **Work Package G:** Safely integrate into `NeuralIntentRouter` with transparent provenance.
