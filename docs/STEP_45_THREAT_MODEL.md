# Step 45 Threat Model: Neural Inference Validation & Generation Quality Layer

**Date:** 2026-09-30  
**Status:** RATIFIED & ACTIVE  
**Scope:** Neural Inference Pipeline, Sampling Safety, Generation Quality Controls, Checkpoint Identity Verification, Deterministic Reproducibility

---

## 1. Threat Landscape Overview

Step 45 establishes the validation, generation quality, and model identity layer on top of the Step 44 neural inference pipeline.

Because this layer will become the stable evaluation interface for trained ChakrView checkpoints, it must guarantee strict numerical safety, fail-closed handling of malformed inputs or corrupted checkpoints, deterministic output reproducibility, and airtight security boundaries.

---

## 2. Threat Analysis Matrix

| Threat ID | Threat Category | Severity | Attack Surface | Mitigation Summary |
|:----------|:----------------|:--------:|:---------------|:-------------------|
| **GQT-01** | Checkpoint Incompatibility / Corruption | CRITICAL | Checkpoint file loading | `validate_checkpoint_compatibility()` verifies architecture name, param count, layer count, and shapes; fails closed on mismatch. |
| **GQT-02** | Silent Weight Mutation During Eval | CRITICAL | Model eval loop | Pre/post-flight SHA-256 weight hash check on every execution; `torch.no_grad()` enforcement. |
| **GQT-03** | Pathological Logits (NaN / Inf) | HIGH | Model output layer | Assert `torch.isfinite(logits).all()`; fail closed with `PathologicalLogitsError`. |
| **GQT-04** | Degenerate Probability Distributions | HIGH | Sampler softmax / multinomial | Check probability sums $> 0$ and no NaNs; raise `SamplingProbabilityError` on degenerate distribution. |
| **GQT-05** | Invalid Sampling Hyperparameters | MEDIUM | `SamplingConfig` initialization | Strict boundary checks: $T \ge 0$, top-k $\ge 0$, $0 < \text{top\_p} \le 1.0$, $1.0 \le \text{repetition\_penalty} \le 10.0$. |
| **GQT-06** | Premature EOS Termination | MEDIUM | Autoregressive loop | Enforce `min_new_tokens`: suppress EOS token from stopping generation before minimum requested length. |
| **GQT-07** | Runaway Generation Loop | MEDIUM | Autoregressive loop | Hard upper bound ceiling `max_new_tokens`, sequence limit guard ($\le 512$). |
| **GQT-08** | Non-Deterministic Stochastic Divergence | MEDIUM | Seeded sampling | Step-offset deterministic RNG seeding: `torch.Generator.manual_seed((seed + step) % (2**31 - 1))`. |
| **GQT-09** | Context Window Overflow ($T > 512$) | HIGH | Long prompt + generation | Hard ceiling check: prompt + max_new_tokens $\le 512$; fail closed unless explicit truncation requested. |
| **GQT-10** | Secret / Credential Leakage in Eval | HIGH | Prompt, context envelope, eval logs | Regex-based secret scanning across combined context; fail closed with `SecretLeakageInContextError`. |
| **GQT-11** | Tenant Isolation Bypass | HIGH | Multi-tenant context envelopes | Assert `request.tenant_id == envelope.tenant_id`; reject cross-tenant context. |
| **GQT-12** | RAG Prompt Injection Crossing to Instructions | HIGH | Retrieved document chunks | Wrap all retrieved external chunks in non-executable `[RETRIEVED EXTERNAL EVIDENCE]` passive data blocks. |
| **GQT-13** | Repetition Penalty Numerical Instability | MEDIUM | Extreme penalty values ($r \to \infty$) | Bound repetition penalty to $[1.0, 10.0]$; check finite values after penalty application. |
| **GQT-14** | Incomplete Output Provenance & Audit Loss | LOW | Evaluation result recording | Structured `InferenceResult` recording model identity, tokenizer identity, seed, and weight hash verification. |

---

## 3. Detailed Threat Mitigations

### GQT-01: Checkpoint Incompatibility & Corruption
- **Failure Mode:** Loading a checkpoint trained with different layer counts, embedding dimensions, or corrupted tensors leads to silent dimension mismatches or undefined behavior.
- **Mitigation:**
  `validate_checkpoint_compatibility(checkpoint_data, expected_config)`:
  1. Inspects `model_state_dict` keys against `ChakrMicro` expected parameters.
  2. Asserts identical shape for every tensor (e.g. `embedding.weight` must be `[4096, 192]`).
  3. Validates total parameter count equals exactly 3,443,136.
  4. Raises `IncompatibleCheckpointError` on any discrepancy.

### GQT-04: Degenerate Probability Distributions
- **Failure Mode:** Logits with extreme values causing softmax underflow (all zeros) or NaNs in sampling distributions.
- **Mitigation:**
  `Sampler.sample()` checks:
  ```python
  if torch.isnan(probs).any() or probs.sum() <= 0:
      raise SamplingProbabilityError("Degenerate probability distribution encountered in Sampler.")
  ```

### GQT-06: Premature EOS Termination
- **Failure Mode:** An untrained or early-stage model emits EOS (token ID 1) at step 0, producing an empty response when the caller requested substantive output.
- **Mitigation:**
  If `step < config.min_new_tokens` and `next_token_id in stop_tokens`:
  The loop masks out `stop_tokens` or ignores EOS termination until `min_new_tokens` is reached.

### GQT-08: Deterministic Reproducibility
- **Failure Mode:** Subtle differences in thread scheduling or unseeded generators yield different tokens for the same prompt, config, and seed.
- **Mitigation:**
  The PyTorch RNG generator is instantiated specifically for the tensor device and seeded with `(seed + step) % (2**31 - 1)`. Greedy decoding ($T=0.0$) uses deterministic `torch.argmax()`.
