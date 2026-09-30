# STEP 46 — THREAT MODEL & SECURITY SPECIFICATION

## Local Model Runtime & Interactive Streaming Session Layer

**Date:** 2026-09-30  
**Status:** Approved Specification  
**Component:** Local Model Runtime (`LocalModelRuntime`, `LocalModelSession`, `StreamGenerator`)  

---

## 1. Threat Matrix

| Threat ID | Threat Description | Attack Vector / Trigger | Severity | Mitigation Strategy | Enforcement Mechanism |
|---|---|---|---|---|---|
| **LMR-01** | Context Window Overflow in Multi-Turn Dialogue | Accumulation of user turns and assistant responses exceeding 512 context limit | **Critical** | Governed Context Compaction & Sliding Budget Allocation | `ConversationContextManager` calculates total tokens across history, cognitive memory, and prompt; evicts oldest non-pinned turns before tokenization |
| **LMR-02** | Secret Leakage via Interactive Streaming | User prompt or memory retrieval contains API keys or credentials; streamed token-by-token | **High** | Pre-Flight Scanning & Fail-Closed Enforcement | `InferenceContextBuilder.scan_for_secrets()` executes before prefill or streaming commences; raises `SecretLeakageInContextError` |
| **LMR-03** | Cross-Session State Pollution | Reused session object retains prompt history or KV cache from prior tenant/user | **High** | Strict Session Isolation & Clean Cache Reset | `LocalModelSession.reset()` zeroes KV cache; session instances enforce immutable `tenant_id` and `session_id` |
| **LMR-04** | Streaming Generator Abort / Cache Desync | Caller breaks out of streaming loop prematurely; KV cache sequence counter corrupted | **Medium** | Generator Cleanup & Re-entry Guard | Streaming generator uses `try...finally` block to reset active sequence state and invalidate stale intermediate decode state |
| **LMR-05** | Premature EOS Suppression Failure in Streaming | Stop token emitted before `min_new_tokens` horizon is met during streaming | **Medium** | Streaming Active Stop Token Masking | Logits of stop tokens masked with $-10^9$ during steps $< \text{min\_new\_tokens}$; yields valid continuations |
| **LMR-06** | Prompt Injection via Multi-Turn Dialogue | Malicious user input attempts to override system prompt or spoof assistant replies | **High** | Turn Demarcation & Role Isolation | Structured `ConversationTurn` dataclass separates user, assistant, and system roles with unambiguous demarcation markers |
| **LMR-07** | Unbounded Memory Growth in Long Sessions | Endless conversation turns consume unbounded host RAM | **Medium** | Hard Turn History Cap & Bounded Memory Stores | Max conversation turns capped (default: 32 turns); older turns pruned or summarized into persistent cognitive memory |
| **LMR-08** | Neural Weight Mutation ($\Delta W \neq 0$) | Interactive execution mutates parameter tensors in memory | **Critical** | Pre & Post-Invocation Cryptographic Verification | SHA-256 weight hash checked at runtime initialization and periodically; fails closed on mismatch with canonical hash |
| **LMR-09** | Tampered Checkpoint Loading | Local runner loads corrupted or unverified weights file | **Critical** | Strict Pre-Flight Checkpoint Validation | `load_and_validate_checkpoint()` validates state dict structure, shapes, unique parameter count (3,443,136), and SHA-256 |
| **LMR-10** | Pathological Logits / Degenerate Probabilities | Logits contain NaNs or infinite values during streaming | **High** | Strict Sampling Safety (`strict_safety=True`) | Sampler verifies logit finiteness and raises `SamplingProbabilityError` on degenerate distribution, preventing corrupted token stream |

---

## 2. Invariant Specifications

The following invariants MUST remain enforced without exception across all Step 46 additions:

1. **Parameter Immutability ($\Delta W = 0$):**
   - No parameter tensor may be modified in-place or updated during local execution.
   - Canonical Weight SHA-256 remains:
     `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`
2. **Context Window Ceiling:**
   - Active sequence length strictly $\le 512$ tokens at all times.
3. **Tenant & Trust Isolation:**
   - Conversation sessions cannot cross tenant boundaries.
   - Cognitive memory items retrieved during conversation must preserve trust level provenance (`LOCAL_VERIFIED_MEMORY`).
4. **Clean Abort Semantics:**
   - If a generator is closed early (e.g. `break` in consumer loop), the session must remain in a valid, reusable, or cleanly resettable state.
