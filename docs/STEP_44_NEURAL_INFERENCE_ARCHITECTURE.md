# Step 44 Architecture Specification: End-to-End Neural Inference Pipeline

**Date:** 2026-09-30  
**Status:** RATIFIED & ACTIVE  
**Component:** `chakrview/runtime/pipeline.py`

---

## 1. Motivation

Across Steps 1 through 43, ChakrView established:
- An indigenous byte-level BPE tokenizer (Step 3).
- The frozen ChakrMicro transformer core ($\Delta W = 0$, Step 4).
- Autoregressive generation mechanics with KV caching and sampling (Step 10).
- RAG and knowledge retrieval via BM25 (Steps 9, 11).
- Persistent cognitive memory (Step 24, 25, 43).
- Federated cognitive task graphs and synthesis (Step 42, 43).

However, an application calling the neural core previously had to manually coordinate tokenization, context assembly, tensor wrapping, KV caching, sampling, logits checking, detokenization, and security scanning.

Step 44 delivers the **unified, end-to-end Neural Inference Pipeline** (`InferenceEngine`) providing a cohesive, strongly typed contract that connects tokenizer, context, model, sampling, and post-forward verification into a deterministic, CPU-first inference subsystem.

---

## 2. End-to-End Pipeline Workflow

```
+----------------------------------------------------------------------------------------------------+
|                                           InferenceRequest                                         |
|  - raw prompt text OR explicit prompt_tokens                                                       |
|  - optional CognitiveContextEnvelope (Step 42)                                                     |
|  - optional PersistentCognitiveMemoryAdapter (Step 43)                                             |
|  - optional GovernedKnowledgeRetrievalCapability (Step 43)                                         |
|  - GenerationConfig (max_new_tokens, temperature, top_k, stop_tokens, seed)                        |
|  - tenant_id, session_id, truncate_if_overflow flag                                                |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                      InferenceContextBuilder                                       |
|  1. Tenant Boundary Validation: Verify tenant_id across request and context envelope               |
|  2. Secret Scanning: Regex scanning across all input text (raises SecretLeakageError)              |
|  3. Context Synthesis: Merge working memory, external RAG evidence, and prompt                     |
|     - Demarcate external RAG evidence with passive delimiters: [RETRIEVED EXTERNAL EVIDENCE]       |
|  4. Token Budget Allocation: Verify prompt_tokens + max_new_tokens <= 512                         |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                         BPETokenizer                                               |
|  - Encode assembled prompt text to integer token IDs [t_0, t_1, ..., t_k]                          |
|  - Verify all token IDs in [0, 4096)                                                               |
|  - Prepend BOS (token 0) if requested                                                              |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                  ChakrMicro (Frozen Neural Core)                                   |
|  - Pre-flight SHA-256 weight hash assertion == EXPECTED_WEIGHT_HASH                               |
|  - Execute under torch.no_grad() and model.eval()                                                  |
|  - Prefill pass: Populate KVCache across 6 transformer layers                                      |
|  - Incremental decoding: decode_next() for each generated token                                    |
|  - Validate output logits: torch.isfinite(logits).all() (no NaNs / Infs)                           |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                    Deterministic Sampler                                           |
|  - Greedy (argmax) if temperature == 0.0                                                           |
|  - Temperature scaling, top-k filtering, repetition penalty                                        |
|  - Stop on EOS (token 1) or max_new_tokens                                                         |
|  - Monotonic token accumulation                                                                    |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                        Tokenizer Decode                                            |
|  - Lossless UTF-8 decoding of generated token IDs                                                  |
|  - Post-flight SHA-256 weight hash assertion (verify dW = 0)                                       |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+----------------------------------------------------------------------------------------------------+
|                                         InferenceResult                                            |
|  - generated_text: str                                                                             |
|  - generated_token_ids: List[int]                                                                  |
|  - prompt_token_count, generated_token_count, total_token_count                                    |
|  - stop_reason: StopReason (EOS, MAX_TOKENS, CONTEXT_LIMIT)                                         |
|  - latency_ms: float                                                                               |
|  - provenance: Dict[str, Any] (citations, trust levels)                                            |
|  - weight_hash_verified: bool (True)                                                               |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. Core Abstractions

### 3.1 `InferenceRequest`
```python
@dataclass
class InferenceRequest:
    prompt: Optional[str] = None
    prompt_tokens: Optional[List[int]] = None
    context_envelope: Optional[CognitiveContextEnvelope] = None
    generation_config: Optional[GenerationConfig] = None
    tenant_id: str = "default_tenant"
    session_id: str = "default_session"
    add_bos: bool = True
    add_eos: bool = False
    truncate_if_overflow: bool = False
```

### 3.2 `InferenceContext`
```python
@dataclass
class InferenceContext:
    full_prompt_text: str
    token_ids: List[int]
    tenant_id: str
    session_id: str
    trust_levels: Dict[str, str]
    sources: List[Dict[str, Any]]
```

### 3.3 `InferenceResult`
```python
@dataclass
class InferenceResult:
    text: str
    token_ids: List[int]
    prompt_tokens: List[int]
    input_token_count: int
    output_token_count: int
    total_token_count: int
    generation_config: GenerationConfig
    stop_reason: StopReason
    latency_ms: float
    provenance: Dict[str, Any]
    weight_hash_verified: bool
```

### 3.4 `InferenceEngine`
The central pipeline class coordinating:
1. `validate_contract()`: Tokenizer vocabulary == 4096, Model vocabulary == 4096, Special tokens BOS=0, EOS=1, PAD=2.
2. `execute(request: InferenceRequest) -> InferenceResult`: Full pipeline invocation.
3. `generate_text(prompt: str, ...) -> str`: High-level convenience method.
4. `forward(token_ids: List[int]) -> torch.Tensor`: Low-level forward pass returning validated logits.

---

## 4. Tokenizer $\longleftrightarrow$ Model Contract

The pipeline enforces strict invariant consistency between the tokenizer and model:
1. **Vocabulary Equality:**
   $$\text{tokenizer.vocab\_size} \equiv \text{model.config.vocab\_size} = 4096$$
2. **Token Range Invariant:**
   $$\forall t \in \text{token\_ids}, \quad 0 \le t < 4096 \quad \text{and} \quad t \in \mathbb{Z}$$
3. **Special Token Invariant:**
   $$\text{BOS} = 0, \quad \text{EOS} = 1, \quad \text{PAD} = 2$$
4. **Sequence Horizon Invariant:**
   $$\text{len}(\text{prompt\_tokens}) + \text{max\_new\_tokens} \le 512$$
   If exceeded, fail closed with `ContextOverflowError` unless `truncate_if_overflow=True`.

---

## 5. Security & Trust Invariants

1. **`EXTERNAL KNOWLEDGE != VERIFIED MEMORY`**:
   Retrieved chunks from RAG are tagged with `trust_level="RETRIEVED_EXTERNAL"` and formatted into inert passive blocks.
2. **Zero Secret Exposure**:
   Any credential detected in input text or context envelope triggers `SecretLeakageInContextError`.
3. **Tenant Isolation**:
   Cross-tenant context envelopes raise `CognitiveContextTenantViolationError`.
4. **Frozen Weights ($\Delta W = 0$)**:
   Model weights SHA-256 must match `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` before and after execution.
