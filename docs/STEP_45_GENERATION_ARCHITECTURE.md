# Step 45 Architecture Specification: Neural Inference Validation & Generation Quality Layer

**Date:** 2026-09-30  
**Status:** RATIFIED & ACTIVE  
**Component:** `chakrview/runtime/pipeline.py` & `chakrview/runtime/sampling.py`

---

## 1. Motivation & Objective

Step 44 established the end-to-end neural inference pipeline connecting the tokenizer and frozen ChakrMicro core.

Step 45 hardens this pipeline into a **stable, trustworthy evaluation and generation-quality layer**. This establishes the exact contract, numerical validation, repetition controls, model identity verification, and evaluation harness needed before beginning pretraining work.

---

## 2. Stable Inference Contract

### 2.1 Input Contract (`InferenceRequest`)
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

### 2.2 Configuration Contract (`GenerationConfig` & `SamplingConfig`)
```python
@dataclass
class GenerationConfig:
    max_new_tokens: int = 64
    min_new_tokens: int = 0                  # New in Step 45: suppresses premature EOS
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    stop_token_ids: List[int] = field(default_factory=lambda: [1])  # EOS = 1
    context_window: int = 512
    context_overflow_policy: str = "stop"
    stream_interval: int = 1
```

```python
@dataclass
class SamplingConfig:
    temperature: float = 0.0                 # 0.0 = greedy argmax
    top_k: int = 0                           # 0 = disabled
    top_p: float = 1.0                       # 1.0 = disabled
    repetition_penalty: float = 1.0          # [1.0, 10.0]
    min_prob: float = 0.0                    # [0.0, 1.0)
    seed: Optional[int] = None               # Deterministic PRNG seed
```

### 2.3 Output Contract (`InferenceResult` & `ModelIdentity`)
```python
@dataclass
class ModelIdentity:
    architecture_name: str = "ChakrMicro"
    version: str = "0.1.0"
    parameter_count: int = 3_443_136
    vocab_size: int = 4096
    max_context_len: int = 512
    weight_hash: str = "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
    tokenizer_checksum: str = "7498d92adeef7c6db98d89a444a7f0e303dd5e7ea4b679a95781a95e6347c617"
```

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
    stop_reason: StopReason                  # EOS, MAX_TOKENS, CONTEXT_LIMIT
    latency_ms: float
    model_identity: ModelIdentity            # Complete model identity telemetry
    reproducibility: Dict[str, Any]          # seed, is_greedy, strategy
    provenance: Dict[str, Any]
    weight_hash_verified: bool
```

---

## 3. Checkpoint Compatibility Validation

To prevent loading corrupted or incompatible model weights, Step 45 introduces `validate_checkpoint_compatibility()`:

```python
def validate_checkpoint_compatibility(
    checkpoint: Dict[str, Any],
    expected_config: Optional[ModelConfig] = None,
) -> bool:
    """
    Validate that checkpoint dictionary is compatible with ChakrMicro:
    1. Must contain 'model_state_dict'.
    2. Total parameter count must match expected (3,443,136).
    3. Layer count (6) and tensor shapes must match ModelConfig.
    4. Raises IncompatibleCheckpointError on any discrepancy.
    """
```

---

## 4. Deterministic Reproducibility Scope

1. **Greedy Mode ($T = 0.0$):**
   - Strictly deterministic across all calls on the same platform and CPU architecture.
   - Guaranteed identical token sequences for identical prompt and weights.
2. **Seeded Stochastic Mode ($T > 0.0$, `seed` specified):**
   - Uses dedicated `torch.Generator(device=cpu)` seeded with `(seed + step) % (2**31 - 1)`.
   - Guaranteed identical token sequences across runs on identical PyTorch versions and platforms.
3. **Cross-Platform Limitations:**
   - Float math optimizations (e.g. AVX-512 vs NEON) can introduce negligible float discrepancies in extreme boundary logits; greedy ties are resolved deterministically by index ordering.

---

## 5. Evaluation Harness (`scripts/evaluate_neural_generation.py`)

A standardized evaluation harness measuring engine behavior across 6 prompt categories:
1. **Factual Continuation:** Tests stable factual completion format.
2. **Open Continuation:** Tests multi-sentence decoding.
3. **Simple Reasoning:** Tests step-by-step token sequence handling.
4. **Multilingual (Hindi-English):** Tests Devanagari byte-level BPE tokens.
5. **Boundary Cases:** Near-context ceiling prompts ($T \to 512$).
6. **Negative / Security Cases:** Injection attempts and secret detection.

*Note: The harness measures pipeline reliability, not model world knowledge or intelligence.*
