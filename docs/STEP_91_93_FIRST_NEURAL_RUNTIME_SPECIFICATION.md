# Specification: Steps 91–93 First End-to-End Neural Runtime

## 1. Overview and Architectural Purpose
Milestone Steps 91–93 establishes the first sovereign, end-to-end cognitive-neural execution path in ChakrView.
It connects the frozen neural core (`ChakrMicro`, 3,443,136 parameters), the `BPETokenizer`, resource-adaptive budgeting (`ResourceDetector`, `HardwareCapability`), and the cognitive subsystems (`PersistentProjectBrain`, structured reasoning, autonomous investigation, and self-evaluation).

The runtime execution flow:
```
INPUT QUERY
  ↓
TOKENIZER (Encoding & Token Bounds Validation)
  ↓
RESOURCE & CONTEXT BUDGET (LOW_RESOURCE / STANDARD / ACCELERATED)
  ↓
CHAKRMICRO (Greedy Autoregressive Decode with ΔW = 0 Verification)
  ↓
COGNITIVE RUNTIME ROUTING
  ├── DIRECT_INFERENCE: Self-contained, general prompt
  ├── PPB_RETRIEVAL: Involves project files, modules, or historical tasks
  ├── INVESTIGATION_LOOP: Critical knowledge is UNKNOWN / INSUFFICIENT
  └── ABSTAIN: Unresolvable gap, contradiction, or unauthorized query
  ↓
VERIFICATION & EPISTEMIC CLASSIFICATION (FACT / INFERRED / CONTESTED / UNKNOWN)
  ↓
FINAL RESPONSE & DURABLE KNOWLEDGE PERSISTENCE
```

---

## 2. Implemented Subsystems & Modules

### 2.1 Step 91: Neural Inference Contract (`chakrview/runtime/neural_inference_contract.py`)
- **`NeuralInferenceContract`**: Encapsulates model forward pass and autoregressive token generation.
  - Enforces `vocab_size == 4096`, `BOS == 0`, `EOS == 1`, `PAD == 2`.
  - Token bounds validation (`TokenBoundsError` raised on non-integer or out-of-bounds tokens).
  - Finite logits verification (`PathologicalLogitsError` raised on NaNs or Infinities).
  - Pre- and post-execution weight hashing (`WeightMutationDetectedError` raised if $\Delta W \neq 0$).
  - Bounded autoregressive generation with explicit stop reasons (`MAX_TOKENS`, `STOP_TOKEN`, `CONTEXT_LIMIT`, `ABSTAINED`).
  - Native structured abstention (`abstain()`).

### 2.2 Step 92: Resource-Adaptive Inference Pipeline (`chakrview/runtime/adaptive_inference_pipeline.py`)
- **`ResourceAdaptiveInferencePipeline`**: Adapts execution parameters dynamically based on detected or specified hardware capabilities:
  - **`LOW_RESOURCE`**: Bounded context window $\le 256$, maximum generation tokens clamped to 32, sequential processing, CPU-only fallback.
  - **`STANDARD`**: Context window $\le 512$, generation budget 64 tokens, multi-core CPU parallelism.
  - **`ACCELERATED`**: Context window $\le 512$, generation budget 128 tokens, GPU/accelerator allocation if available.
  - **CPU-First Guarantee**: Never assumes GPU availability; runs with identical semantics on CPU when accelerators are absent.

### 2.3 Step 93: Cognitive Runtime Integration (`chakrview/runtime/cognitive_neural_runtime.py`)
- **`EndToEndCognitiveNeuralRuntime`**: Sovereign orchestrator combining neural generation with cognitive verification:
  - **`CognitiveRoute`**: Classifies query into `DIRECT_INFERENCE`, `PPB_RETRIEVAL`, `INVESTIGATION_LOOP`, or `ABSTAIN`.
  - **PPB Integration**: Selectively queries targeted project knowledge and injects bounded summaries into the prompt context rather than loading the whole repository.
  - **Investigation Loop**: When an unknown entity is queried, creates an `InvestigationRequirement`, gathers evidence, verifies it using `EvidenceVerifier`, and commits verified facts to PPB.
  - **Honest Abstention**: Expresses absence of information as `EpistemicStatus.UNKNOWN` rather than inventing answers or hallucinating APIs.

---

## 3. Proven Capabilities
1. **End-to-End Neural Inference**: Real token encoding, forward pass, logits verification, greedy decoding, and Unicode reconstruction without mocks.
2. **Deterministic Reproducibility**: Exact token-for-token identical outputs under deterministic temperature ($0.0$).
3. **Hardware Adaptation**: Automatic adjustment of context windows ($256$ vs $512$) and generation budgets ($32$, $64$, $128$) across hardware profiles without crashing.
4. **CPU-First Execution**: Seamless execution in CPU-only environments without GPU dependencies.
5. **PPB Context Grounding**: Grounded project records retrieved and prepended to neural generation context.
6. **Autonomous Investigation for Unknown Entities**: Real investigation loop triggers evidence verification and commits `FACT` records to PPB.
7. **Explicit Abstention**: Clean `[ABSTAIN]` responses produced when information is unavailable.
8. **Durable Knowledge Across Restarts**: Tested that reloaded sessions query past investigations from SQLite PPB without re-running work.
9. **Frozen Neural Core Invariant**: Parameters ($3,443,136$) and SHA-256 weight hash (`c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`) verified bit-exact before and after every inference pass ($\Delta W = 0$).

---

## 4. Partially Proven Capabilities
1. **Accelerated Hardware Exploitation**: Explicit capability contract and device detection exist; physical CUDA execution was tested in software CPU-fallback mode as no NVIDIA hardware is active in this test runner.
2. **Context Window Compaction**: Truncation and selective context slicing operate correctly up to 512 tokens; full AST semantic summarization for giant files relies on PPB chunking.

---

## 5. Unproven Capabilities
1. **General Domain Conversational Fluency**: The 3.4M parameter neural model is in baseline initialization; it is not a pre-trained general knowledge LLM and will not generate fluent open-domain English without subsequent pretraining/fine-tuning milestones.
2. **Autonomous Weight Self-Evolution**: Deliberately disabled; self-improvement is strictly confined to project knowledge updates and task decomposition heuristics.

---

## 6. Authority & Security Boundaries
- **Zero Neural Write Authority**: The neural core outputs passive data tokens only. It cannot execute code, mutate SQLite tables, or alter filesystem files without cognitive gates.
- **Fail-Closed Token & Logit Bounds**: Non-integer tokens, out-of-vocab IDs, and non-finite logits raise immediate exceptions.
- **Epistemic Classification**: Statements generated by the model are marked `INFERRED` or `UNKNOWN`, never elevated to `FACT` without empirical verification.
