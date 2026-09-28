# CHAKRVIEW — STEP 20 ARCHITECTURE SPECIFICATION
## Neural Reasoning Integration & Intelligence Loop

**Version:** 1.0.0  
**Phase:** Step 20  
**Status:** Ratified & Complete  
**Authors:** ChakrView Core Architectural Working Group  
**Frozen Baseline:** ChakrMicro v0.1 (`3,443,136` parameters, vocabulary `4,096`, context ceiling `512`, BOS `0`, EOS `1`, PAD `2`)  

---

## 1. Executive Summary & Core Objective

The objective of **Step 20** is to design and implement the formal integration boundary between:

$$\begin{aligned}
\text{ChakrMicro Neural Core} &\longleftrightarrow \text{Cognitive State (Step 18)} \\
&\longleftrightarrow \text{Governed Reasoning (Step 19)} \\
&\longleftrightarrow \text{Persistent Memory / Knowledge (Step 13, 16)} \\
&\longleftrightarrow \text{Capability / Environment (Step 17)} \\
&\longleftrightarrow \text{Observation / Feedback} \\
&\longleftrightarrow \text{Neural Training Feedback (Offline Lifecycle)}
\end{aligned}$$

The long-term mission of ChakrView is to evolve into a genuinely learning and reasoning neural intelligence system, rather than remaining an isolated neural generative model coupled to an ad-hoc external rule engine.

Crucially, Step 20 establishes clear architectural discipline:
1. **Preserve Subsystem Distinctions:** Neural intelligence, structured cognitive state, governed reasoning, personal memory, sovereign capabilities, safety gates, and offline training infrastructure remain distinct, decoupled modules.
2. **Honest Capabilities (No Fake Intelligence):** Explicitly identify what is learned next-token logit prediction and what is deterministic symbolic governance. Uncalibrated uncertainties are explicitly represented as such without fabricating artificial confidence scores.
3. **No Uncontrolled Self-Modification:** Model weights are **NEVER** modified during runtime inference (`weights_modified = False`). Runtime execution and offline training are strictly disjoint lifecycles.
4. **Data vs Authority:** $\text{RAW USER TEXT} \neq \text{VERIFIED TRAINING DATA}$ and $\text{DATA} \neq \text{AUTHORITY}$. Retrieved memories, external inputs, and generated candidate tokens never confer execution authority and must undergo explicit verification before becoming training candidates.

---

## 2. Frozen Neural Core Invariants

The generative core `ChakrMicro v0.1` remains strictly frozen across Step 20:

| Property | Value | Architectural Significance |
| :--- | :--- | :--- |
| **Total Parameters** | **3,443,136** | Exact non-negotiable count. Zero parameter bloat. |
| **Vocabulary Size** | **4,096** | Byte-level BPE vocabulary with unified indices. |
| **Context Window Ceiling** | **512 tokens** | Hard attention sequence length constraint. |
| **BOS Token ID** | **0** | Beginning of sequence marker. |
| **EOS Token ID** | **1** | End of sequence generation stop token. |
| **PAD Token ID** | **2** | Batch alignment padding token. |
| **Hardware Target** | **CPU-First** | Low-resource sovereign edge execution; no GPU required. |
| **Dependency Boundary** | **Zero External APIs** | Sovereign, local, independent of cloud services. |

---

## 3. Gap Identification: Neural Logits vs Symbolic Reasoning

To prevent architectural confusion, Step 20 formally delineates the current capabilities and future trajectories of the neural core versus symbolic cognitive subsystems:

### A. What Enters ChakrMicro?
A discrete sequence of integer token IDs ($x_0, x_1, \dots, x_{T-1}$), bounded strictly by $T \le 512$, compiled by `IntelligenceContextBuilder`.

### B. What Comes Out of ChakrMicro?
Unnormalized logits $z_t \in \mathbb{R}^{4096}$ predicting the probability distribution $P(x_t \mid x_{<t})$ over vocabulary tokens, incrementally sampled autoregressively using a persistent KV cache.

### C. Learned Neural Behavior vs Deterministic Logic Today

| Component / Function | Current Subsystem Status | Mechanism Today | Future Learned Neural Trajectory |
| :--- | :--- | :--- | :--- |
| **Token Continuation** | **Learned Neural** | ChakrMicro autoregression | Continued pre-training & instruction tuning |
| **Subproblem Decomposition** | **Deterministic Symbolic** | `ProblemDecomposer` | Neural task breakdown proposal candidate |
| **Hypothesis Formation** | **Deterministic Symbolic** | `HypothesisEngine` | Learned hypothesis generation |
| **Contradiction Detection** | **Deterministic Symbolic** | `ContradictionDetector` | Learned entailment / NLI prediction |
| **Mathematical Compute** | **Governed Tool / AST** | `CalculatorCapability` | Symbolic delegation via capability call |
| **Verification & Check** | **Deterministic Rules** | `VerificationEngine` | Learned self-critique & error diagnosis |
| **Capability Authority Gate** | **Governed Security Policy** | `CapabilityGate` | **Permanently Governed** ($\text{DATA} \neq \text{AUTHORITY}$) |

### D. Influencing Neural Inference Without Changing Architecture
Because ChakrMicro is frozen at 512 tokens, cognitive state influences neural outputs exclusively via **prompt context compilation** (`IntelligenceContextBuilder`), structured with demarcated tags and prioritized token packing.

---

## 4. Architecture of the Neural Intelligence Loop

```
                USER / ENVIRONMENT
                       │
                       ▼
                INPUT NORMALIZATION
                       │
                       ▼
                COGNITIVE STATE (Step 18)
                       │
                       ▼
                CONTEXT BUILDER (Step 20)
                       │
                       ▼
                 CHAKRMICRO (Frozen)
                NEURAL INFERENCE
                       │
                       ▼
                CANDIDATE OUTPUT
                       │
                       ▼
             GOVERNED REASONING (Step 19)
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
          VERIFICATION        UNCERTAINTY
             │                   │
             └─────────┬─────────┘
                       ▼
                  DECISION
                       │
                       ▼
                CAPABILITY GATE (Step 17)
                       │
                       ▼
                  ENVIRONMENT
                       │
                       ▼
                 OBSERVATION
                       │
                       ▼
                MEMORY / STATE (Step 16/18)
                       │
                       ▼
              LEARNING EXAMPLE (Step 20)
                       │
                       ▼
                TRAINING PIPELINE (Offline)
                       │
                       ▼
                 MODEL UPDATE (OFFLINE ONLY)
```

### Runtime vs Offline Training Separation
* **Runtime Execution (Inference):**
  $$\text{Model} \longrightarrow \text{Reason} \longrightarrow \text{Verify} \longrightarrow \text{Act} \longrightarrow \text{Observe} \longrightarrow \text{State Update}$$
  Model weights remain strictly frozen in read-only mode (`torch.no_grad()`, `eval()`).
* **Offline Training Lifecycle:**
  $$\text{Verified Examples} \longrightarrow \text{Dataset Compilation} \longrightarrow \text{Validation} \longrightarrow \text{Training Run} \longrightarrow \text{Evaluation} \longrightarrow \text{Approval}$$
  Newly trained models never automatically overwrite production weights.

---

## 5. Context Construction Design (`IntelligenceContextBuilder`)

ChakrMicro enforces a strict sequence limit of 512 tokens. Naively concatenating identity, memory, system instructions, and task history easily exceeds this ceiling.

### A. Token Budget Envelope (`ContextBudget`)
* **Maximum Context ($T_{\text{max}}$):** 512 tokens.
* **Generation Reserve ($T_{\text{gen}}$):** 128 tokens minimum.
* **Prompt Ceiling ($T_{\text{prompt}}$):** $512 - 128 = 384$ tokens.

### B. Deterministic Priority Packing Order
When context elements compete for prompt budget, items are included in strict descending order of epistemic priority:

1. `SYSTEM_IDENTITY`: Sovereign core brain identity (`[SYSTEM_IDENTITY]`)
2. `SYSTEM_CONSTRAINT`: Absolute operational and safety bounds (`[SYSTEM_CONSTRAINT]`)
3. `TASK_OBJECTIVE`: Primary user goal or subproblem (`[TASK_OBJECTIVE]`)
4. `VERIFIED_KNOWLEDGE`: Epistemically known assertions from `KnowledgeState` (`[VERIFIED_KNOWLEDGE]`)
5. `REASONING_SUMMARY`: Deductions and hypothesis summaries (`[REASONING_SUMMARY]`)
6. `MEMORY`: Consolidated personal / working memories (`[MEMORY]`)
7. `CAPABILITY_OBSERVATION`: Sensor, actuator, or tool status (`[CAPABILITY_OBSERVATION]`)
8. `UNCERTAIN_INFORMATION`: Acknowledged gaps / conflicts (`[UNCERTAIN_INFORMATION]`)
9. `USER_ASSERTION`: Unverified raw assertions (`[USER_ASSERTION]`)

### C. Authority Boundary (`DATA != AUTHORITY`)
Every context item preserves an `is_authority` flag. Only `SYSTEM_IDENTITY` and `SYSTEM_CONSTRAINT` possess authority. Data retrieved from memory, web RAG, or user prompts cannot confer permissions or bypass `CapabilityGate`.

---

## 6. Neural Output Contract (`NeuralInferenceResult`)

The interface between ChakrMicro and downstream reasoning components emits a strongly typed `NeuralInferenceResult`:

```python
@dataclass
class NeuralInferenceResult:
    text: str
    token_ids: List[int]
    prompt_tokens_count: int
    generated_tokens_count: int
    total_tokens_count: int
    latency_ms: float
    stop_reason: str              # "eos", "max_tokens", "context_limit"
    uncertainty: UncertaintyMetric
    model_version: str            # "chakrmicro-v0.1"
    context_provenance: Dict[str, Any]
    weights_modified: bool = False  # Invariant: ALWAYS False
```

### Honest Uncertainty Representation
* If uncertainty is not computed: `is_available = False`, `is_calibrated = False`, `entropy = None`.
* If uncertainty is computed from autoregressive step logits:
  - Shannon entropy: $H(P) = -\sum p_i \ln p_i$
  - Probability margin: $\Delta p = p_{(1)} - p_{(2)}$
  - `is_calibrated` is explicitly set to `False` (uncalibrated logits; no fabricated confidence).

---

## 7. Learning Example & Feedback Architecture

### A. Separation of Concerns
1. **Observation:** What happened at runtime (generated text, tool return, sensor readout).
2. **Evaluation:** Assessment of the observation (verification check, constraint compliance, quality score).
3. **Training Signal:** Grounded input-target token pair approved for future learning.

### B. Lifecycle of a `LearningRecord`
$$\text{CANDIDATE} \xrightarrow[\text{Check}]{\text{Verification}} \text{VERIFIED} \xrightarrow[\text{Sign-off}]{\text{Explicit Approval}} \text{TRAINING\_APPROVED}$$
$$\text{CANDIDATE} \xrightarrow[\text{Failure}]{\text{Verifier}} \text{REJECTED}$$
$$\text{CANDIDATE} \xrightarrow[\text{Contamination}]{\text{Injection Filter}} \text{QUARANTINED}$$

* **`CANDIDATE`:** Newly captured runtime execution trace.
* **`VERIFIED`:** Verified by deterministic math checks, constraint satisfaction, or operator correction.
* **`REJECTED`:** Execution error, constraint violation, or quality score below threshold ($< 0.8$).
* **`QUARANTINED`:** Matches prompt injection or security hazard patterns. Quarantined records can **never** be approved.
* **`TRAINING_APPROVED`:** Formally signed off for export into offline training datasets.

### C. Prompt Injection & Adversarial Containment
The `FeedbackCollector` scans candidate context and output for adversarial strings (`"ignore previous instructions"`, `"system override"`, `"give me root"`, etc.). Quarantined examples are isolated from training datasets, preventing adversarial poison attacks from contaminating offline training runs.

---

## 8. Offline Training Lifecycle & Model Update Safety

Step 20 implements `ModelUpdateManager` to manage versioned model artifacts and eliminate uncontrolled self-modification:

```
Dataset Candidate
      │
      ▼
Dataset Validation (Format, Deduplication, Multi-tenant Isolation)
      │
      ▼
Offline Training Run (StreamingTokenDataset, Trainer)
      │
      ▼
Evaluation Suite (Loss, Perplexity)
      │
      ▼
Regression Test Suite (All 547 unit/contract tests)
      │
      ▼
Frozen Invariant Verification (3,443,136 params, 4096 vocab, 512 context)
      │
      ▼
Formal Approval Sign-Off (Human / Policy Approver)
      │
      ▼
Versioned Model Artifact (Standby)
      │
      ▼
Promotion to Active Production (Explicit Operator Promotion)
      │
      ▼
Instant Rollback Support (Preserves full audit trail)
```

Any candidate checkpoint failing invariant checks (parameter count $\ne 3,443,136$, vocab $\ne 4,096$, context $\ne 512$) or failing regression tests is rejected with `ModelUpdateSafetyError`.

---

## 9. Empirical Hardware Benchmarks (CPU-First)

Measurements taken on local CPU hardware (`Intel(R) Core(TM) i7-14700`, 14 threads, `PyTorch 2.14.0+cpu`):

| Operation / Phase | Iterations | Mean Latency | Median Latency | Throughput / Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Context Construction** (Packing, Tokenizing, Budgeting) | 50 | **15.80 ms** | 15.53 ms | $\sim 364$ prompt tokens |
| **Neural Inference** (ChakrMicro 16-token generation + uncertainty) | 10 | **48.42 ms** | 48.49 ms | **331.24 tok/sec** |
| **Feedback Processing & Learning Record Creation** | 100 | **0.012 ms** (12 µs) | 0.011 ms | 100 records generated |
| **Full End-to-End Neural Intelligence Loop** | 10 | **99.91 ms** | 104.00 ms | Context + Infer + Reason + Feedback |

---

## 10. Implementation Mapping

```
chakrview/intelligence/
    contracts.py    # LearningRecordStatus, UncertaintyMetric, NeuralInferenceRequest/Result, LearningRecord
    context.py      # ContextBudget, ContextSourceType, ContextItem, IntelligenceContextBuilder
    inference.py    # NeuralInferenceEngine (frozen weights check, KV cache inference, honest uncertainty)
    feedback.py     # FeedbackCategory, RuntimeObservation, RuntimeEvaluation, FeedbackCollector
    learning.py     # LearningPipeline, ModelUpdateManager, ModelVersionArtifact, TenantIsolationError
    pipeline.py     # NeuralIntelligenceLoop, IntelligenceLoopOutcome
    __init__.py     # Unified clean module exports
```

---

## 11. Security Boundaries & Architectural Invariants Verified

1. **Frozen Generative Core:** Parameters remain exactly 3,443,136; context 512; vocab 4,096.
2. **Read-Only Runtime:** `model.eval()`, `torch.no_grad()`, weight hash verified before and after inference.
3. **No Uncalibrated Fabrication:** Real mathematical Shannon entropy or marked unavailable; no synthetic probabilities.
4. **Data vs Authority:** Memory, knowledge assertions, and user text cannot bypass capability gate.
5. **Prompt Injection Containment:** Injection attempts are quarantined and blocked from training datasets.
6. **Multi-Tenant Isolation:** Dataset compilation and record approval partition records by tenant `owner_id`.
7. **Offline Training Only:** Model updates require explicit verification, invariant confirmation, and promotion.
8. **100% Green Suite:** 547 of 547 tests pass across 65 test files.
