# Single-Field Entity Copy Pilot — Formal Training Proposal

**Status**: **PREPARED ONLY — TRAINING NOT EXECUTED**  
**Authorization Gate**: Requires explicit user authorization before execution  
**Target Milestone**: v0.7 Capability Track  
**Estimated Compute**: ~45–60 seconds on CPU (250–500 steps, batch size 2)

---

## 1. Scientific Hypothesis

In the 100-step SFT V3 pilot, ChakrMicro failed to achieve exact task matches (0/30) because it was simultaneously tasked with multi-field JSON formatting, schema keys, boolean flags, and entity extraction across 5 disparate domains. 

**Hypothesis**: By isolating the training objective to **a single entity copying task** with explicit dialog prompt framing (`User: ... Assistant: <val>`), ChakrMicro can develop induction/pointer attention circuits from context tokens to generation tokens without being overwhelmed by multi-field syntax.

---

## 2. Dataset Construction & Strict Isolation

### Rules & Invariants:
1. **Zero Access to Test Data**: Held-out `test.jsonl` remains completely sealed and untouched.
2. **Context-Grounded Verification**: 100% of training and validation targets must be verified exact substrings of the context passage.
3. **Template Family & Document Split**:
   - Train set: 200 synthetic examples across 4 template families (`atc_train`, `inv_train`, `srv_train`, `met_train`).
   - Validation set: 40 synthetic examples using strictly disjoint template families (`atc_val`, `inv_val`, `srv_val`, `met_val`) with unique entities and contexts.
   - Zero overlap of contexts, target values, or host IDs between train and validation.

### Example Construction:
```text
Context: "Cache telemetry: node_cache_01 registered status active with latency 12ms."
Question: "Copy the exact status from context:"
Prompt: "User: In the following context: \"Cache telemetry: node_cache_01 registered status active with latency 12ms.\"\nCopy the exact status from context:\nAssistant:"
Target: " active"
```

---

## 3. Training & Supervision Contract

- **Base Checkpoint**: Canonical Step 10,000 (`96357705...e9f471ee`).
- **Prompt Token Masking**: Prompt tokens masked with `-100`.
- **Target Supervision**: Supervise ONLY the target entity tokens and terminal `<eos>` (Token ID 1).
- **Optimization Parameters**:
  - Optimizer: AdamW (`lr=2e-4`, `weight_decay=0.01`, `betas=(0.9, 0.95)`).
  - Learning Rate Schedule: 25 steps linear warmup, cosine decay to `2e-5`.
  - Batch Size: 2 (effective batch size 2).
  - Maximum Steps: 300 steps.
  - Validation Interval: Evaluate validation loss and greedy exact match every 25 steps.

---

## 4. Predefined Acceptance & Stopping Criteria

| Metric | Baseline (Step 10,000) | Minimum Success Threshold | Target Goal | Early Stop / Rollback Trigger |
| :--- | :---: | :---: | :---: | :--- |
| **Exact Match Copying** | **0/40 (0.0%)** | **>= 24/40 (60.0%)** | **>= 32/40 (80.0%)** | < 4/40 (10.0%) at step 200 $\to$ Terminate |
| **Normalized Value Match** | **0/40 (0.0%)** | **>= 28/40 (70.0%)** | **>= 36/40 (90.0%)** | < 6/40 (15.0%) at step 200 $\to$ Terminate |
| **Doctest Code Attractor Rate** | 15.0% | **<= 2/40 (5.0%)** | **0/40 (0.0%)** | > 10/40 (25.0%) at step 100 $\to$ Terminate |
| **Validation Loss** | ~5.8 nats | **< 2.5 nats** | **< 1.8 nats** | Loss divergence (NaN/Inf or > 6.0) |
| **Pretraining Replay Regression** | Baseline | **<= 10% loss delta** | **<= 5% loss delta** | > 25% degradation on 10 replay records |

---

## 5. Rollback Strategy & Checkpoint Isolation

- All artifacts written exclusively to `experiments/single_field_copy_pilot/`.
- If early stop triggers or acceptance thresholds are not met:
  1. The experiment candidate will NOT be promoted.
  2. The run will be labeled `PILOT_FAILED` with empirical logs preserved.
  3. Canonical Step 10,000 remains the sovereign active checkpoint.
