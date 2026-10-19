# CHAKRVIEW SFT V3 LEAKAGE-SAFE EVALUATION PROTOCOL

## 1. Objective and Scope

This protocol establishes the formal validation and acceptance framework for evaluating future SFT V3 candidate checkpoints in ChakrView.

The overarching design principle is:
> **Native neural capability and inference-layer scaffolding must be measured and reported independently. An inference constraint that forces syntactic compliance must never be credited as neural intelligence.**

This protocol governs the transition from experimental pilot models to candidate evaluation without risking train/validation contamination or misleading metric aggregation.

---

## 2. Capability Evaluation Layers

Evaluation of any model checkpoint must strictly separate three distinct layers:

### Layer A: Native Generation (Zero-Constraint)
- **Decoding Mechanism**: Unconstrained greedy decoding (`argmax` over full vocabulary, temperature=0.0).
- **Core Questions**:
  1. Does the neural model naturally recognize that the prompt demands JSON format?
  2. Does the model output valid JSON syntax without external intervention?
  3. Does the model naturally stay on task, or does it collapse into pretraining attractor basins (e.g., Python doctests, code blocks, or loop repetition)?
  4. Are the predicted values semantically accurate?
- **Permitted Modifications**: None. No prefix trie, no JSON regex, no schema masking, and no post-generation string repair.

### Layer B: Constrained Decoding (Inference Control Plane)
- **Decoding Mechanism**: Logit-masking via the Schema Prefix Trie (`SchemaConstrainedPrefixTrieDecoder`).
- **Core Questions**:
  1. Does the decoder prevent known syntax errors (unquoted keys, illegal characters, type mismatches)?
  2. Does the decoder avoid deadlocks (`deadlock_count == 0`), terminating with valid closing punctuation within the token budget?
  3. Does the constraint introduce out-of-vocabulary tokens or unviable token fragment traps?
- **Reporting Rule**: Metrics collected under Layer B evaluate **runtime tooling robustness**, NOT native model capability.

### Layer C: Semantic Task Performance (Information Extraction Accuracy)
- **Decoding Mechanism**: Measured across both Layer A (where valid) and Layer B.
- **Core Questions**:
  1. Did the model extract the exact entity specified in the input context passage?
  2. Are all required schema fields populated with semantically correct types and values?
  3. Did the model abstain appropriately if the requested entity was missing or contradictory?
- **Scoring Principle**: Syntactic validity is a prerequisite for scoring, but **a valid JSON object with incorrect field values receives an Exact Task Match score of 0.0**.

---

## 3. Metric Definitions

All evaluation runs must record and report the following deterministic metrics:

| Metric Name | Calculation / Rule | Target Layer | Definition of Success |
| :--- | :--- | :--- | :--- |
| **JSON Opening Rate** | Output begins with `^\s*\{` | Layer A, B | 100% |
| **Valid JSON Rate** | Successful `json.loads(output)` without error | Layer A, B | Layer A: >0% (target 60%+), Layer B: 100% |
| **Completed Root Object Rate** | Root JSON object closes (`}`) before token exhaustion | Layer A, B | Layer A: >50%, Layer B: 100% |
| **Required Key Coverage** | `len(keys_present & required_keys) / len(required_keys)` | Layer A, B | 100% |
| **Unknown Key Rate** | Records emitting keys not defined in prompt schema | Layer A | 0% |
| **Exact Task Match (EM)** | Parsed JSON matches ground-truth target object exactly (`parsed == target`) | Layer A, B | Baseline target: >50% on grounded val |
| **Per-Field Accuracy** | Average exact match rate across individual schema keys | Layer A, B | Baseline target: >70% |
| **Deadlock Rate** | Decoder encounters state where no valid tokens exist | Layer B | Strictly 0.0% |
| **Budget Exhaustion Rate** | Generation reaches `max_tokens` before emitting `}` or `<|endoftext|>` | Layer A, B | Strictly <10% at 64 tokens |
| **Anti-Attractor Compliance** | Output contains 0 occurrences of Python keywords (`import`, `def`, `doctest`, `__main__`) outside string values | Layer A, B | 100% |

---

## 4. Dataset Isolation and Anti-Leakage Rules

To ensure that reported evaluation metrics reflect true generalization rather than pattern memorization, the following controls are mandatory:

### 4.1 Template-Family Disjointness
1. Every training and validation record must be tagged with a `template_family_id` (e.g., `health_check_v1`, `telemetry_v2`, `abstention_v1`).
2. **Zero Template Sharing**: No `template_family_id` present in `train.jsonl` may appear in `val.jsonl` or `test.jsonl`.
3. Syntactic variation (e.g., shifting instruction position, altering delimiter phrasing) must be partitioned at the family level.

### 4.2 Entity and Context Disjointness
1. Entity values (e.g., system IDs, node names, usernames, timestamps) used in validation context passages must never appear in training context passages.
2. Exact string target matching between train and val splits is strictly prohibited (`train_targets & val_targets == empty_set`).

### 4.3 Held-Out Test Split Protection
1. `test.jsonl` remains completely frozen and held out.
2. Neither development audits, hyperparameter searches, nor architecture adjustments may access or inspect `test.jsonl`.
3. Formal benchmark runs on `test.jsonl` may only be executed once an SFT V3 candidate model has officially passed all validation gates under this protocol.

---

## 5. Checkpoint Comparison Protocol

Any evaluation comparing a candidate model against a baseline must satisfy:

1. **Exact Baseline Identification**:
   - Canonical Pretrained Baseline: Step-10,000 checkpoint (`training_checkpoint_step_10000.pt`, SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
   - SFT Pilot Reference: SFT V2 candidate checkpoint (`sft_best_checkpoint.pt`, SHA-256: `436bdb89e62e0ef461e511b0ac410d01368ce626569b64c27d803ec97d953bd6`).
2. **Deterministic Sampling**:
   - Temperature = 0.0 (greedy decoding).
   - Fixed random seeds for any stochastic tie-breaking (`seed=42`).
3. **Execution Verification**:
   - Environment: Python virtual environment (`.\.venv\Scripts\python.exe`).
   - Hardware Backend: CPU-first reference execution (`torch.device("cpu")`), single-thread or fixed CPU affinity.
   - Verified Tokenizer: `vocab.json` (`db4d6119...`), `merges.json` (`ada4b1dc...`).

---

## 6. Acceptance Criteria for SFT V3 Success

A future SFT V3 training run shall be deemed successful **only if all of the following conditions are met**:

1. **No Pseudo-Improvements**: A decrease in validation cross-entropy loss **alone does NOT constitute a success**. The candidate must demonstrate measurable improvements on downstream task accuracy.
2. **Native JSON Emergence (Layer A)**:
   - Valid JSON syntax rate without constraints must improve from **0.0% to >= 25.0%** on held-out validation records.
   - Code-attractor contamination must decrease from **85.0% to <= 10.0%**.
3. **Constrained Task Accuracy (Layer B & C)**:
   - Under schema constraints, Exact Task Match (EM) must achieve **>= 75.0%** on grounded extraction tasks (compared to 0.0% in SFT V2).
   - Zero decoder deadlocks (`0%`) across all validation prompts.
4. **Generalization Over Memorization**:
   - Evaluation on unseen entity contexts must show >= 70% per-field accuracy, proving the model is extracting from the prompt rather than outputting memorized training values.
