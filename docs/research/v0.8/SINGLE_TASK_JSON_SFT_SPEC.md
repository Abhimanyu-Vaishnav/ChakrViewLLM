# CHAKRVIEW v0.8: CONTROLLED SINGLE-TASK JSON-SFT SPECIFICATION
## Research & Experiment Gate: Single-Task Structural JSON Synthesis Readiness

- **Status**: PROPOSED — PENDING FORMAL HUMAN APPROVAL
- **Execution Authority**: NONE (Training, optimizer steps, dataset generation, and weight modification are strictly unauthorized).
- **Target Architecture**: ChakrMicro (3,443,136 unique parameters, 6 layers, $d_{\text{model}}=192$, $n_{\text{heads}}=6$, $d_{\text{ff}}=768$, max_seq_len=256).
- **Canonical Parent Model**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`  
  - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Canonical Tokenizer**: `data/experiments/vocab_4096` (Vocab: 4,096 tokens)  
  - `vocab.json` SHA-256: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`  
  - `merges.json` SHA-256: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`

---

## 1. Scientific Objective & Precise Capability Definition

### 1.1 Core Scientific Question
Can an ultra-compact 3.44M-parameter neural core (ChakrMicro) learn and retain strict syntactic JSON output constraints and correct value extraction when instruction-following training is isolated to a single homogeneous structural task, or is the capacity ceiling insurmountable at this parameter scale?

### 1.2 Precise Capability Definition: Fixed-Key Generalization to Unseen Values
The proposed task evaluates **fixed-key schema compliance and in-context value binding**, NOT the invention of arbitrary new JSON schemas or unseen keys:
- The schema is fixed across all examples: `{"status": <string>, "code": <integer>}`.
- The learning objective is:
  1. Producing parseable JSON syntax (`{`, `"`, `:`, `,`, `}`).
  2. Adhering to the invariant two-key schema (`"status"` as string, `"code"` as integer).
  3. Extracting and binding the correct entity values from context into the appropriate JSON fields on unseen component entities, unseen status descriptors, and unseen integer codes.
  4. Generalizing across a limited set of prompt instruction variants.

### 1.3 Isolated Hypothesis
When trained on a single-task distribution of two-key JSON synthesis with 30% base continuation replay:
1. The model will avoid the multi-task attractor collapse observed in Pilot A and Pilot B (which mixed 7 conflicting categories).
2. The model will demonstrate a statistically significant increase in valid JSON syntax and exact schema adherence over the canonical baseline parent.
3. The base continuation capability will remain stable without immediate `<EOS>` degeneration on replay prompts.

---

## 2. Hierarchical Capability Decomposition & Scoring

To prevent ambiguous evaluation, capabilities are split into strict hierarchical tiers:

1. **JSON Syntax Parseability**: The output string parses under `json.loads()` without syntax error.
2. **Exact Schema Adherence**: The parsed JSON is a dictionary containing exactly the keys `{"status", "code"}` (no missing keys, no extra hallucinated keys), where `isinstance(obj["status"], str)` and `isinstance(obj["code"], int)`.
3. **Field-Level Value Correctness**:
   - `status_match`: `obj["status"] == expected_status`
   - `code_match`: `obj["code"] == expected_code`
4. **Full Semantic Task Success**: The object parses, satisfies exact schema adherence, AND achieves both `status_match` and `code_match`. A syntactically valid JSON object that contains incorrect values is scored as **Syntax Pass, Semantic Fail**.
5. **Replay Continuation Preservation**: The model maintains non-zero entropy and continues unformatted text without premature `<EOS>` or repetitive loops on held-out continuation prompts.

---

## 3. Dataset Generation Specification & Leakage Controls

The experiment specifies 200 total curated records across three disjoint splits. **The dataset does not yet exist and cannot be generated until explicit human authorization.**

### 3.1 Split Allocation
- **Training Set (`train.jsonl`)**: 140 pairs (100 JSON task pairs + 40 Continuation Replay pairs).
- **Validation Set (`val.jsonl`)**: 30 pairs (20 JSON task pairs + 10 Continuation Replay pairs).
- **Held-Out Test Set (`test.jsonl`)**: 30 pairs (20 JSON task pairs + 10 Continuation Replay pairs).

### 3.2 Prompt & Response Contract
- **Prompt Structure**:
  ```text
  Instruction: {INSTRUCTION_TEMPLATE}
  Context: {SYSTEM_CONTEXT}
  Response:
  ```
- **Target Response**:
  ```text
  {"status": "<STATUS_VALUE>", "code": <CODE_VALUE>}
  ```
- **Replay Structure**:
  - Prompt: Pretraining corpus snippet (20–40 tokens).
  - Target: Natural continuation (20–40 tokens), with `append_eos=False`.
  - Content Domain: Replay prompts must be drawn from general language pretraining chunks that do not mention JSON, schemas, brackets, or system telemetry keys to prevent leakage into the JSON evaluation.

### 3.3 Leakage Prevention & Entity Partitioning
1. **Unseen Component Entities in Test**:
   - Component names in Test (e.g., `cryo_pump_omega`, `optical_relay_12`) must be strictly disjoint from Training and Validation sets.
   - Status strings in Test (e.g., `"standby_degraded"`, `"calibrating"`) must not appear in Training.
   - Code integers in Test must be sampled from a reserved disjoint range (e.g., Test: `800-899`, Train: `100-699`, Val: `700-799`).
2. **Prompt Phrasing Diversity & Acknowledged Limitation**:
   - Exactly 4 instruction framing templates are specified:
     - T1: `"Extract the system status and status code as a JSON object with keys \"status\" and \"code\"."`
     - T2: `"Parse the telemetry reading below into JSON format with \"status\" and \"code\" fields."`
     - T3: `"Convert the following component report into a JSON dictionary containing \"status\" and \"code\"."`
     - T4: `"Given the system state description, return a JSON object structured as {\"status\": string, \"code\": integer}."`
   - *Limitation Disclosure*: Four templates provide limited phrasing variation rather than broad natural language generalization. Passing across these 4 templates demonstrates structural robustness against slight template perturbation, but cannot support claims of general instruction following.
3. **Duplicate Detection & Manifest Verification**:
   - Automated check: Hash-based exact duplicate detection and normalized string matching (lowercase, whitespace-collapsed) across prompt bodies.
   - Pre-training gate requirement: Dataset creation script must emit `manifest.json` recording SHA-256 hashes of `train.jsonl`, `val.jsonl`, and `test.jsonl`, along with an automated programmatic assertion confirming 0% entity overlap between splits.
4. **Data Isolation & Split Inaccessibility**:
   - `test.jsonl` must be isolated during training. SFT training code and checkpoint selection algorithms are strictly prohibited from loading or reading `test.jsonl`.
   - Checkpoint selection is governed exclusively by validation loss on `val.jsonl`.

---

## 4. Evaluation Suite & Baseline Comparison

Every metric will be evaluated identically on:
1. **Parent Baseline**: Canonical Step 10,000 checkpoint (`training_checkpoint_step_10000.pt`).
2. **Candidate Checkpoint**: Best validation loss checkpoint from the Single-Task SFT run.

### 4.1 Quantitative Metrics & Exact Denominators

| Metric | Numerator | Denominator |
| :--- | :--- | :--- |
| **JSON Parse Rate** | Count of outputs where `json.loads(text)` succeeds | Total Held-Out JSON Prompts ($N=20$) |
| **Schema Match Rate** | Count of parsed JSON dicts with exact keys `{"status", "code"}` and correct types (`str`, `int`) | Total Held-Out JSON Prompts ($N=20$) |
| **Status Value Match** | Count of outputs where parsed `status` equals ground truth string | Total Held-Out JSON Prompts ($N=20$) |
| **Code Value Match** | Count of outputs where parsed `code` equals ground truth integer | Total Held-Out JSON Prompts ($N=20$) |
| **Full Task Success** | Count of outputs where both `status` and `code` match ground truth in valid schema | Total Held-Out JSON Prompts ($N=20$) |
| **Empty Output Rate** | Count of outputs with 0 decoded characters or whitespace-only | Total Held-Out JSON Prompts ($N=20$) |
| **Immediate EOS Rate** | Count of outputs whose first generated token was `<EOS>` (token 1) | Total Held-Out JSON Prompts ($N=20$) |
| **Repetition / Loop Rate** | Count of outputs terminating due to 3-gram repetition threshold | Total Held-Out JSON Prompts ($N=20$) |
| **Length Truncation Rate** | Count of outputs reaching `max_new_tokens` (64) without `<EOS>` | Total Held-Out JSON Prompts ($N=20$) |
| **Replay Non-Empty Rate** | Count of replay outputs with non-empty decoded text | Total Replay Prompts ($N=10$) |
| **Replay Meaningful Continuation**| Count of replay outputs continuing text without degenerate 1-token repetition (repetition ratio $< 0.40$) | Total Replay Prompts ($N=10$) |

### 4.2 Decoding Parameters (Held Strict & Consistent)
- Decoding mode: Greedy deterministic ($T=0.0$, top_k=None).
- `max_new_tokens`: 64.
- Stop tokens: `[1]` (`<EOS>`).

### 4.3 Baseline Status: Pending Frozen Dataset Generation
- **Parent Baseline Status**: **PENDING FROZEN TEST SET GENERATION**.
- *Rationale*: Although the canonical parent checkpoint was evaluated on historical multi-task benchmarks (yielding 0/4 on general structured JSON in Pilot B), the exact 20 two-key JSON test prompts for this experiment have not yet been generated.
- *Requirement*: Prior to candidate training, once dataset generation is authorized, the canonical parent checkpoint must be evaluated on the frozen `test.jsonl` under the exact evaluation harness. The resulting predictions, token IDs, and scores must be committed to `experiments/research_v08_json_sft/parent_baseline_results.json`.

### 4.4 Acceptance Criteria for Candidate Success
To accept the candidate as demonstrating a genuine structural capability advancement:
1. **JSON Parse Rate**: $\ge 50.0\%$ (at least 10/20 valid JSON objects on held-out test prompts, compared to measured parent baseline).
2. **Schema Match Rate**: $\ge 40.0\%$ (at least 8/20 matching the exact two-key contract).
3. **Full Semantic Task Success**: $\ge 25.0\%$ (at least 5/20 extracting correct status and code).
4. **Replay Meaningful Continuation**: $\ge 80.0\%$ (at least 8/10 non-degenerate continuations).
5. **No Catastrophic Regression**: Zero regression on general validation perplexity ($< 10\%$ drift).

*Scientific Scope & Limitations*:
- Because $N=20$ for test prompts, passing these criteria provides **preliminary evidence of narrow structural adaptability**, not general reasoning or broad instruction following.
- If the model achieves high parse rate but 0% semantic accuracy, the result must be documented as "Syntactic Prior Acquisition without Contextual Value Binding".

---

## 5. Checkpoint, Training & Rollback Controls

### 5.1 Optimization Parameters
- Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-8}$, weight decay=0.01).
- Learning Rate: Peak $3.0 \times 10^{-5}$ with linear warmup (10 steps) and cosine decay to $3.0 \times 10^{-6}$.
- Batch Size: 4 (gradient accumulation steps: 1).
- Max Steps: 100.
- Validation Cadence: Every 10 steps on `val.jsonl`.
- Checkpoint Saving: Best validation checkpoint saved to `sft_single_task_json_best.pt`.
- Fixed Seed: 42 (torch, numpy, random).

### 5.2 Estimated Resource Limits
- Memory: Estimated $\le 1.5$ GB system RAM on CPU.
- Runtime: Estimated $\le 12$ minutes total on modern multi-core x86 CPU.

### 5.3 Abort Criteria
Execution will immediately abort if:
1. Validation loss increases for 3 consecutive evaluation cycles (divergence/overfitting).
2. Relative weight drift $\|\Delta W\| / \|W\| > 4.0\%$ across the network.
3. NaN or Inf encountered in loss or gradient norms.
4. Total execution exceeds 15 minutes on CPU.

### 5.4 Rollback & Immutability Guarantee
1. Canonical parent checkpoint (`experiments/batch16_21_combined/...`) and tokenizer files are mounted read-only and never modified.
2. Candidate outputs are stored in a new isolated directory: `experiments/research_v08_json_sft/`.
3. If acceptance criteria are not met, the candidate is marked `EXPERIMENT_FAILED_CAPACITY_BOUND` and discarded, with zero changes to production runtime.
