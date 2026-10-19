# SFT V3 PILOT EXECUTION & EVALUATION PROTOCOL
**Status:** Candidate Protocol for Separately Authorized Training Pilot  
**Hardware Target:** CPU-First Reference Execution  
**Model:** ChakrMicro (3,443,136 Trainable Parameters, Tied Embeddings)  
**Safety Gate:** Specification Only (ZERO Training Authorized by this Document)

---

## 1. Candidate Pilot Configuration

If and when authorized by the operator via an explicit flag (`--execute-authorized-training`), the SFT V3 pilot shall execute under the following frozen hyperparameters:

```json
{
  "experiment_name": "sft_v3_grounded_pilot",
  "base_checkpoint": {
    "path": "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
    "sha256": "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
  },
  "dataset_paths": {
    "train_file": "experiments/sft_v3_dataset_builder/train.jsonl",
    "train_sha256": "57affaa613eaf0e31be752bc680dddcd3c748e1cee3752942db93e8dea270804",
    "val_file": "experiments/sft_v3_dataset_builder/validation.jsonl",
    "val_sha256": "4c4c77d5db4884e06ed4e06550bd4badba6a4793f37a1d5a38d0c9512cbc1cf8"
  },
  "tokenizer": {
    "directory": "data/experiments/vocab_4096",
    "vocab_sha256": "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a",
    "merges_sha256": "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7"
  },
  "hyperparameters": {
    "optimizer": "AdamW",
    "pilot_steps": 100,
    "batch_size": 4,
    "learning_rate": 3e-5,
    "min_learning_rate": 1e-6,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "seed": 42,
    "val_interval": 20,
    "save_interval": 50,
    "save_best_checkpoint": true,
    "max_seq_len": 512,
    "strip_prompt_delimiters": false,
    "device": "cpu"
  },
  "output_directory": "experiments/sft_v3_pilot_readiness/pilot_run"
}
```

---

## 2. Capability Evaluation Framework

A future SFT V3 pilot checkpoint must be evaluated against the frozen Step-10,000 baseline across three capability layers:

### Layer A: Native Unconstrained Generation (Greedy, Temp=0.0)
- **Valid JSON Syntax**: Output must parse via `json.loads` without schema enforcement. Target: improve from **0.0% to >= 25.0%**.
- **Code Attractor Contamination**: Suppress `{"__main__":\n import doctest` rate from **83.3% to <= 15.0%**.
- **Native Value Accuracy**: Field values extracted correctly from context without runtime constraints.

### Layer B: Schema-Constrained Decoding (Prefix Trie Scaffolding)
- **Syntax Completion**: Valid JSON syntax under schema constraints must achieve **>= 80.0%** (up from 76.7% in dry-run baseline).
- **Deadlock Freedom**: Decoder must maintain **0.0% deadlocks**.
- **Budget Exhaustion**: Maximum tokens capped at 64; budget exhaustion rate must remain <= 15.0%.

### Layer C: Semantic Task Correctness & Grounding
- **Exact Task Match (EM)**: Ground truth JSON equality (`parsed == expected_answer`). Target: improve from **0.0% to >= 50.0%** on held-out validation contexts.
- **Per-Field Accuracy**: Correct extraction of status, code, metric, value, feature, enabled, name, and count.
- **Abstention Accuracy**: Correct output of `null` and `error` when context lacks facts.

### Replay & Catastrophic Forgetting Probes
- **Immediate-EOS Probes**: Must achieve **4/4 passing** (zero early `<|endoftext|>` completions on benchmark replay prompts).
- **Active Continuation Length**: Average generated token length on natural text prompts must remain **>= 35.0 tokens**.

---

## 3. Explicit Pilot Success and Failure Criteria

### Success Criteria (Pilot Declared Successful):
1. **Downstream Task Generalization**: Exact Task Match on validation contexts reaches **>= 40.0%** under constrained decoding (or >= 20.0% unconstrained).
2. **Code Attractor Reduction**: Python boilerplate outputs drop by at least **50 percentage points** relative to baseline.
3. **No Forgetting**: All 4 canonical replay probes pass with active continuation.
4. **Zero Deadlocks**: Schema prefix trie encounters 0 deadlocks across validation.

### Failure Criteria (Requires Rollback & Diagnostic Review):
1. **Pseudo-Improvement**: Validation loss decreases, but Exact Task Match remains 0.0%.
2. **Immediate-EOS Collapse**: 1 or more replay probes terminate immediately with `<|endoftext|>`.
3. **Severe Attractor Persistence**: Model continues generating doctest strings on > 50% of validation prompts.
4. **NaN or Gradient Explosion**: Training loss destabilizes or outputs NaN.

---

## 4. Rollback and Safety Rules

If a failure condition is triggered during an authorized run:
1. Immediately stop the training job.
2. Checkpoints saved under `pilot_run/` must be marked `FAILED_EVALUATION`.
3. Canonical Step-10,000 checkpoint and tokenizer artifacts remain untouched.
4. No modified weights may be copied into `release/` or `runtime/`.
