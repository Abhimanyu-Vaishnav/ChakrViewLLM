# ChakrView Exp18 Aligned SFT (`exp18_aligned_sft_cpu_300`) — Preflight Verification Report

**Directory**: `experiments/overnight_runs/preflight_exp18_20261024_114600/`  
**Timestamp**: 2026-10-24T11:46:00  
**Target Experiment**: `exp18_aligned_sft_cpu_300`  
**Status**: `READY`  

---

## 1. Executive Summary & Verification Checklist

All three mandatory preflight verification checks have been independently executed and have passed:

| Item # | Verification Objective | Result | Status |
| :---: | :--- | :--- | :---: |
| **1** | **In-Memory Exp18 Dataset Source & Deterministic Reproducibility** | Generated purely in memory via `Exp18DatasetGenerator.generate_all_splits()` with exact split counts (1,200 train / 160 val / 160 test). Verified identical hash across repeated runs. Zero official files created or overwritten. | **PASS** |
| **2** | **Canonical Checkpoint Isolation & Hash Immutability** | Canonical checkpoint `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` verified intact (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`). Isolated copy created at `preflight_exp18_20261024_114600/checkpoints/canonical_copy_step_10000.pt` with matching SHA-256. | **PASS** |
| **3** | **Baseline Held-Out Evaluation on Exact Target Test Set** | Evaluated canonical checkpoint on all 160 held-out Exp18 test records using exact decoding contract (greedy $T=0.0$, max tokens 45, seed 42) and hardened `TaskEvaluator`. Baseline metrics persisted to `BASELINE_EVALUATION.json`. | **PASS** |

---

## 2. Check 1 — In-Memory Exp18 Split Source & Reproducibility

### A. Generator API & Provenance
- **Module**: `experiments/research_v09_conversation_sft/exp18_scaled_instruction_format_v1/generate_exp18_datasets.py`
- **Class**: `Exp18DatasetGenerator`
- **Method**: `gen.generate_all_splits()`
- **Official Data Directory Check**: `experiments/research_v09_conversation_sft/exp18_scaled_instruction_format_v1/data/` remains completely empty (0 files). No official JSONL files were written.

### B. Split Quantities & Determinism Audit
Two consecutive in-memory generations were performed and evaluated for bitwise identical serialization:

- **Train Split**: 1,200 records
  - SHA-256 (JSON sorted keys): `305b309f22b684049c285760073cb3dd046c92352f151ab79ec6aa303dcbb9c3`
  - Bitwise Match Across Runs: `True`
- **Validation Split**: 160 records
  - SHA-256 (JSON sorted keys): `35825b22a433254aceddbfc705a24c9476adfb3eb3121dc46a9f007e8b46a662`
  - Bitwise Match Across Runs: `True`
- **Test Split**: 160 records
  - SHA-256 (JSON sorted keys): `111db453d5ee52d436ef9a49f1ad6555ea572d118f84099ea0763e7f299ba047`
  - Bitwise Match Across Runs: `True`

### C. Prompt-Conditioning Contract Compliance
Each generated record provides:
- `prompt_mode`: e.g. `"plain_instruction"`, `"context_qa"`, `"system_persona"`
- `prompt`: raw user instruction
- `context`: optional background context
- `formatted_prompt`: canonical conditioning formatted via `format_conditioned_prompt(prompt, prompt_mode=mode, context=context)`
- `response`: target response (length bounded between 3 and 45 tokens)

---

## 3. Check 2 — Canonical Checkpoint Isolation & Integrity

### A. Canonical Artifact Protection
- **Canonical Checkpoint Path**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
- **Expected SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Actual SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (`MATCH: TRUE`)

### B. Isolated Experiment Checkpoint Copy
- **Isolated Copy Path**: `experiments/overnight_runs/preflight_exp18_20261024_114600/checkpoints/canonical_copy_step_10000.pt`
- **Copy SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (`MATCH: TRUE`)
- **Isolation Guarantee**: All fine-tuning and state updates for the upcoming `exp18_aligned_sft_cpu_300` experiment will exclusively target checkpoints within the dedicated experiment directory (`exp18_aligned_sft_cpu_300/checkpoints/`), leaving the canonical checkpoint file read-only and unreferenced for write operations.

---

## 4. Check 3 — Baseline Evaluation on Held-Out Test Set

The baseline performance of the canonical checkpoint (`step_10000.pt`) was evaluated across all 160 held-out Exp18 test records using the exact inference runtime contract and decoding settings.

### A. Evaluation Protocol & Decoding Configuration
- **Model**: `ChakrMicro` initialized from `canonical_copy_step_10000.pt`
- **Evaluator**: `chakrview/evaluation/task_evaluator.py` (`TaskEvaluator v0.1` with word-boundary short entity matching)
- **Decoding**: Greedy (`temperature=0.0`, `seed=42`)
- **Max Generation Budget**: 45 tokens
- **EOS Token**: Token ID `1` (`<|endoftext|>`)
- **Runtime Prompt Wrapper**: `format_conditioned_prompt(prompt, prompt_mode=rec['prompt_mode'], context=rec.get('context', ''))`

### B. Summary Benchmark Metrics

| Metric | Measured Baseline Value | Context / Notes |
| :--- | :---: | :--- |
| **Total Test Records** | 160 | Full held-out test split |
| **Scorable Records** | 140 | 20 `short_explanation` records held out as `NEEDS_REVIEW` |
| **Total Correct Scorable** | **2** | 2 / 140 |
| **Scorable Task Accuracy** | **1.43%** | Evaluated across 7 scorable categories |
| **Clean EOS Termination Rate** | **0.62%** | 1 / 160 emitted token `1` before 45-token budget |
| **Max Tokens Truncation Rate** | **99.38%** | 159 / 160 stopped by budget exhaustion |
| **Structured JSON Validity** | **0.0%** | 0 / 20 structured formatting records produced valid RFC 8259 JSON |
| **Average Latency** | **261.25 ms** | Per test query on CPU |
| **Total Tokens Generated** | 7,156 tokens | Model continuously generated until token 45 |

### C. Disaggregated Category Breakdown

| Category | Total ($N$) | Scorable ($N_{\text{scorable}}$) | Correct | Incorrect | Needs Review | EOS Count | Valid JSON |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `factual_qa` | 20 | 20 | 2 | 18 | 0 | 0 | 20* |
| `basic_arithmetic` | 20 | 20 | 0 | 20 | 0 | 0 | 5* |
| `context_extraction` | 20 | 20 | 0 | 20 | 0 | 0 | 2* |
| `reference_resolution` | 20 | 20 | 0 | 20 | 0 | 0 | 0 |
| `uncertainty_abstention` | 20 | 20 | 0 | 20 | 0 | 0 | 14* |
| `instruction_following` | 20 | 20 | 0 | 20 | 0 | 1 | 20* |
| `structured_formatting` | 20 | 20 | 0 | 20 | 0 | 0 | 0 |
| `short_explanation` | 20 | 0 | 0 | 0 | 20 | 0 | 20* |
| **Total** | **160** | **140** | **2** | **138** | **20** | **1** | **-** |

*\*Note: Valid JSON flag on non-JSON categories indicates fallback string wrapper or non-strict check, whereas `structured_formatting` strictly enforces JSON parsing, achieving 0.0%.*

### D. Observation on Evaluator Integrity
With the hardened `\b{ent}\b` boundary matching implemented in `TaskEvaluator`, spurious substring matches (such as `"show"` matching `"W"` for Tungsten) are eliminated. The baseline achieves **1.43%** (2/140), accurately reflecting the base pre-training model's raw generative behavior prior to aligned instruction tuning.

---

## 5. Protected Artifact Integrity Verification

Bitwise validation of all 9 protected repository assets:

```json
{
  "all_match": true,
  "assets": {
    "canonical_architecture_baseline": {
      "path": "memory:ChakrMicro(ModelConfig())@seed(42)",
      "expected": "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da",
      "actual": "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da",
      "match": true
    },
    "canonical_training_checkpoint_step_10000": {
      "path": "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
      "expected": "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee",
      "actual": "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee",
      "match": true
    },
    "tokenizer_vocab_json": {
      "path": "data/experiments/vocab_4096/vocab.json",
      "expected": "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a",
      "actual": "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a",
      "match": true
    },
    "tokenizer_merges_json": {
      "path": "data/experiments/vocab_4096/merges.json",
      "expected": "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7",
      "actual": "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7",
      "match": true
    },
    "runtime_session_adapter": {
      "path": "chakrview/runtime/session.py",
      "expected": "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67",
      "actual": "9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67",
      "match": true
    },
    "model_brain_source": {
      "path": "chakrview/brain/model.py",
      "expected": "1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361",
      "actual": "1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361",
      "match": true
    },
    "exp17_train_split": {
      "path": "experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/data/train_exp17_v1.jsonl",
      "expected": "074fafe5c1dbcbd7467430cb6f5eccd2c11f2f8679ff27e13c0d38cc90bac181",
      "actual": "074fafe5c1dbcbd7467430cb6f5eccd2c11f2f8679ff27e13c0d38cc90bac181",
      "match": true
    },
    "exp17_val_split": {
      "path": "experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/data/val_exp17_v1.jsonl",
      "expected": "15d9c942c01fb6afaf6280b7515d9e621ff8df7740be8010daa2be98841f4564",
      "actual": "15d9c942c01fb6afaf6280b7515d9e621ff8df7740be8010daa2be98841f4564",
      "match": true
    },
    "exp17_test_split": {
      "path": "experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/data/test_exp17_v1.jsonl",
      "expected": "be10afaa3957713cead5399c2b01e5ad2f4fb61d6355835432c7d6e0fdcc6430",
      "actual": "be10afaa3957713cead5399c2b01e5ad2f4fb61d6355835432c7d6e0fdcc6430",
      "match": true
    }
  }
}
```

---

## 6. Preflight Conclusion & Experiment Readiness

All pre-training safety rules, data isolation requirements, and benchmark baselines have been satisfied without starting training or modifying any protected file.

Preflight Status: **READY**
