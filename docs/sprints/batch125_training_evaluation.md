# Batch 125 — Controlled Training and Free-Running Evaluation

## Executive Summary

Batch 125 executed a rigorously controlled training and evaluation comparison on the newly constructed prompt-diverse two-digit benchmark (`experiments/batch123_126_prompt_diverse_multidigit/`).

Two distinct model initializations were evaluated under strictly identical conditions:
- **Arm A (Canonical Base Initialization)**:
  - Base weights: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (SHA-256: `96357705a683...`).
  - Saved Checkpoint: `multidigit_armA_canonical_base_step200.pt`
  - Checkpoint SHA-256: `b3ede17e9222eca0989db008fb0151001e7c79cbb5522cc0c3d1e1736f24e86b`
- **Arm B (Copy-Task Pre-Trained Initialization)**:
  - Base weights: `experiments/batch107_110_operand_copy_benchmark/checkpoints/copy_task_sft_step150.pt` (SHA-256: `e84b2ea9b633...`).
  - Saved Checkpoint: `multidigit_armB_copy_init_step200.pt`
  - Checkpoint SHA-256: `54486e88bae0b9f131a84868de2714f62095f0293ee13f007086466465461929`

---

## 1. Experimental Protocol & Hyperparameters

- **Training Records**: 100 records (70 tool additions across 60 distinct two-digit pairs and 8 templates, 30 negative controls).
- **Evaluation Records**: 45 records (35 tool requests, 10 negative controls).
- **Training Steps**: 200 steps.
- **Batch Size**: 4.
- **Optimizer**: AdamW ($lr=3\times 10^{-4}$, weight decay 0.01, gradient clipping 1.0).
- **Seed**: 42.
- **Decoding**: Free-running greedy decoding from cold-reloaded checkpoints; zero teacher forcing.

---

## 2. Evaluation Results Summary ($N = 45$)

| Evaluation Metric | Arm A (Canonical Base Init) | Arm B (Copy-Task Init) |
| :--- | :---: | :---: |
| **Tool Examples Evaluated ($N$)** | 35 | 35 |
| **Tool Parse Rate** | 71.4% (25/35) | 71.4% (25/35) |
| **Tool Schema Validity Rate** | 71.4% (25/35) | 71.4% (25/35) |
| **Tool Name Accuracy (`calculator`)** | 71.4% (25/35) | 71.4% (25/35) |
| **Operation Accuracy (`add`)** | 71.4% (25/35) | 71.4% (25/35) |
| **Operand A Accuracy** | 20.0% (7/35) | **34.3% (12/35)** |
| **Operand B Accuracy** | 11.4% (4/35) | **28.6% (10/35)** |
| **Exact-Call Accuracy** | **5.7% (2/35)** | **20.0% (7/35)** |
| **Allowlisted Execution Success** | 71.4% (25/35) | 71.4% (25/35) |
| **End-to-End Correctness** | **5.7% (2/35)** | **20.0% (7/35)** |
| **Output Collapse Rate** | 28.6% | 28.6% |
| **Negative Controls Evaluated ($N$)** | 10 | 10 |
| **Total False-Positive Rate** | **0.0% (0/10)** | **0.0% (0/10)** |
| **Clean Refusal Rate** | **100.0% (10/10)** | **100.0% (10/10)** |

---

## 3. Subcategory Breakdown by Generalization Boundary

### Partition 1: Familiar Pairs with Held-Out (Unseen) Templates ($N = 10$)
Evaluates pairs seen during training, but framed with novel prompt phrasings:
- **Arm A (Canonical Base)**: Parse rate 50.0%, Exact call = **10.0% (1/10)**.
- **Arm B (Copy-Init)**: Parse rate 50.0%, Exact call = **50.0% (5/10)**.
  - On the held-out template `"Execute calculator addition: first={a}, second={b}."`, Arm B achieved **100% exact-call accuracy (5/5)**!
  - On the held-out template `"Determine arithmetic total for inputs: A is {a} and B is {b}."`, both models refused to trigger tool calls, emitting clean non-tool refusals.

### Partition 2: Unseen Pairs with Familiar Templates ($N = 15$)
Evaluates novel two-digit pairs framed with prompt templates seen during training:
- **Arm A (Canonical Base)**: Parse rate 100.0%, Exact call = **6.7% (1/15)**, A acc = 20.0%, B acc = 6.7%.
- **Arm B (Copy-Init)**: Parse rate 100.0%, Exact call = **13.3% (2/15)**, A acc = 33.3%, B acc = 33.3%.
  - Qualitative analysis reveals partial token binding: e.g., on prompt `[74, 53]`, Arm B predicted `a: 75, b: 53` (exact match on tens/units components); on `[79, 40]`, Arm B predicted `a: 79, b: 4`.

### Partition 3: Unseen Pairs with Held-Out Templates ($N = 10$)
The intersection of unseen pairs and unseen prompt syntax:
- **Arm A (Canonical Base)**: Exact call = **0.0% (0/10)**.
- **Arm B (Copy-Init)**: Exact call = **0.0% (0/10)**.
  - Zero double-generalization: combining novel two-digit multi-token numbers with novel prompt phrasings causes complete degradation of exact argument matching.

### Partition 4: Negative Controls & Out-of-Domain Requests ($N = 10$)
- **Arm A**: 100.0% clean refusal (10/10), 0.0% false positives.
- **Arm B**: 100.0% clean refusal (10/10), 0.0% false positives.
  - Tool gating remained 100% robust in both arms.
