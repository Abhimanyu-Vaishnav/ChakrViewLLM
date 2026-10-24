# Next Controlled Experiment Proposal: Exp20

**Proposed Experiment**: `exp20_single_task_contextual_binding_cpu_400`  
**Status**: Proposal Only — Do Not Execute Without Authorization  
**Author**: Antigravity ML Diagnostic Specialist  
**Date**: October 24, 2026  

---

## 1. Executive Summary & Research Question

### Primary Research Question
Does isolating fine-tuning to a single, grounded task family (`context_extraction`) enable ChakrMicro (3.44M parameters) to learn semantic entity binding and non-zero task accuracy, or does the 3.44M architecture inherently fail contextual copying regardless of task interference?

### Primary Hypothesis ($H_1$)
Removing cross-task interference from heterogeneous categories (arithmetic, JSON, QA, code) and dedicating all 400 optimizer steps to grounded contextual extraction will allow the model's attention heads to bind target entities from the context into the response, achieving $\ge 15.0\%$ task accuracy on held-out context extraction records.

### Null Hypothesis ($H_0$)
Even when trained exclusively on contextual extraction, ChakrMicro will fail to achieve $\ge 5.0\%$ task accuracy on held-out records, demonstrating that 3.44M parameters is below the architectural threshold required for cross-attention copy-paste mechanisms in this tokenizer space.

---

## 2. Experimental Design & Principal Intervention

### Single Principal Intervention
**Task Isolation**: Rather than diluting 1,200 training steps across 8 disparate categories (150 examples each), Exp20 trains exclusively on **Contextual Entity Extraction** (`context_extraction`).
- No arithmetic.
- No JSON or code formatting.
- No factual trivia from ungrounded memory.
- Every training example provides an explicit, short context and asks to extract a specific entity (e.g. `"In Orbital Expedition X, who is assigned as Y?"`).

### Comparison Baseline
- **Exp18 Step 240**: `context_extraction` accuracy = **0.0% (0/20)**, EOS rate = 50.0% (10/20).
- **Exp19 Step 340**: `context_extraction` accuracy = **0.0% (0/20)**, EOS rate = 55.0% (11/20).

---

## 3. Dataset & Split Specification

1. **Training Set**:
   - 400 deterministically generated, verified contextual extraction records.
   - Clean, diverse entity names, roles, and orbital missions.
   - 5 rotating syntactic response frames to prevent prefix collapse ($\le 25\%$ concentration).
2. **Validation Set**:
   - 50 isolated contextual extraction records (disjoint entity names and mission numbers).
3. **Held-Out Test Set**:
   - The exact 20 frozen `context_extraction` records from the Exp18/Exp19 frozen test manifest (`EXP18_TEST_CONTEXT_EXTRACTION_0000` to `0019`).
4. **Contamination & Leakage Checks**:
   - Zero entity or mission overlap between train, val, and test splits. Verified by bitwise string matching.

---

## 4. Training Configuration & Hyperparameters

- **Base Checkpoint**: Isolated copy of canonical `training_checkpoint_step_10000.pt` (`SHA-256: 96357705...`).
- **Device**: CPU (`torch.device("cpu")`).
- **Steps**: Exactly 400 optimizer steps.
- **Batch Size**: 2.
- **Learning Rate**: $5 \times 10^{-5} \to 5 \times 10^{-6}$ (Cosine decay).
- **Weight Decay**: 0.01.
- **Gradient Clipping**: 1.0.
- **Evaluation Interval**: Every 20 steps.
- **Checkpoint Selection**: Lowest validation loss on the 50-record validation set.
- **Estimated Runtime**: ~25 seconds on CPU.

---

## 5. Metrics & Explicit Go / No-Go Decision Criteria

| Metric | Baseline (Exp19) | Exp20 Target | Measurement Method |
| :--- | :---: | :---: | :--- |
| **Context Extraction Task Accuracy** | **0.0%** (0/20) | **$\ge 20.0\%$** ($\ge 4/20$) | Hardened substring/entity match from `TaskEvaluator` |
| **EOS Completion Rate** | 55.0% (11/20) | $\ge 80.0\%$ ($\ge 16/20$) | Detection of token id 1 |
| **Cyclical Repetition Rate** | 55.0% (11/20) | $\le 15.0\%$ ($\le 3/20$) | `"city of the city"` / runaway tokens |
| **Average Query Latency (CPU)** | ~280 ms | $\le 250$ ms | Per-record wall clock timer |

### Decision Gate
- **GO (Task Interference Proven)**: If accuracy $\ge 20.0\%$, task interference is confirmed as the primary bottleneck of Exp18/Exp19. We proceed to curriculum learning / modular task heads.
- **NO-GO (Capacity Limit Proven)**: If accuracy remains $0.0\%$, model capacity / copy-head failure at 3.44M parameters is established empirically. We then justify architectural scaling to ChakrMini (~15M parameters).

---

## 6. What This Experiment Cannot Prove
- It cannot prove that the model can do general reasoning or multi-step logic.
- It cannot prove that the model can format arbitrary JSON.
- It is strictly an isolation probe to resolve Hypothesis H1 vs H3.
