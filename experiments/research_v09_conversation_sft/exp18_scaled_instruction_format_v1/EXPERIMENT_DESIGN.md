# ChakrView Exp18 — Scaled Instruction & Format Fine-Tuning Specification

**Experiment Identifier**: `exp18_scaled_instruction_format_v1`  
**Parent Directory**: `experiments/research_v09_conversation_sft/exp18_scaled_instruction_format_v1/`  
**Status**: DESIGN & GOVERNANCE STAGE ONLY (`READY_FOR_EXP18_PREFLIGHT`) — NO TRAINING EXECUTED  
**Author**: Antigravity ML Systems & Evaluation Agent  
**Date**: 2026-10-23  
**Target Model**: ChakrMicro v0.1 (3,443,136 parameters, ModelConfig(d_model=192, n_layers=6, n_heads=6, hidden_dim=512, max_seq_len=512))  
**Base Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`  
(Immutable Canonical Hash: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`)  
**Execution Device Policy**: Strict CPU-Only Local Execution (AMD64, PyTorch CPU, No external APIs/LLMs/remote engines)

---

## 1. Executive Summary & Problem Formulation

### 1.1 Empirical Baseline & Exp17 Findings
In Exp17 (`exp17_instruction_sft_v1`), Supervised Fine-Tuning on 320 compact prompt-response pairs established substantial progress in conversational turn-boundary mechanics:
1. **Turn Termination (EOS)**: Increased from 0.0% (baseline) to **45.0%** on the held-out test split, and from 5.0% to **60.0%** on the Normal Chat Gate v0.1 suite.
2. **Repetition Suppression**: Reduced mean trigram repetition deficit from 46.2% to **13.3%** on Normal Chat Gate v0.1.
3. **Empty Output Reduction**: Decreased from 27.5% to **5.0%** on the held-out test split.

However, independent metric reconciliation verified via Task Evaluator v0.1 confirmed a stark reality:
- **Scorable Task Accuracy**: Remained at **0.0%** (0 / 35 scorable test items correct).
- **Structured JSON Formatting**: Remained at **0.0%** (0 / 5 test items emitted valid RFC 8259 JSON).
- **Mechanistic Root Cause**: The model learned the *surface form* of terminating turns (`<EOS>`) and suppressing loops, but because Exp17 used only 40 training examples per category (with synthetic identifiers like `Apollo_train_0`), the small 3.44M model failed to generalize token prediction into factual recall, arithmetic calculation, context extraction, or valid syntax generation.

### 1.2 Mission Objective for Exp18
Exp18 scales and refines the training distribution to bridge the gap between turn termination and task correctness:
- Scale training dataset to **1,200 curated, verified examples** (150 examples per category across 8 categories).
- Introduce explicit syntax scaffolding for structured JSON formatting.
- Diversify lexical templates and entity slots to prevent formulaic pattern memorization.
- Retain strict prompt-masked autoregressive loss ($\mathcal{L}_{\text{CE}}$ computed strictly on target response and `<EOS>`).
- Preserve the canonical checkpoint and architecture completely untouched.
- Subject all checkpoints to deterministic evaluation via Task Evaluator v0.1.

---

## 2. Experimental Hypotheses & Falsification Conditions

| Hypothesis | Mechanism | Primary Metric | Falsification / Failure Threshold |
| :--- | :--- | :--- | :--- |
| **H1: Format Syntax Acquisition** | Scaling JSON examples from 40 to 150 with varied keys and valid schema structures teaches bracket pairing and key-value syntax. | Valid RFC 8259 JSON parsing rate on held-out test set (`structured_formatting`). | $< 20.0\%$ valid JSON objects generated on test split. |
| **H2: Extractive Context Grounding** | Increased context-question pairs (150 train) with diverse span lengths forces cross-attention from question to context tokens. | Exact entity extraction accuracy on held-out test set (`context_extraction`). | $0.0\%$ task accuracy on context extraction. |
| **H3: Arithmetic Generalization** | Broader operand ranges and naturalistic phrasing decouple calculation from the fixed pretraining `Step-by-Step Solution:` prior. | Word-boundary numerical match on held-out test set (`basic_arithmetic`). | $0.0\%$ task accuracy on arithmetic test set. |
| **H4: EOS Termination Retention** | Scaled SFT will not degrade turn-boundary alignment achieved in Exp17. | `STOP_TOKEN` emission rate across full held-out test set. | $< 50.0\%$ EOS termination rate on test split. |
| **H5: Stutter Suppression Stability** | Increased data diversity will maintain low repetition without degenerate looping. | Trigram repetition deficit on Normal Chat Gate v0.1. | $> 15.0\%$ mean repetition deficit. |

---

## 3. Architecture & System Invariants

### 3.1 Model Invariants
- **Architecture**: `ChakrMicro` v0.1 ([chakrview/brain/model.py](file:///d:/Project/ChakrView/chakrview/brain/model.py)).
- **Parameters**: Exactly 3,443,136 unique parameters.
- **Config**: `d_model=192`, `n_layers=6`, `n_heads=6`, `hidden_dim=512`, `max_seq_len=512`, `rms_norm_eps=1e-5`, `tie_embeddings=True`.
- **Pretrained Checkpoint**: Initialized strictly from `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`.
- **Output Destination**: All Exp18 checkpoints and outputs reside strictly within `experiments/research_v09_conversation_sft/exp18_scaled_instruction_format_v1/checkpoints/`.

### 3.2 Tokenizer Vocabulary Invariants
- **Vocabulary Size**: Exactly 4096 tokens ([data/experiments/vocab_4096/vocab.json](file:///d:/Project/ChakrView/data/experiments/vocab_4096/vocab.json)).
- **Special Tokens**: Exactly three tokens:
  - `<BOS>` = 0
  - `<EOS>` = 1
  - `<PAD>` = 2
- **Delimiters**: All prompt wrappers (`Instruction:`, `Response:`, `Context:`, `Question:`, `Answer:`) decompose into existing BPE subwords. No new tokens or special IDs are added.

---

## 4. Compute, Runtime, and Storage Budgets

- **Execution Environment**: Local Windows AMD64, PyTorch CPU execution (`device = "cpu"`).
- **Training Samples**: 1,200 training examples.
- **Batch Size**: 2 sequences (with dynamic batch padding and sequence length cap of 512).
- **Steps per Epoch**: $1200 / 2 = 600$ steps.
- **Target Training Steps**: 1,200 steps (2.0 epochs).
- **Empirical Step Time**: $\approx 0.065$ seconds per forward-backward-optimizer step on 20 logical CPU cores.
- **Total Compute Duration**: $1200 \text{ steps} \times 0.065 \text{ s/step} \approx 78 \text{ seconds}$ ($\approx 1.3$ minutes).
- **Validation Run Time**: Evaluation of 160 validation samples every 100 steps ($\approx 12 \times 4 \text{ s} \approx 48 \text{ seconds}$).
- **Estimated Peak RAM**: $< 1.2 \text{ GB}$ resident memory.
- **Estimated Disk Usage**:
  - Dataset JSONL files: $\approx 500 \text{ KB}$.
  - Model checkpoints (3 intermediate + 1 best + 1 final): $5 \times 13.8 \text{ MB} \approx 69 \text{ MB}$.
  - Total Disk Footprint: $< 100 \text{ MB}$.
