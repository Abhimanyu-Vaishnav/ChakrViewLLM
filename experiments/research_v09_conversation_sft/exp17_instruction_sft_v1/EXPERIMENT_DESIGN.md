# ChakrView Exp17 Instruction SFT v1 — Experiment Design Specification

**Experiment ID**: `exp17_instruction_sft_v1`  
**Parent Directory**: `experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/`  
**Date**: 2026-10-22  
**Target Model**: ChakrMicro (3,443,136 parameters, ModelConfig(d_model=192, n_layers=6, n_heads=6, max_seq_len=512))  
**Base Checkpoint**: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` (SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`)  
**Execution Environment**: Strict CPU Execution (PyTorch CPU, Windows AMD64)  
**Status**: DESIGN & PREFLIGHT PHASE ONLY (Gated — No training permitted without human approval)

---

## 1. Executive Summary & Problem Formulation

In the **Normal Chat Gate v0.1** evaluation, the unassisted canonical ChakrMicro base checkpoint (`step_10000`) scored **0 / 180 points (0.0%)** under greedy unpenalized decoding (`temp=0.0`, `rep_penalty=1.0`).

### Empirical Baseline Bottlenecks:
1. **Pretraining Synthetic Header Bias (Directly Observed)**:
   The base model defaults to memorized pretraining math headers (`Step-by-Step Solution:\nThe total number of...`) across factual questions, conversational prompts, and instruction tasks (6/20 outputs, 30.0%).
2. **Pervasive Degenerate Looping (Directly Observed)**:
   100% of outputs exhibited subword token bigram loops ($\ge 3$ duplicate bigrams); 75.0% exhibited word-trigram duplication.
3. **Turn-Boundary Termination Absence (Directly Observed)**:
   100% of outputs hit the 48-token generation limit (`stop_reason: MAX_TOKENS`) without emitting token ID `1` (`<EOS>`).
4. **Context Blindness in Generative Head (Directly Observed)**:
   When provided with clear conditioning context, the generative head fails to attend to or extract ground truth facts, generating ungrounded formulaic strings.

### Exp17 Objective:
Design a compact, reproducible, CPU-compatible **Instruction Supervised Fine-Tuning (SFT)** experiment to establish whether a targeted conversational dataset and prompt-masked loss can:
- Overcome the pretraining arithmetic bias.
- Align the model with conversational turn boundaries (`<BOS>`, role delimiters, `<EOS>`).
- Establish turn-boundary termination (`<EOS>`) before hitting generation budgets.
- Enable basic instruction following, factual question answering, context extraction, and appropriate abstention.

---

## 2. Experimental Hypotheses & Falsification Conditions

| Hypothesis | Mechanism | Primary Test / Metric | Falsification Condition |
| :--- | :--- | :--- | :--- |
| **H1: Turn-Boundary Termination Alignment** | Exposing the model to explicit `<EOS>` targets at response ends with masked prompts will teach the model that turns are finite. | Percentage of test generations terminating with `STOP_TOKEN` (`<EOS>`) rather than `MAX_TOKENS`. | $< 60\%$ of test prompts emit `<EOS>` under greedy decoding. |
| **H2: Arithmetic Domain De-biasing** | Fine-tuning on diverse conversational tasks will suppress the pretraining synthetic math prior (`Step-by-Step Solution:`). | Occurrence of arithmetic headers on non-arithmetic prompts in Normal Chat Gate. | $\ge 2$ non-arithmetic prompts still emit `Step-by-Step Solution:`. |
| **H3: Single-Turn Instruction Adherence** | Prompt-masked cross-entropy loss trains the generative head to condition strictly on instruction context. | Task completion score on held-out test split and Normal Chat Gate. | Model fails to achieve $\ge 40\%$ task completion on held-out test set. |
| **H4: Context Grounding Activation** | Pairing context-QA prompts with extracted text spans activates attention to context tokens. | Accuracy on extractive context-QA test prompts. | Model hallucinates irrelevant tokens instead of extracting entities from context. |

---

## 3. Architecture & Serialization Policy

### 3.1 Model Invariants
- **Architecture**: `ChakrMicro` (Frozen structural hyperparameters: `vocab_size=4096`, `d_model=192`, `n_layers=6`, `n_heads=6`, `hidden_dim=512`, `max_seq_len=512`, `rms_norm_eps=1e-5`, `tie_embeddings=True`).
- **Total Parameters**: Exactly 3,443,136 unique parameters.
- **Base Checkpoint Binding**: Loaded from `training_checkpoint_step_10000.pt`. Parameter digest verified prior to fine-tuning.

### 3.2 Tokenizer Vocabulary Constraints
- **Special Token Set**: Exactly 3 designated special tokens ([chakrview/tokenizer/special_tokens.py](file:///D:/Project/ChakrView/chakrview/tokenizer/special_tokens.py)):
  - `<BOS>` = `0`
  - `<EOS>` = `1`
  - `<PAD>` = `2`
- **Delimiters as Standard Vocabulary BPE Tokens**:
  - Delimiters such as `Instruction:`, `Response:`, `Context:`, `Question:`, `Answer:`, `<|im_start|>`, `<|im_end|>` are **NOT** added as new special token IDs.
  - Adding special tokens would alter `vocab_size` (4096) and invalidate the canonical embedding matrix `[4096, 192]`.
  - All prompt formatting strings decompose cleanly into existing subword BPE tokens (validated with 100% roundtrip fidelity).

### 3.3 Prompt Conditioning Formats
Exp17 supports two compatible prompt formatting schemes:
1. **Canonical Two-Turn Instruction Format** (Aligned with existing `plain_instruction` and `context_qa` modes in [neural_runtime_adapter.py](file:///D:/Project/ChakrView/chakrview/runtime/neural_runtime_adapter.py)):
   - *Plain Instruction*:
     ```text
     Instruction: {instruction}
     Response: {response}
     ```
   - *Context QA*:
     ```text
     Context: {context}
     Question: {question}
     Answer: {response}
     ```
2. **ChatML-style Delimiter Format** (Aligned with [chakrview/runtime/conversation.py](file:///D:/Project/ChakrView/chakrview/runtime/conversation.py)):
   ```text
   <|im_start|>user
   {user_message}<|im_end|>
   <|im_start|>assistant
   {assistant_response}<|im_end|>
   ```

To ensure immediate, drop-in compatibility with the existing 20-prompt Normal Chat Gate harness, **Exp17 v1 standardizes on Canonical Two-Turn Instruction Format** (`plain_instruction` and `context_qa`).

---

## 4. Hardware & Resource Budgeting (CPU Compatibility)

- **Execution Environment**: Windows AMD64, PyTorch CPU execution (`device = "cpu"`).
- **Benchmarked Empirical Step Time**: **~0.55 seconds per training step** (batch size = 2, seq len = 256, forward + backward + AdamW step).
- **Step & Epoch Sizing**:
  - Total Training Samples: 320 records.
  - Batch Size: 2.
  - Steps per Epoch: 160 steps.
  - Target Epochs: 2.5 (400 total optimizer steps).
- **Total Compute Duration**:
  - 400 steps $\times$ 0.55 s/step $\approx$ **220 seconds (~3.7 minutes)**.
  - Validation passes (every 25 steps, 10 batches $\times$ 0.2 s $\approx$ 2 s): $\approx$ 32 seconds.
  - Total training run time: **$\le$ 5 minutes on CPU**.
- **Memory Footprint**:
  - Model weights: 3.44M $\times$ 4 bytes $\approx$ 13.8 MB.
  - Optimizer states (AdamW): $2 \times 13.8\text{ MB} \approx 27.6\text{ MB}$.
  - Gradients + activations: $\approx 15\text{ MB}$.
  - Total RAM required: **$< 200\text{ MB}$**, safely within typical system headroom.

---

## 5. Experiment Directory Structure

All Exp17 artifacts and specifications are self-contained within:
```text
experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/
├── EXPERIMENT_DESIGN.md              (This master specification)
├── DATASET_SPEC.md                   (Data generation, categories, schemas, and leakage rules)
├── TRAINING_PROTOCOL.md              (Loss masking, optimizer, LR schedule, and checkpointing)
├── EVALUATION_PROTOCOL.md            (Gate evaluation, repetition heuristics, and metrics)
├── PREFLIGHT_CHECKLIST.md            (Pre-run verification gates, trust checks, and stop conditions)
├── ARTIFACT_MANIFEST_TEMPLATE.json   (Provenance manifest template for hashes and metadata)
├── data/                             (To be populated upon approval)
│   ├── train_exp17_v1.jsonl          (320 training examples)
│   ├── val_exp17_v1.jsonl            (40 validation examples)
│   └── test_exp17_v1.jsonl           (40 held-out test examples)
└── checkpoints/                      (To be created upon training run)
    ├── sft_checkpoint_step_100.pt
    ├── sft_checkpoint_step_200.pt
    ├── sft_checkpoint_step_400.pt
    └── sft_checkpoint_best.pt
```

---

## 6. Go / No-Go Decision Gate

Training execution must remain **BLOCKED** until all preflight checklist conditions in [PREFLIGHT_CHECKLIST.md](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp17_instruction_sft_v1/PREFLIGHT_CHECKLIST.md) are satisfied and explicit human authorization is granted.
