# SFT Experiment Protocol: Batch 2821–2860
## Controlled SFT Experiment Preparation, Curated Instruction Dataset & Training Pipeline

### 1. Objective
Establish a reproducible, strictly controlled Supervised Fine-Tuning (SFT) experiment pipeline for ChakrView Micro v0.1 (`d_model=192`, `n_layers=6`, `n_heads=6`, `hidden_dim=512`, `vocab_size=4096`, `max_seq_len=512`).

This protocol defines the dataset architecture, token formatting and masking contract, training configuration, evaluation design, and preflight verification gates to improve instruction adherence and context-grounded response accuracy while protecting canonical checkpoints from catastrophic forgetting or corruption.

**Absolute Constraint**: No optimizer steps or weight modifications are executed during preparation. Training requires explicit user authorization.

---

### 2. Dataset Design & Governance
- **Dataset Path**: `experiments/batch2821_2860/SFT_DATASET.jsonl`
- **Total Examples**: 240 locally authored instruction examples.
- **Provenance**: `locally_authored_batch2821_2860` (clean, verified local examples; no synthetic benchmark scraping, no external APIs).
- **Split Distribution**:
  - `TRAIN`: 192 examples (80.0%) -> `experiments/batch2821_2860/train.jsonl`
  - `VALIDATION`: 24 examples (10.0%) -> `experiments/batch2821_2860/val.jsonl`
  - `TEST`: 24 examples (10.0%) -> `experiments/batch2821_2860/SFT_EVALUATION_SET.json`
- **Split Isolation**: Examples are partitioned strictly by task instances to prevent train/test leakage.
- **Task Categories (30 examples each: 24 train / 3 val / 3 test)**:
  1. `simple_instruction`: Direct questions, definitions, factual enumerations.
  2. `context_extraction`: Extracting specific entities/values explicitly present in the prompt.
  3. `context_grounded_qa`: Answering questions conditioned on short domain passages.
  4. `short_summarization`: Concise, single-sentence distillations of supplied technical paragraphs.
  5. `structured_output`: Formatted outputs matching strict JSON schemas.
  6. `uncertainty_insufficient_info`: Refusing to fabricate when information is missing from context.
  7. `conversational_acknowledgement`: Natural assistant greetings and project status acknowledgements.
  8. `bounded_text_transformation`: Deterministic string manipulations (casing, sorting, acronyms).

---

### 3. Causal Loss Masking & Tokenization Contract
- **Tokenizer**: Canonical BPE Tokenizer (`release/chakrview-v0.5.0/data/experiments/vocab_4096`, vocab size 4096).
- **Special Tokens**: `<BOS>` = 0, `<EOS>` = 1, `<PAD>` = 2.
- **Masking Mechanism**:
  - `input_seq = [BOS] + prompt_tokens + response_tokens`
  - `target_seq = [-100]*len(prompt_tokens) + response_tokens + [EOS]`
  - Causal property: Input at index `len(prompt_tokens)` (the terminal prompt token) predicts `target_seq[len(prompt_tokens)] == response_tokens[0]`.
  - Prompt tokens and padding tokens are masked with ignore index `-100`, ensuring gradients are computed strictly on response tokens.
- **Context Limit**: Hard sequence length cap at 512 tokens.

---

### 4. Proposed Training Configuration
- **Parent Checkpoint**: Canonical Step 10,000 (`experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`, SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`).
- **Target Output Directory**: `experiments/batch2821_2860/sft_pilot_run/` (isolated from canonical directories).
- **Pilot Steps**: 100 steps.
- **Batch Size**: 4 (with dynamic padding per batch).
- **Learning Rate**: Peak `5e-5` with cosine decay to minimum `5e-6`.
- **Weight Decay**: `0.01`.
- **Gradient Clipping**: `1.0`.
- **Validation Cadence**: Every 10 steps.
- **Checkpoint Cadence**: Every 25 steps (atomic writes with `.tmp` staging).
- **Seed**: `42` (deterministic random state saving for bit-exact resumption).

---

### 5. Dual-Track Evaluation Strategy
To prevent false improvement or regression, the SFT candidate will be evaluated against:
1. **Batch 2781 Generation Benchmark (28 cases)**: Immutable regression check testing raw continuation, repetition probes, and context boundaries.
2. **Batch 2821 SFT Held-Out Test Set (24 cases)**: Evaluating instruction following, exact entity extraction, structured JSON validity, and honest abstention on missing facts.
