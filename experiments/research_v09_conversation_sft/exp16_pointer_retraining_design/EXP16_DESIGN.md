# ChakrView Exp16 — Pointer Head Retraining & Generalization Design

**Experiment Target**: `experiments/research_v09_conversation_sft/exp16_pointer_retraining_design/`  
**Date**: 2026-10-22  
**Status**: `RESEARCH & TRAINING DESIGN FROZEN — NO TRAINING EXECUTED`  
**Predecessors**: Exp15 (Pointer Prototype), Exp04 (Robustness Challenge Audit)  

---

## 1. Executive Mission & Conceptual Scope

The forensic audits of Exp14, Exp15, and Exp04 have established:
1. **Generative Failure**: The 3.44M ChakrMicro Causal LM cannot reliably bind dynamic conversational entities without hallucinating memorized pretraining/SFT prefixes ($0\%$ held-out binding).
2. **Template Shortcut Failure**: The Exp15 Field-Conditioned Pointer scored $98.33\%$ on rigid synthetic templates, but collapsed to **$35.00\%$ overall** and **$5.88\%$ on answerable spans** when tested on Exp04 under length jitter and conversational variance.
3. **Training Label Corruption**: In Exp15, sublist token matching corrupted $41\%$ of training labels to $[0, 0]$, biasing the model toward unconditional abstention.

**Exp16 Mission**:  
Design a rigorous, controlled retraining protocol for the `FieldConditionedSpanPointer` head that uses:
- The verified **fail-closed character-to-token span alignment adapter** ([`span_aligner.py`](file:///D:/Project/ChakrView/experiments/research_v09_conversation_sft/exp04_pointer_robustness/span_aligner.py)).
- **Length-jittered, structurally varied conversational training dialogues** with variable target positions.
- An **unseen held-out challenge test split** strictly isolated from training, validation, and threshold selection.

> [!IMPORTANT]
> **Scope Invariant**: Pointer performance measures **query-conditioned contextual span extraction**, not general neural conversational reasoning. Natural language generation, turn orchestration, and dialogue management remain outside the pointer mechanism.

---

## 2. Non-Negotiable Architectural & Cryptographic Safeguards

1. **Backbone Freeze**: The 3.44M ChakrMicro backbone loaded from `training_checkpoint_step_10000.pt` (SHA-256: `96357705...`) remains **100% frozen** (`requires_grad = False` across all 3,443,136 backbone parameters).
2. **Tokenizer Immutability**: Canonical `vocab.json` (`db4d6119...`) and `merges.json` (`ada4b1dc...`) remain untouched.
3. **Protected Runtime**: `chakrview/runtime/session.py` (`9080ff51...`) is strictly preserved and never modified.
4. **No Git Mutations**: Staging, commits, pushes, and branch changes are prohibited.
5. **No Optimization in this Task**: Exp16 is a frozen design specification; no training loops, weight updates, or optimizer steps are executed.

---

## 3. Training & Validation Data Generation Architecture

### A. Fail-Closed Alignment Pipeline
Every training and validation example must be compiled using the Exp04 `span_aligner.py`:
- Target substrings are identified by character offsets in the prompt.
- Tokens overlapping the character's UTF-8 byte span are extracted.
- Exact decoded match `tokenizer.decode(tokens).strip() == target.strip()` is verified.
- If alignment fails, the generator raises `AlignmentError` and aborts compilation. **Zero silent fallbacks to `[0, 0]` are permitted.**
- Position `[0, 0]` is reserved exclusively for verified semantic abstentions.

### B. Anti-Shortcut Structural Diversity
To prevent the pointer head from memorizing narrow token offset windows:
1. **Preamble Length Jitter**: Dialogue preambles must vary between 15 and 90 tokens using natural conversation fillers, shifting the target entity's start token $s^*$ across the range $[15, 120]$.
2. **Multiple Same-Type Candidate Entities**: Prompts must introduce 2–4 candidate entities of the same type (e.g. 3 server nodes, 2 frequency settings) before specifying the queried target.
3. **Turn-Recency & Superseded Updates**: Later turns explicitly update, correct, or supersede earlier turns to force the query head to attend to conversational recency.
4. **Controlled Abstention Ratio**: Semantic abstentions (missing info, unanswerable queries, true ambiguity) must constitute **exactly 25–30%** of the training split, preventing the model from collapsing into default abstention.

---

## 4. Split Strategy & Anti-Leakage Contracts

| Split | Size | Purpose | Source Diversity |
| :--- | :---: | :--- | :--- |
| **Train Split** | 300 examples | Parameter optimization | Distinct entity namespaces, varied dialogue contexts |
| **Validation Split** | 60 examples | Checkpoint selection & early stopping | Zero entity or dialogue overlap with Train |
| **Held-Out Test Split** | 120 examples | Final benchmark evaluation | Completely independent; sealed during training |

**Leakage Controls**:
- Zero shared entity names between Train/Val and Test.
- Distinct dialogue topics, variable sentence orderings, and zero literal duplicate prompts.
- Test split remains sealed until the candidate checkpoint is selected via validation loss.

---

## 5. Model Architecture & Trainable Parameter Accounting

The model architecture remains the verified `FieldConditionedSpanPointer`:
- **Backbone**: Frozen `ChakrMicro` (3,443,136 parameters).
- **Field Query Embedding**: $E_f \in \mathbb{R}^{10 \times 192}$ (1,920 parameters).
- **Start Projections**: $W_q^{\text{start}}, W_k^{\text{start}} \in \mathbb{R}^{192 \times 192}$ (73,728 parameters).
- **End Projections**: $W_q^{\text{end}}, W_k^{\text{end}} \in \mathbb{R}^{192 \times 192}$ (73,728 parameters).
- **Total Trainable Parameters**: **149,376** (4.16% of total network).

$$\text{Logits}_{\text{start}}(t) = \frac{(E_f W_q^{\text{start}}) (H_t W_k^{\text{start}})^T}{\sqrt{192}}$$
$$\text{Logits}_{\text{end}}(t) = \frac{(E_f W_q^{\text{end}}) (H_t W_k^{\text{end}})^T}{\sqrt{192}}$$

---

## 6. Training Budget & Reproducible Optimization Policy

- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, $\text{weight\_decay} = 1\times 10^{-4}$).
- **Learning Rate**: $5\times 10^{-4}$ with linear warmup (20 steps) and cosine decay.
- **Batch Size**: 8.
- **Max Epochs**: 15 (approx. 560 steps).
- **Loss Function**: Symmetrical Cross-Entropy:
  $$\mathcal{L} = \frac{1}{2} \left( \text{CE}(\text{Logits}_{\text{start}}, s^*) + \text{CE}(\text{Logits}_{\text{end}}, e^*) \right)$$
- **Early Stopping & Checkpoint Selection**: Checkpoint selection is strictly governed by **Validation Exact-Span Accuracy**, breaking ties by lowest validation loss.
- **Seed**: Fixed global seed `42` (`torch.manual_seed(42)`).

---

## 7. Comparative Arms & Fair Baselines

To prove that field-conditioned span attention delivers genuine capability without privileged oracle access, Exp16 specifies 4 comparative arms:

1. **Arm A: Frozen Causal LM Backbone**: Native next-token greedy generation (verifies generative prefix hallucination).
2. **Arm B: Length-Agnostic First-Match Pattern Baseline**: Simple regex/token pattern baseline using only prompt text (no oracle metadata access).
3. **Arm C: Query-Agnostic Pointer Ablation**: Pointer model with field conditioning disabled ($E_f = \mathbf{0}$), testing whether slot queries contribute semantic guidance.
4. **Arm D: Field-Conditioned Pointer Prototype**: Candidate trained model.

---

## 8. Pre-Registered Acceptance Criteria

Before any training or test evaluation is launched, the following acceptance thresholds are permanently frozen:

1. **Held-Out Test Exact-Span Accuracy**: $\ge \mathbf{75.00\%}$ ($90 / 120$ exact matches).
2. **Answerable Span Accuracy**: $\ge \mathbf{70.00\%}$ on non-abstention test examples.
3. **Abstention Precision & Recall**: $\ge \mathbf{75.00\%}$ precision and recall on genuine abstention cases.
4. **Length Invariance**: Exact-span accuracy must not degrade by more than 15 percentage points between short ($<100$) and medium ($100–200$) contexts.

If any criterion is not met, the experiment status will be recorded as `ROBUSTNESS THRESHOLD NOT MET`.
