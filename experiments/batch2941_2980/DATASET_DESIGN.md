# Dataset Design: Batch 2941–2980
## Balanced 70/30 SFT Mixture Architecture: Instruction Adherence & Pretraining Replay

### 1. Architectural Motivation & Hypothesis

In Batch 2861–2900, pure instruction fine-tuning (`100%` instruction-wrapped pairs) induced catastrophic suppression of unformatted natural text continuations. When evaluated on the Batch 2781 benchmark, the model emitted `<EOS>` immediately on 71.4% of raw completion prompts, because it learned that terminal punctuation outside an `Instruction:` wrapper signals a sequence boundary.

**Hypothesis**:
A balanced **70% instruction / 30% pretraining-replay** data mixture provides sufficient instruction supervision to learn task formatting while preserving the base model's unconditioned continuation priors and preventing premature `<EOS>` collapse.

---

### 2. Dataset Composition (400 Total Examples)

| Component | Category | Examples | Target Behavior | Loss Supervision Policy |
|---|---|---|---|---|
| **Instruction (70%)** | `simple_instruction` | 40 | Direct concept definitions and factual answers | Response tokens + `<EOS>` supervised |
| **Instruction (70%)** | `context_extraction` | 40 | Exact entity/port/value extraction from prompt | Response tokens + `<EOS>` supervised |
| **Instruction (70%)** | `context_grounded_qa` | 40 | Domain questions answered strictly from context | Response tokens + `<EOS>` supervised |
| **Instruction (70%)** | `short_summarization` | 40 | Concise, single-sentence distillations | Response tokens + `<EOS>` supervised |
| **Instruction (70%)** | `structured_output` | 40 | JSON formatted schemas (arrays, objects, dicts) | Response tokens + `<EOS>` supervised |
| **Instruction (70%)** | `uncertainty_insufficient_info` | 40 | Honest refusal when information is absent | Response tokens + `<EOS>` supervised |
| **Instruction (70%)** | `bounded_text_transformation` | 40 | Deterministic string manipulation | Response tokens + `<EOS>` supervised |
| **Replay (30%)** | `base_continuation_replay` | 120 | Unconditioned prose, math, and Python code continuation | Response tokens supervised; **`<EOS>` NOT appended** |

---

### 3. Pretraining Replay Policy & Causal Loss Masking

- **Source Corpus**: Sourced exclusively from verified training directories (`data/raw/english`, `data/raw/code`, `data/raw/mathematics`). Held-out test passages are strictly excluded.
- **Formatting Policy**: No `Instruction:` or `Response:` wrappers are used for replay data.
- **EOS Policy**:
  - Replay examples represent open document continuations.
  - Therefore, `append_eos = False`.
  - Input sequence: `[BOS] + prompt_tokens + response_tokens[:-1]`
  - Target sequence: `[-100]*len(prompt_tokens) + response_tokens`
  - The model is trained to predict the next natural token without forcing an artificial `<EOS>` termination at the end of the window.

---

### 4. Split Isolation and Contamination Controls

- **Split Ratio**:
  - `Train`: 320 examples (80.0%)
  - `Validation`: 40 examples (10.0%)
  - `Test`: 40 examples (10.0%)
- **Isolation Check**:
  - Partitioning occurs per-task family before dataset serialization.
  - Zero prompt overlap between `train`, `val`, and `test`.
  - Zero overlap between replay examples and any held-out benchmark questions.
