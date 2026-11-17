# UCIP-08: Inference Pipeline Comparison

## 1. Pipeline Architectural Matrix

Three generation pipelines exist in the codebase:

| Parameter / Feature | Pipeline A: Evaluator 800 (`evaluator_800.py`) | Pipeline B: Direct Neural Runner (`run_m3_evaluations.py`) | Pipeline C: Interactive Chat CLI (`chakrview/chat.py`) |
|---|---|---|---|
| **Tokenizer** | `vocab_4096` | `vocab_4096` | `vocab_4096` |
| **BOS / EOS Tokens** | `add_bos=True, add_eos=False` | `add_bos=True, add_eos=False` | `add_bos=True, add_eos=False` |
| **Max Prompt Budget** | Unbounded (Direct tensor cat) | Unbounded | 384 tokens (Turn-budgeted) |
| **Max Generation Tokens** | 32 tokens | 32 tokens | 48 tokens |
| **Sampling Temperature** | `0.0` (Argmax) | `0.0` (Argmax) | `0.0` (Argmax) |
| **Repetition Penalty** | **`1.0` (None)** | **`1.0` (None)** | **`1.15` (Active)** |
| **No-Repeat N-Gram** | **`0` (Disabled)** | **`0` (Disabled)** | **`3` (Active)** |
| **Memory Engine** | Disabled | Disabled | Configurable (`--memory off`) |
| **KV Cache** | Disabled (Full causal recompute) | Disabled | Disabled |

---

## 2. Impact of Sampling Penalty Differences

When tested on the exact same checkpoint (`chakrview_ucip07_medium_model.pt`) and prompt `"What does a CPU do?"`:

- **Pipeline A (Evaluator 800, Rep Penalty = 1.0)**:
  `"A converts program instructions, copatience of a of a gitorgolivitation."`
  The model's native unigram transition distribution produces immediate attractor repetition (`"of a of a"`).
- **Pipeline C (Chat CLI, Rep Penalty = 1.15, No-Repeat 3-gram)**:
  `"A converts program instructions, copatience of a magnet and use lives the dogance to whitation."`
  When `"of a "` attempts to repeat, the 3-gram blocker and repetition penalty zero out the logits of `"of"` and `"a"`. Because the model has poor entropy calibration, the next highest logit corresponds to completely irrelevant lexical tokens (`"magnet"`, `"dogance"`).

**Finding**: Repetition penalties cannot manufacture language competence. On a model with high uncertainty, suppressing repetitions simply scatters the rollout into random vocabulary tokens.
