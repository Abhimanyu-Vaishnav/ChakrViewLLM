# ChakrView Exp17 Validation — Normal Chat Gate Re-evaluation

**Evaluation Suite**: `experiments/normal_chat_gate_v01/prompts.jsonl` (20 Frozen Normal Chat Prompts)  
**Evaluated Models**:
- **Baseline**: Canonical Step 10,000 Checkpoint (`96357705...`)
- **SFT Model**: Exp17 Step 225 Best Checkpoint (`sft_best_checkpoint.pt`, SHA-256: `fa9b5e8c...`)
**Decoding Harness**: Greedy decoding (`temp=0.0`, `max_new_tokens=48`, `active_eos_stop=True`, CPU)

---

## 1. Quantitative Benchmark Metrics

| Metric | Pre-SFT Baseline (Step 10,000) | Exp17 SFT (Step 225 Best) | Relative Delta |
| :--- | :---: | :---: | :---: |
| **Total Frozen Prompts** | 20 | 20 | 0 |
| **Clean Turn Termination (`EOS%`)** | **5.0%** (1 / 20) | **60.0%** (12 / 20) | **+55.0%** |
| **Max Token Truncation Ceiling** | **95.0%** (19 / 20) | **40.0%** (8 / 20) | **-55.0%** |
| **Empty Outputs** | 0 / 20 (0.0%) | 2 / 20 (10.0%) | +10.0% |
| **Repetition Babble Loops** | 19 / 20 (95.0%) | 4 / 20 (20.0%) | -75.0% |

---

## 2. Capability Differentiation Analysis

To maintain scientific integrity, the following five evaluation dimensions must be strictly separated:

### Dimension 1: Turn Termination & Boundary Learning (`SUBSTANTIAL IMPROVEMENT`)
- **Observed Change**: The baseline model continued generating tokens until truncated by the 48-token ceiling in 19 out of 20 prompts (95%). The SFT model emitted the `<EOS>` stop token in 12 out of 20 prompts (60%).
- **Finding**: The model learned that conversational assistant responses have an endpoint.

### Dimension 2: Response Formatting & Stutter Eradication (`MODERATE IMPROVEMENT`)
- **Observed Change**: The baseline model suffered from an infinite repetition loop:
  `"\n\nStep-by-Step Solution:\nThe total number of the total of the total of the total of the total of the total of the total of the tot"`
- The SFT model completely stopped producing this pre-training babble artifact.
- **Limitation**: In 2 cases (`"Write the word HELLO backwards"`, `"Explain in one short sentence why leaves are green"`), the SFT model produced an immediate empty string (`""`) followed by `EOS`.

### Dimension 3: Instruction Compliance (`PRIMITIVE / MINIMAL`)
- **Observed Change**: When instructed to write `"HELLO"` backwards, the model outputted empty text. When asked for arrival time (Station A to Station B 120 miles away at 60 mph), the model emitted: `"The pronoun is assigned as the primary payload payload payload payload payload payload payload does not cover"`.
- **Finding**: Constraint compliance (word reversals, exact formatting, conditional logic) remains near zero.

### Dimension 4: Factual & Task Correctness (`ZERO / NOT MASTERED`)
- **Observed Change**:
  - Prompt: `"What planet is known as the Red Planet?"`
  - Baseline Output: `"\n\nStep-by-Step Solution:\nThe total number of the total..."`
  - SFT Output: `"The Hermes_train_3."` `[STOP_TOKEN]`
- **Finding**: The model halted with an EOS token, but the factual entity (`"Hermes_train_3"` vs `"Mars"`) is incorrect. Task accuracy across the 20 normal chat prompts is **0.0%**.

### Dimension 5: General Conversational Quality (`UNCONVERGED`)
- The 3.44M parameter model has not achieved coherent multi-turn dialogue. It acts as an early prototype that has acquired response brevity and turn termination, but lacks the world representation required to hold a factual conversation.
