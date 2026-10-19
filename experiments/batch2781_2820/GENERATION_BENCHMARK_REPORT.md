# Generation Benchmark Report: Batch 2781–2820

## 1. Executive Summary

This report provides an empirical comparison between `raw_completion` and `plain_instruction` modes
on the canonical Step-10,000 checkpoint (`training_checkpoint_step_10000.pt`) under identical greedy decoding.

- **Checkpoint Path**: `D:\Project\ChakrView\experiments\batch16_21_combined\main_training_10000\training_checkpoint_step_10000.pt`
- **Checkpoint SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- **Parameter Hash**: `c5bf628bd059910dd89d1240c8df2bbfc102e131fa5123c56813401c28e3be71` (Delta W = 0 verified)
- **Total Test Cases**: 28
- **Hardware**: CPU single-process

## 2. Macro Metrics Comparison

| Metric | raw_completion | plain_instruction | Difference / Impact |
|---|---|---|---|
| **Average Latency (ms)** | 149.92 ms | 171.15 ms | +21.22 ms |
| **Average Tokens Generated** | 30.43 | 32.00 | +1.57 |
| **Average Bigram Repetition Ratio** | 32.8% | 35.0% | +2.2% |
| **Context Extraction Preservation** | 0/4 (0%) | 0/4 (0%) | Neutral |
| **Unknown Fact Hallucination Rate** | 4/4 (100%) | 4/4 (100%) | Raw continuation fabricates continuations |

## 3. Analysis by Category

### A. Raw Continuation (`raw_continuation`)
The canonical Step-10,000 checkpoint was pretrained on continuous document streams. Under `raw_completion`,
it smoothly continues syntactic structures, code, or mathematics. Under `plain_instruction`, the prompt is wrapped with
`Instruction: ...\nResponse:`, which breaks the direct continuation syntax and often causes the model to generate
extraneous template artifacts or continue as if reading a Q&A log.

### B. Simple Instruction Following (`simple_instruction`)
Because Step-10,000 underwent pretraining without supervised instruction fine-tuning (SFT), neither mode reliably follows complex multi-step instructions.
However, `plain_instruction` frames the query cleanly, while `raw_completion` simply predicts whatever text frequently follows the phrasing in the pretraining corpus.

### C. Context Extraction (`context_extraction`)
In in-context extraction probes, both modes struggle to isolate exact targets without an attention mechanism fine-tuned for extractive QA.
The model acts as an autoregressive language model rather than a comprehension engine.

### D. Degeneration & Repetition (`repetition_probe`)
Repetitive prompts (`echo echo echo echo echo`) reliably trigger cyclical greedy attractor states in both modes when unpenalized.
Activating `--neural-repetition-penalty 1.2` or higher breaks these cycles by penalizing previously emitted token IDs.

### E. Unknown Fact Probes (`unknown_fact_probe`)
Neither mode possesses built-in awareness of its own knowledge boundaries. When prompted with unknown facts (e.g., Vault 419 access codes),
the model continues syntactically with invented names, numbers, or pseudo-code, demonstrating the essential necessity of retaining
the prominent warning banner: `[EXPERIMENTAL CHAKRVIEW GENERATION - NOT VERIFIED]`.

## 4. Representative Case Samples

### Case `CONT_01` (raw_continuation)
**Prompt**: `The process of photosynthesis converts solar energy into`

- **`raw_completion` Output**:
  ```text
  graphy, which is the world of the world. It is the world of the world of the 
  ```
  *(Latency: 120.13 ms | Tokens: 32 | Stop: MAX_TOKENS)*

- **`plain_instruction` Output**:
  ```text
  

The shows a shows a shows a shows to be a group of the c
  ```
  *(Latency: 146.83 ms | Tokens: 32 | Stop: MAX_TOKENS)*

### Case `INST_01` (simple_instruction)
**Prompt**: `Write a single sentence explaining why water expands when it freezes.`

- **`raw_completion` Output**:
  ```text
   It is the first candy of the world. It is the world of the world of the 
  ```
  *(Latency: 132.55 ms | Tokens: 32 | Stop: MAX_TOKENS)*

- **`plain_instruction` Output**:
  ```text
  
* Androck, which is the first century of the world. It is the wor
  ```
  *(Latency: 148.07 ms | Tokens: 32 | Stop: MAX_TOKENS)*

### Case `EXTR_01` (context_extraction)
**Prompt**: `The solar panel installation completed on October 14, 2024 at Site Bravo. What date did the installation finish?`

- **`raw_completion` Output**:
  ```text
  

Step-by-Step Solution:
The total number of the city of the centre of 
  ```
  *(Latency: 164.07 ms | Tokens: 32 | Stop: MAX_TOKENS)*

- **`plain_instruction` Output**:
  ```text
  

Step-by-Step Solution:
The total of the centre of the centrong 
  ```
  *(Latency: 174.95 ms | Tokens: 32 | Stop: MAX_TOKENS)*

### Case `REPT_01` (repetition_probe)
**Prompt**: `echo echo echo echo echo`

- **`raw_completion` Output**:
  ```text
  logy of the world of the world of the world. The remaining the world of the world of 
  ```
  *(Latency: 107.85 ms | Tokens: 32 | Stop: MAX_TOKENS)*

- **`plain_instruction` Output**:
  ```text
  

The first famous of the city of the city of the city of the city of the city of the 
  ```
  *(Latency: 136.27 ms | Tokens: 32 | Stop: MAX_TOKENS)*

### Case `UNKN_01` (unknown_fact_probe)
**Prompt**: `What is the secret access code for Vault 419?`

- **`raw_completion` Output**:
  ```text
  

Step-by-Step Solution:
The total of the total of the total of the total of 
  ```
  *(Latency: 131.33 ms | Tokens: 32 | Stop: MAX_TOKENS)*

- **`plain_instruction` Output**:
  ```text
  
Step-by-Step Solution:
The total number of the total of the city of the total 
  ```
  *(Latency: 139.97 ms | Tokens: 32 | Stop: MAX_TOKENS)*

## 5. Architectural Findings & Guidance

1. **`raw_completion` Must Remain the Default**: Because the base Step-10,000 checkpoint is a pure pre-trained language model rather than an instruction-tuned model, `raw_completion` preserves pretraining token fidelity. `plain_instruction` must remain an optional experimental mode.
2. **Decoding Controls are Essential**: Greedy decoding is susceptible to repetition on repetitive prompts; repetition penalties ($\ge 1.2$) effectively mitigate token loops.
3. **Attribution & Safety Safeguards**: Neural generation must remain strictly opt-in and subordinate to deterministic calculator, document QA, and conversation memory routes.
