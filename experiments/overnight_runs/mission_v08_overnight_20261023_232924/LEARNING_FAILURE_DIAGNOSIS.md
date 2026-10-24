# LEARNING FAILURE FORENSIC DIAGNOSIS — EXP18 SFT RUN

**Experiment Investigated**: `EXP18_SFT_V08_BOUNDED_120`  
**Run Artifacts**: `experiments/overnight_runs/mission_v08_overnight_20261023_232924/checkpoints/sft_exp18_run/`  
**Evaluator**: TaskEvaluator v0.1 (`chakrview/evaluation/task_evaluator.py`)  
**Evaluation Protocol**: Frozen 160-item held-out Exp18 test set, greedy decoding, max 45 new tokens.

---

## 1. Executive Summary

In the overnight mission, the 120-step SFT run succeeded numerically (loss decreased monotonically from 4.6159 to 3.3562, and EOS completion rose from 0.0% to 8.75%). However, measured task accuracy failed to improve (1.43% for SFT vs 2.14% for baseline across 140 scorable test records).

Forensic investigation of the actual training implementation, prompt tokenization, and decoding outputs uncovered the exact root cause of this failure:
1. **Severe Prompt Formatting Mismatch (Root Cause #1)**:
   - In `SFTDataset.encode_example`, training examples were tokenized using `rec["prompt"]` directly (e.g. `"What is the capital city of Japan?"`).
   - In evaluation (`NeuralRuntimeAdapter.generate` and `stage_g_comparative_eval.py`), prompts were formatted using `format_conditioned_prompt` (e.g. `"Instruction: What is the capital city of Japan?\nResponse:"`).
   - Empirical test: Prompting the SFT checkpoint with the formatted prefix triggered the base pre-training prior (`"\nStep-by-Step Solution:\nThe total number of..."`), whereas prompting it with the unformatted raw string correctly suppressed the step-by-step degenerate prior and outputted conversational text.
2. **Short Step Horizon on Low-Capacity Architecture (Root Cause #2)**:
   - 120 steps with batch size 2 (effective batch size 2, 240 examples sampled out of 1,200) was sufficient to learn token-level language patterns and EOS placement (+8.75% EOS), but insufficient to transfer factual and arithmetic reasoning into a 3.4M parameter model.
3. **Metric Noise Floor and Substring Artifacts (Root Cause #3)**:
   - Reconciling the 3 baseline "correct" records revealed that single-character element symbols (e.g. Tungsten symbol `"W"`) matched substring occurrences in degenerate outputs (`"show"`). The true baseline task accuracy was 0.0%, and the difference between 3/140 and 2/140 is statistical noise.

---

## 2. Ranked Hypotheses & Forensic Evidence

| Hypothesis | Rank | Forensic Evidence | Verified / Refuted |
| :--- | :---: | :--- | :---: |
| **H1: Conditioning Format Distribution Shift** | **#1 (Primary Root Cause)** | `SFTDataset` passed unformatted `rec["prompt"]` to tokenizer; inference passed `"Instruction: ...\nResponse:"`. Direct model testing proved the formatted prompt triggers pre-training degeneration while raw prompt does not. | **CONFIRMED DEFECT** |
| **H2: Step Horizon Undersampling** | **#2 (Contributing Factor)** | 120 steps sampled only 240 instances across 8 categories (30 examples per category). Not enough gradient steps to encode multi-domain task competence into 3.4M parameters. | **CONFIRMED LIMITATION** |
| **H3: Evaluator Noise & False-Positive Artifacts** | **#3 (Measurement Factor)** | Single-letter matching in `_score_factual_qa` (e.g. `"W"`) scored repeated nonsense as correct in baseline. True accuracy of both checkpoints is near 0.0%. | **CONFIRMED FLAW** |
| **H4: Training Pipeline Broken / Non-finite Loss** | Refuted | Causal sequence construction, prompt masking (`-100`), gradient clipping, AdamW weight updates, and parameter tying were rigorously validated. Loss is finite and decreases stably. | **REFUTED** |
| **H5: Corrupted Checkpoint or Lineage** | Refuted | SHA-256 for canonical checkpoint bitwise matches `96357705...`. Checkpoint copy and weight loading `strict=True` pass completely. | **REFUTED** |

---

## 3. Verified Code Defects & Required Repairs

### Defect 1: `SFTDataset` Bypasses Prompt Conditioning Formatting
In `chakrview/training/sft_engine.py`:
```python
# CURRENT FLAWED IMPLEMENTATION:
for idx in indices:
    rec = self.records[idx]
    append_eos = rec.get("append_eos", True)
    inp, tgt = self.encode_example(rec["prompt"], rec["response"], append_eos=append_eos)
```
When records contain `formatted_prompt` or require `format_conditioned_prompt(prompt, mode, context)`, `SFTDataset` trains on the raw unadorned prompt, creating an irreconcilable train/test distribution gap.

### Defect 2: 1-Letter Substring Matching in `_score_factual_qa`
In `chakrview/evaluation/task_evaluator.py`:
```python
# CURRENT FLAWED LOGIC:
matched = any(ent.lower() in gen_clean.lower() for ent in target_entities)
```
When target entity is a single character (like `"W"` for Tungsten or `"P"` for Phosphorus), any generated word containing "w" or "p" (e.g. `"show"`, `"problem"`) is incorrectly scored as CORRECT.
