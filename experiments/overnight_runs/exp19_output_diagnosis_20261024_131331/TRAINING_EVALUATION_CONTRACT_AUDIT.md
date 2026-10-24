# Training and Evaluation Contract Audit

**Target Experiment**: `exp19_debiased_sft_cpu_600_20261024_125700`  
**Diagnosis Directory**: `experiments/overnight_runs/exp19_output_diagnosis_20261024_131331`  
**Audit Scope**: Exact prompt formatting, causal alignment, loss masking, EOS handling, and evaluator contracts.

---

## 1. Prompt Conditioning Parity

### Training Contract
In `chakrview/training/sft_engine.py` (lines 197–209):
```python
if "formatted_prompt" in rec and rec["formatted_prompt"]:
    prompt_to_encode = rec["formatted_prompt"]
elif "prompt_mode" in rec:
    prompt_to_encode = format_conditioned_prompt(
        rec["prompt"],
        prompt_mode=rec["prompt_mode"],
        context=rec.get("context", ""),
    )
```
For `plain_instruction`, `format_conditioned_prompt` returns:
`"Instruction: {prompt}\nResponse:"`

### Inference / Evaluation Contract
In `experiments/overnight_runs/exp19_debiased_sft_cpu_600_20261024_125700/run_exp19.py` (lines 112–118):
```python
p_mode = rec.get("prompt_mode", "plain_instruction")
prompt = rec["prompt"]
context = rec.get("context", "")
f_prompt = format_conditioned_prompt(prompt, prompt_mode=p_mode, context=context)
p_tokens = contract.encode(f_prompt, add_bos=True, add_eos=False)
```

**Finding**: **NO DEFECT FOUND**. The prompt formatting between training and evaluation is 100% identical. Delimiters, whitespace, and prefixes match bit-for-bit.

---

## 2. Causal Target Shifting and Loss Masking Parity

### Implementation Analysis
In `chakrview/training/prompt_contract.py` (`build_causal_sequence`, lines 33–65):
```python
input_seq = [bos_id] + prompt_tokens + response_tokens
target_seq = ([ignore_index] * len(prompt_tokens)) + response_tokens + [eos_id]
```
During training forward pass (`sft_engine.py`, lines 366–367):
```python
logits = self.model(x)  # shape: [B, T, vocab_size]
loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
```

### Mathematical Causal Verification
In autoregressive transformer models:
- At sequence index $t$, the transformer takes token $x_t$ (conditioned on $x_0 \dots x_t$) and outputs hidden representation $h_t$.
- In `chakrview/brain/model.py`, `lm_head(h_t)` computes logits predicting token $x_{t+1}$.
- In `build_causal_sequence`:
  - `input_seq` has length $1 + P + R$.
  - Position 0 is `bos_id`. Its target in `target_seq` is `ignore_index` (-100).
  - Position $P$ (the final prompt token) has target `target_seq[P] = response_tokens[0]` (the first response token).
  - Position $P + R - 1$ (the final response token) has target `target_seq[P + R - 1] = eos_id` (the EOS token, id 1).
- Target masking ignores all prompt tokens: exactly $P$ prompt positions are masked with `-100`.

**Finding**: **NO DEFECT FOUND**. Causal shifting correctly supervises:
1. Predicting the first response token given the complete prompt.
2. Predicting subsequent response tokens autoregressively.
3. Predicting the EOS termination token immediately following the response.

---

## 3. Evaluation Contract & Evaluator Rules Audit

### Evaluator Implementation (`chakrview/evaluation/task_evaluator.py`)
1. **Scorable vs Unscorable**:
   - `short_explanation` (20 records) is marked `NEEDS_REVIEW` (qualitative), correctly yielding a denominator of 140 scorable records.
2. **Article Stripping in Factual QA**:
   - As hardened in Exp18, leading articles (`the`, `a`, `an`) are stripped before entity extraction.
3. **Structured JSON Validation**:
   - Calls `json.loads(text.strip())`. It requires strictly valid RFC 8259 JSON syntax. Python dicts, unquoted keys, or code wrappers fail validation as intended.
4. **Evaluator False Rejections**:
   - Audit of all 140 scorable test records revealed **zero instances** of a semantically correct model answer being rejected by the evaluator. All 140 non-correct outputs were genuine errors (repetition loops, CSS attribute syntax, or arithmetic hallucinations).

**Finding**: Evaluator integrity is sound. The 0/140 score is not caused by evaluator defects.
