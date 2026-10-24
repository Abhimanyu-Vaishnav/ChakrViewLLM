# PROPOSAL FOR CONTROLLED FOLLOW-UP EXPERIMENT (EXP18-v2 SFT)

**Proposed Experiment ID**: `EXP18_SFT_CONDITIONED_400`  
**Precondition**: Implementation of verified defects in `SFTDataset` and `TaskEvaluator`.

---

## 1. Background & Hypothesis

The forensic analysis of `EXP18_SFT_V08_BOUNDED_120` confirmed that the model failed to learn instruction following because `SFTDataset` passed the unadorned prompt `rec["prompt"]` to the tokenizer, while evaluation passed the template-formatted prompt `"Instruction: ...\nResponse:"`.

### Primary Hypothesis
If training examples are encoded using the identical prompt conditioning template (`rec.get("formatted_prompt")` or `format_conditioned_prompt(rec["prompt"], mode, context)`) that is presented during inference, the model will learn the association between the instruction delimiter and response generation, eliminating the fallback to pre-training degenerate completions.

---

## 2. Minimal Required Code Updates (Unprotected Files Only)

### In `chakrview/training/sft_engine.py`:
Update `SFTDataset.get_batch`:
```python
for idx in indices:
    rec = self.records[idx]
    append_eos = rec.get("append_eos", True)
    # Use formatted_prompt if present, or construct via format_conditioned_prompt
    prompt_str = rec.get("formatted_prompt")
    if not prompt_str:
        prompt_str = format_conditioned_prompt(
            rec["prompt"],
            prompt_mode=rec.get("prompt_mode", "plain_instruction"),
            context=rec.get("context", "")
        )
    inp, tgt = self.encode_example(prompt_str, rec["response"], append_eos=append_eos)
    batch_inputs.append(inp)
    batch_targets.append(tgt)
```

### In `chakrview/evaluation/task_evaluator.py`:
Update `_score_factual_qa`:
```python
# For single-character entities (e.g. chemical symbols 'W', 'P', 'K'), require whole-word boundary matching:
import re
matched = False
for ent in target_entities:
    if len(ent) <= 2:
        pattern = r"\b" + re.escape(ent) + r"\b"
        if re.search(pattern, gen_clean, re.IGNORECASE):
            matched = True
            break
    else:
        if ent.lower() in gen_clean.lower():
            matched = True
            break
```

---

## 3. Experiment Specification

- **Base Checkpoint**: Isolated copy of canonical `step_10000.pt` (SHA-256: `96357705...`).
- **Steps**: 300 steps.
- **Batch Size**: 2 with gradient accumulation of 2 (effective batch size 4; sampling 1,200 examples across training).
- **Learning Rate**: 5e-5 decaying to 5e-6 via cosine schedule.
- **Evaluation**: Frozen 160-item held-out Exp18 test set with greedy decoding.
- **Success Criteria**:
  1. Measurable reduction in "Step-by-Step Solution" degenerate completions on test set.
  2. Increase in scorable task accuracy above 10.0%.
  3. Increase in EOS completion rate above 25.0%.
