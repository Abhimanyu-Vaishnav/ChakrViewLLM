# Exp20 Evaluator Audit & Score Interpretation

**Audit Directory**: `experiments/overnight_runs/exp20_single_task_contextual_binding_cpu_400_20261024_133947/audits/forensic_audit_20261024_134923/`  

---

## 1. End-to-End Data Path Verification

The complete data pipeline was inspected:
1. **Raw Context and Task Prompt**:
   - `Context: Official expedition roster dispatch: Victoria Sterling is assigned as the chief orbital navigator for Orbital Expedition TEST-10.`
   - `Question: In Orbital Expedition TEST-10, who is assigned as the chief orbital navigator?`
2. **Formatted Prompt Conditioning**:
   - `format_conditioned_prompt(prompt, prompt_mode="context_qa", context=ctx)` correctly constructs:
     `"Context: {ctx}\nQuestion: {prompt}\nAnswer:"`
3. **Tokenizer Encoding**:
   - `contract.encode(formatted_prompt, add_bos=True, add_eos=False)` produces token IDs starting with BOS (0) followed by the complete prompt text. Context length is 56 tokens, well below the 512 context window. Zero truncation occurred.
4. **Model Forward & Autoregressive Decoding**:
   - Greedy decoding ($T=0.0$, `max_new_tokens=45`, `seed=42`) emits tokens until token ID 1 (EOS) is sampled.
5. **Evaluator Scoring**:
   - `TaskEvaluator.evaluate_record(rec, gen_str, stop_reason=stop_reason)` evaluates `outcome` using substring entity matching and strict category criteria.

---

## 2. Evaluation Reproduction & Verification of the 0/20 Score

Inference was independently re-executed across all 20 test records using `checkpoints/sft_best_checkpoint.pt`:
- **Bit-Exact Reproduction**: **20 / 20 generations matched `EVAL_EXP20_PER_RECORD.jsonl` identically**.
- **Scoring Status**: All 20 records were evaluated with `scoring_status == "SCORED"`. There were **0 unscorable or qualitative records**.
- **Evaluator Outcome**: All 20 records received `outcome == "INCORRECT"`.
- **False Rejection Audit**: Every single one of the 20 generated answers was manually inspected. In zero cases did the model output the correct entity name. 

**Conclusion**: The 0/20 score is an undisputed, reproducible factual finding, not an artifact of evaluator strictness or formatting bugs.
