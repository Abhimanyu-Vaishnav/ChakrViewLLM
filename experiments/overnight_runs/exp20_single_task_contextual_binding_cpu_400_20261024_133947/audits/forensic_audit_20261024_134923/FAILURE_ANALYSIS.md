# Exp20 Failure Analysis & Mechanistic Audit

**Audit Directory**: `experiments/overnight_runs/exp20_single_task_contextual_binding_cpu_400_20261024_133947/audits/forensic_audit_20261024_134923/`  

---

## 1. Classification of Failures Across the 20 Held-Out Records

Every generated completion in `EVAL_EXP20_PER_RECORD.jsonl` was audited at the character and token level:

| Failure Category | Record Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Blended / Memorized Entity Hallucination** | **19** | **95.0%** | Model produces correct syntax frame `"<Name> is assigned as the <Role>."`, but the entity `<Name>` is hallucinated by recombining memorized names from the training set. |
| **Incorrect Answer Format** | **1** | **5.0%** | In record 0000, model emitted `'Chlight director.'` instead of the full syntax frame. |
| **Context Lost / Truncated** | 0 | 0.0% | Context was fully supplied in all 20 prompts. |
| **Tokenizer Round-Trip Loss** | 0 | 0.0% | Every entity name round-tripped losslessly through `BpeTokenizer`. |
| **Evaluator False Rejection** | 0 | 0.0% | Zero instances of correct answers being marked incorrect. |

---

## 2. Evidence of Memorization Traps vs Dynamic Induction

When given a novel test context:
1. `Zachary Taylor`: Generated `'Hanna Lin is assigned as the principal systems engineer.'`
   - Training entity pool: `"Hanna Marin"` and `"Maya Lin"`.
2. `Fatima Zahra`: Generated `'Handry Lindry is assigned as the lead flight director.'`
   - Training entity pool: `"Caleb Landry"` and `"Henrik Lindqvist"`.
3. `Gennaro Gattuso`: Generated `'Chloe Mancer is assigned as the senior avionics officer.'`
   - Training entity pool: `"Chloe Bennett"` and `"Julian Mercer"`.
4. `Sunita Williams`: Generated `'Lindry Sato is assigned as the senior avionics officer.'`
   - Training entity pool: `"Caleb Landry"` and `"Kenji Sato"`.

### Mechanistic Interpretation
- The model successfully learned the **syntactic distribution** of the training set: it unconditionally outputs a two-word capitalized name followed by `"is assigned as the"` and the role.
- However, its self-attention layers fail to form an **induction circuit** linking the tokens in the `Context:` block to the output slot. Because the entity tokens in test are novel (zero overlap with training), the model's language head falls back on the high-probability token combinations memorized from the 50 training names.
