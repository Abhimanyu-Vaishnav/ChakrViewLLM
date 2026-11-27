# NR-045: Recommended Remediation Plan (NR-046 Blueprint)

## 1. Executive Summary

Based on the forensic evidence established in **NR-045**, this document specifies the precise, controlled remediation blueprint for **CV-GENESIS-NR-046: SFT Dataset De-duplication, Prompt Normalization & Response-Only Loss Training**.

---

## 2. Recommended NR-046 Work Packages

### Work Package 1: Corpus Normalization & De-Duplication (`data/corpus_v2/`)
1. **Prompt Stripping:** Strip all inner `"User: ...\nAssistant: "` tags from prompt strings across Stage-D and UCIP-14R sources to guarantee pure `<user>\n{text}\n<assistant>\n` alignment.
2. **Elimination of Copy Boilerplate:** Remove all 1,126 single-word token echo samples (Category `COPY`) and synthetic boilerplate templates (`"Summary #..."`).
3. **Response Length Expansion:** Enforce a minimum response length threshold ($\ge 8$ words) for conversational items, retaining informative explanations.
4. **Target Dataset Size:** ~4,500 to 5,000 pristine, high-density multilingual conversational dialogue pairs.

### Work Package 2: Response-Only Loss Masking (`train_conversational_sft_v2.py`)
1. **Prompt Masking Implementation:**
   ```python
   p_ids = tokenizer.encode(f"<user>\n{prompt}\n<assistant>\n")
   r_ids = tokenizer.encode(f"{response}<eos>")
   input_ids = p_ids + r_ids
   labels = [-100] * len(p_ids) + r_ids  # Zero loss on prompt tokens
   ```
2. **Loss Computation:** Verify that 100% of gradient updates are driven solely by response token cross-entropy.
3. **Initialization:** Initialize fine-tuning directly from Canonical Stage-D Best (`experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt`, SHA-256 `2c21632e...`).

### Work Package 3: Inference Runtime Upgrades (`brain_runtime.py`)
1. **Default Decoding Parameters:** Set default sampling parameters to `temperature = 0.7`, `top_p = 0.9`, and `repetition_penalty = 1.15`.
2. **Dynamic RoPE Length Handling:** Retain dynamic frequency recalculation when context length exceeds 256 tokens.
3. **Template Identity:** Ensure `brain_runtime.py` and `interactive_chat.py` use the exact token sequence matched to SFT training.

---

## 3. Explicit What NOT to Change
- **Do NOT scale architecture:** Keep ChakrGenesis-17M (17.06M parameters).
- **Do NOT change tokenizer:** Retain Canonical Byte-Level BPE 4096 (`db4d6119...`).
- **Do NOT use external pretrained weights or hosted LLMs.**
- **Do NOT overwrite historical checkpoints.** Save the remediated checkpoint to a new versioned path: `experiments/genesis_stage_m_046/checkpoints/chakr_1_0_rc2_best.pt`.
