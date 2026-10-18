# Batch 116 — Balanced Tool/No-Tool Training Curriculum Dataset

**Sprint**: ChakrView Master Sprint Batches 115–118  
**Module**: Balanced Tool/No-Tool Training Curriculum  
**Author**: ML Research Engineer  
**Date**: 2026-10-12  

---

## 1. Motivation and Dataset Design

In Batches 95–114, models were trained exclusively on positive tool examples (`TOOL_CALL: ...`). When subsequently evaluated on clarification queries or out-of-domain requests, all models produced 100% false-positive tool calls because they were never exposed to negative examples during SFT.

To calibrate the tool decision gate, we created the **Balanced Tool/No-Tool Training Curriculum**:
- **Positive Examples**: Addition queries requiring calculator execution (`Calculate sum of:\nA={a}\nB={b}`).
- **Negative Controls (Out-of-Domain)**: General knowledge, translation, and open-ended queries requiring standard helpful refusals.
- **Negative Controls (Security)**: Prohibited shell execution and system command injection attempts requiring explicit policy refusals.
- **Negative Controls (Underspecified)**: Missing operands and ambiguous addition requests requiring clarification before calling tools.

---

## 2. Partition Disciplines & Leakage Checks

| Split | Total Records | Tool Addition Queries | Non-Tool Controls | Positive / Negative Ratio |
|---|---|---|---|---|
| **Train** | 80 | 64 | 16 | 4.0 : 1 (80% tool, 20% control) |
| **Validation** | 20 | 16 | 4 | 4.0 : 1 (80% tool, 20% control) |
| **Held-Out Test**| 28 | 20 | 8 | 2.5 : 1 (71.4% tool, 28.6% control)|

### Split Isolation Guarantees:
1. **Operand Pairs**: The 20 held-out test operand pairs have strictly zero overlap with train and validation pairs ($\text{Train} \cap \text{Test} = \emptyset$).
2. **Control Phrasings**: Evaluation non-tool controls feature completely distinct phrasings from training controls (e.g., training uses "What is the capital of Japan?", test uses "What is the capital of Canada?").
3. **Digit Representation**: Both Operand A and Operand B cover all 10 digits $\{0..9\}$ in the held-out test set.

---

## 3. Tokenizer Boundary & Sequence Masking Validation

Every record across all three splits was verified against the canonical BPE tokenizer (`vocab_4096`):
- $\text{encode}(\text{full\_conversation}) \equiv \text{encode}(\text{prompt}) + \text{encode}(\text{target})$ for all 128 records (0 mismatches).
- Prompt tokens receive causal loss mask `-100`.
- The first target token is supervised from the final prompt position `<|im_start|>assistant\n`.
- EOS token (`1`) is explicitly supervised at the completion boundary.
