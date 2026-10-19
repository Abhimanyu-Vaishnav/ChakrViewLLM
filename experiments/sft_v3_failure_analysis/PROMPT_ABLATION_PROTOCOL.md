# PROMPT-CONTRACT & DELIMITER ABLATION PROTOCOL
**Scope:** Controlled Inference-Only Investigation of Delimiter Stems & Attractor Triggers  
**Date:** 2026-10-19  
**Execution Context:** `experiments/sft_v3_failure_analysis/`  
**Safety Gate:** Inference Only (ZERO Training, No New Tokens Introduced)

---

## 1. Objective and Hypothesis

The SFT V3 pilot revealed a major failure mode: in unconstrained greedy decoding, the candidate model entered the pretraining Python code attractor (`{"__main__":\n import doctest`) on **90.0% of structured validation prompts**.

### Core Hypothesis:
The canonical prompt ending:
```text
Instruction: ...
Response:
```
coupled with the expected target opening:
```json
{"status": ...
```
directly activates the strong pretraining association between unindented curly braces and Python module tests / dictionary definitions. 

By varying the text delimiter preceding the generation point—without introducing any out-of-vocabulary or untrained special tokens—we can determine whether the code attractor is triggered specifically by the `\nResponse:` delimiter or if it is an invariant response to all structured extraction tasks.

---

## 2. Tested Prompt Formats (Existing Vocab Only)

We evaluated 5 strictly controlled prompt formats across the exact same 10 representative validation records on both the Canonical Base Checkpoint and the SFT V3 Candidate Checkpoint:

| Format ID | Format Style | Prompt Terminal Suffix | Token IDs of Suffix | Attractor Rate (Base) | Attractor Rate (Candidate) | Valid JSON Rate (Candidate) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Format 1** | Canonical V3 | `\nResponse:` | `[279, 323, 61]` (`'on'`, `'se'`, `':'`) | **0 / 10 (0%)** | **10 / 10 (100%)** | **0 / 10 (0%)** |
| **Format 2** | Explicit JSON Prefix | `\nJSON Output: {` | `[1145, 324, 126]` (`'put'`, `': '`, `'{'`) | **1 / 10 (10%)** | **2 / 10 (20%)** | **0 / 10 (0%)** |
| **Format 3** | Section Delimited | `\n\n### JSON RESPONSE\n{` | `[72, 13, 126]` (`'E'`, `'\n'`, `'{'`) | **0 / 10 (0%)** | **6 / 10 (60%)** | **0 / 10 (0%)** |
| **Format 4** | Chat-Turn Delimited | `User: ...\nAssistant: {` | `[279, 323, 126]` (`'ant'`, `': '`, `'{'`) | **0 / 10 (0%)** | **0 / 10 (0%)** | **0 / 10 (0%)** |
| **Format 5** | Plain Question-Answer | `Question: ...\nAnswer: ` | `[122, 283, 61]` (`'w'`, `'er'`, `':'`) | **0 / 10 (0%)** | **0 / 10 (0%)** | **0 / 10 (0%)** |

---

## 3. Empirical Findings from the Ablation

1. **`\nResponse:` is the Primary Code-Attractor Trigger**:
   - In Format 1 (`\nResponse:`), the candidate model enters the Python doctest attractor on **10/10 (100%)** prompts.
   - When the terminal delimiter is shifted to `User: ...\nAssistant: {` (Format 4), the code attractor rate immediately plummets from **100% to 0%**.
   - When the task is presented as a plain Question-Answer prompt (`\nAnswer: `, Format 5), the code attractor rate is also **0%**.
2. **Pre-supplying the Opening Brace `{` Suppresses the Attractor**:
   - In Format 2 (`\nJSON Output: {`), the code attractor rate drops from 100% to **20%**. 
   - However, even when the attractor is suppressed, the model emits repetitive token lists (e.g. `"strict", "code", "index", "code"`) rather than extracting facts from the context.
3. **Format Changes Eliminate Attractors, But Do Not Impart Extraction**:
   - While Chat-Turn formatting (Format 4) completely neutralizes the doctest attractor (0% attractor rate), **Exact Task Match remains 0/10**. 
   - The model has not learned to attend from the instruction position back into the context tokens to copy entity strings.

---

## 4. Retraining Proposal: Special Token Delimiter (`<|json_start|>`)

If textual delimiter redesign is insufficient for robust production extraction, the formal architecture proposal is to introduce an explicit special token:

### Specifications:
1. **Token Name**: `<|json_start|>` (and corresponding `<|json_end|>`).
2. **Token ID**: Allocate within the unused special token range (`[3, 31]`). Currently `NUM_SPECIAL_TOKENS = 32`, with IDs `3..31` reserved.
3. **Embedding Initialization**: Initialize the embedding vector of `<|json_start|>` as the mean of the tokens `JSON` and `{` in the canonical embedding matrix, preserving variance.
4. **Retraining Requirement**: The model must be trained for at least 300–500 steps so the attention query projections learn to map `<|json_start|>` to contextual key extraction heads.
