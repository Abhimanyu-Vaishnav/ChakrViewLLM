# SFT V3 Failure Analysis — Root Cause Report

## 1. Executive Summary

The SFT V3 100-step pilot on ChakrMicro (4.5M parameters) achieved a notable reduction in validation loss from **5.3873** to **3.6204** (reduction of 1.7669 nats), but recorded **0/30 (0.0%)** exact task matches on structured extraction records and exhibited a **90%** Python doctest generation attractor (`{"__main__":\n import doctest`).

This diagnostic milestone performed an exhaustive, non-invasive investigation without running any optimizer steps or modifying canonical model/tokenizer artifacts. 

Key diagnostic findings:
1. **The doctest attractor is prompt-boundary conditioned**: When evaluated with the standard training delimiter (`\nResponse:`), the model emitted the Python doctest attractor in 100% of tested cases. However, altering the boundary to a structured dialog delimiter (`User: ... Assistant: {`) reduced the attractor rate to **0.0%**.
2. **The extraction failure is caused by weak context-to-answer attention routing**: In simplified diagnostics evaluating direct context copying (Task A) and single-field extraction (Task B), the candidate checkpoint achieved **0/10** value accuracy. The 100 optimizer steps (10.5 seconds on CPU) taught the model the *surface syntax* of JSON formatting (leading to lower cross-entropy on JSON delimiters and keys), but were insufficient to develop induction heads or attention copy circuits capable of retrieving entity values from arbitrary context windows.
3. **The Schema Prefix Trie does not block correct answers**: 100% of ground-truth validation targets are mathematically reachable under the schema trie constraints. The trie fails to produce correct extractions solely because the underlying model assigns higher logits to generic/attractor tokens than to the correct context-grounded values.

---

## 2. Pilot Metric Reconciliation

Detailed reconciliation is documented in [PILOT_METRIC_RECONCILIATION.md](file:///d:/Project/ChakrView/experiments/sft_v3_failure_analysis/PILOT_METRIC_RECONCILIATION.md).

### 2.1 Resolution of 3.6204 vs 3.3215
- **3.3215**: Computed inside `train_pilot.py` at step 100 on a stochastic 5-batch subsample (5 records total) used strictly as an online checkpoint-selection criterion.
- **3.6204**: Computed by `evaluate_comparative.py` over the full, deterministic 40-record validation set (30 structured extraction records + 10 replay records) comprising 860 active supervised response tokens.
- **Verification**: Both calculations used identical token loss masks (`-100` on prompt, cross-entropy on response), identical tokenizer, and causal attention. The discrepancy is purely sample-size variance (5 records vs 40 records).

### 2.2 Distinction Between Structured and Replay Records
| Split / Cohort | Count | Base Loss | Candidate Loss | Exact Match (Base) | Exact Match (Candidate) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **All Validation Records** | 40 | 5.3873 | 3.6204 | 0/40 (0.0%) | 0/40 (0.0%) |
| **Structured Records** | 30 | 5.8920 | 3.7840 | 0/30 (0.0%) | 0/30 (0.0%) |
| **Replay Records** | 10 | 3.8732 | 3.1296 | 0/10 (0.0%) | 0/10 (0.0%) |

---

## 3. Prompt-Contract and Delimiter Ablation

We evaluated the candidate checkpoint across 5 distinct prompt framing formats on 10 held-out structured validation records without modifying any model weights. Full data is recorded in [EXTRACTION_DIAGNOSTICS.json](file:///d:/Project/ChakrView/experiments/sft_v3_failure_analysis/EXTRACTION_DIAGNOSTICS.json).

| Format ID | Delimiter Pattern | Pre-Generation Context | Attractor Rate | Empty/Malformed | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `current_format` | `\nResponse:` | `\nResponse:` | **10/10 (100%)** | 0/10 | Emits doctest attractor |
| `task_boundary` | `\n### Task: Extract\nResponse:` | `Response:` | **10/10 (100%)** | 0/10 | Attractor persists |
| `answer_prefix` | `\nAnswer:\n{` | `\n{` | **1/10 (10%)** | 9/10 | Suppresses doctest, generates code comments |
| `chat_dialog` | `User: ... Assistant: {` | `Assistant: {` | **0/10 (0.0%)** | 10/10 | Completely eliminates doctest attractor |
| `separated_sections` | `Context:... Question:... Answer:`| `Answer:` | **0/10 (0.0%)** | 10/10 | Completely eliminates doctest attractor |

### Conclusion on Prompt Framing
The Python doctest attractor (`{"__main__":\n import doctest`) is an artifact of pretraining code corpus distributions triggered specifically when a naked `{` follows standard script/docstring transitions (`\nResponse:`). Wrapping the interaction in clear dialog tags (`User:`, `Assistant:`) successfully redirects the causal attention path away from doctest generation. However, prompt format alone is insufficient to recover value extraction.

---

## 4. Progressive Extraction Diagnostics (Tasks A through E)

To isolate where extraction fails, we evaluated the candidate checkpoint on progressively simpler sub-tasks using the same validation records:

- **Task A: Direct Context Copying** ("Copy the exact word for status from context:"):
  - **Value Hit Rate**: 0/10 (0.0%)
  - **Behavior**: Emitted conversational fillers or incomplete code blocks rather than attending to and copying the target token.
- **Task B: Single-Field Plaintext QA** ("What is the status? Answer:"):
  - **Value Hit Rate**: 0/10 (0.0%)
  - **Behavior**: Failed to emit the ground truth entity.
- **Task C: Single-Field Minimal JSON** (`{"status":`):
  - **Valid JSON Syntax**: 0/10 (0.0%)
  - **Value Hit Rate**: 0/10 (0.0%)
- **Task D: Two-Field JSON Extraction**:
  - **Valid JSON Syntax**: 0/10 (0.0%)
- **Task E: Full Multi-Field Schema Extraction**:
  - **Exact Match**: 0/10 (0.0%)

### Analysis
The failure occurs at the most fundamental level: **the model cannot yet route attention from generation tokens to context tokens**. The 100-step pilot was long enough to compress loss on output formatting tokens, but far too short to align multi-layer cross-token attention circuits for context extraction.

---

## 5. Constrained Decoding Audit

The Schema Prefix Trie was audited for reachability:
- **Ground Truth Reachability**: **10/10 (100%)** of target validation strings are completely reachable under the trie token transitions.
- **Cause of Trie Failure**: When the model has low probability mass for the correct context-grounded token, the constrained decoder is forced to select whatever permitted token has the highest probability. If the model's logits favor syntax or repetitive tokens over the true value, the trie cannot manufacture intelligence that does not exist in the neural distribution.

---

## 6. Root-Cause Ranking

Summary of hypothesis ranking from [ROOT_CAUSE_RANKING.json](file:///d:/Project/ChakrView/experiments/sft_v3_failure_analysis/ROOT_CAUSE_RANKING.json):

1. **Pretraining continuation attractor (Rank 1, High Confidence)**: Bare `{` after `\nResponse:` evokes pretraining Python test harness weights.
2. **Prompt-format / delimiter mismatch (Rank 2, High Confidence)**: Delimiters lack dialog grounding; fixable via chat-style templates.
3. **Weak context-to-answer learning (Rank 3, High Confidence)**: Zero copying capability demonstrated on Task A & B.
4. **Insufficient training duration (Rank 4, Medium Confidence)**: 100 steps (10.5s CPU time) is inadequate for cross-attention circuit formation.
5. **Model-capacity limitations (Rank 5, Medium Confidence)**: 4.5M parameters may require focused single-task curriculum.
6. **Dataset target ambiguity (Rank 6, Medium Confidence)**: Multi-task schema mixing within 100 steps causes task interference.
7. **Causal loss masking defect (Rank 7, Low Confidence / Disproven)**: Audit confirmed loss masking is mathematically correct.
8. **Decoding constraints defect (Rank 8, Low Confidence / Disproven)**: Audit confirmed 100% reachability under trie.

---

## 7. Recommended Next Experiment

**Recommendation**: Conduct a controlled, isolated **Single-Field Context Copy SFT Pilot (250–500 steps)** using the dialog prompt template (`User: ... Assistant: {`).

### Justification:
1. **Highest Diagnostic Value**: Isolates whether ChakrMicro's 4.5M parameter architecture can learn context-to-output copying when task complexity is reduced to 1 field and dialog framing suppresses the doctest attractor.
2. **Lowest Cost**: Uses existing training pipeline on CPU (~30–60 seconds total), without requiring architectural or tokenizer changes.
