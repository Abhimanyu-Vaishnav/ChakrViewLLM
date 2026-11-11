# CR-EXP-17R: Phase B — Dataset Lineage Discrepancy Resolution

## 1. Discrepancy Statement
The CR-17 report reported:
- 1,419 training examples
- 157 validation examples
- 1,576 total examples

However, the dataset filenames on disk and in git were:
- `conversation_train_1080.jsonl`
- `conversation_val_120.jsonl`

This created an apparent contradiction between the filenames ($1,080 + 120 = 1,200$) and the reported figures ($1,419 + 157 = 1,576$).

## 2. Forensic Investigation & Root Cause
An audit of `experiments/conversation/cr_exp_17/assemble_corpus.py` and `experiments/conversation/cr_exp_17/train_conversation_v091.py` revealed:
1. **Initial Dataset Design:** The corpus assembly script was initially scoped to produce an even 1,200 examples (1,080 train, 120 val, following a standard 90/10 split), and the output file paths were statically defined as `conversation_train_1080.jsonl` and `conversation_val_120.jsonl`.
2. **Corpus Expansion:** During pipeline development, targeted synthetic augmentations were integrated to fix conversational refusal, add entity recall, and provide Hindi/Hinglish grounding. The final curated pool expanded to 1,576 records.
3. **Preservation of Filenames:** In `assemble_corpus.py`, the target filenames were not renamed when the size grew. The script wrote 1,419 records to `conversation_train_1080.jsonl` and 157 records to `conversation_val_120.jsonl`.
4. **Manifest and Training Script Verification:**
   - `conversation_corpus_manifest.json` explicitly documents:
     - `total_records`: 1,576
     - `train_records`: 1,419
     - `val_records`: 157
   - `train_conversation_v091.py` directly loaded `conversation_train_1080.jsonl` (1,419 rows) and `conversation_val_120.jsonl` (157 rows).
   - No row-level filtering was performed inside `train_conversation_v091.py`.

## 3. Dataset Lineage & Integrity Verification Table

| Item | Reported in CR-17 | Independently Verified | Status |
| :--- | :---: | :---: | :---: |
| **Training examples in train file** | 1,419 | **1,419** | **VERIFIED (Exact Match)** |
| **Validation examples in val file** | 157 | **157** | **VERIFIED (Exact Match)** |
| **Total corpus examples** | 1,576 | **1,576** | **VERIFIED (Exact Match)** |
| **Hashed train filename** | `conversation_train_1080.jsonl` | `conversation_train_1080.jsonl` | **VERIFIED (Legacy Filename)** |
| **Hashed val filename** | `conversation_val_120.jsonl` | `conversation_val_120.jsonl` | **VERIFIED (Legacy Filename)** |
| **Actual training inputs** | 1,419 | **1,419** | **VERIFIED** |
| **Exact duplicate prompts in train** | 0 | **0** | **VERIFIED (Zero Duplicates)** |
| **Exact duplicate prompt-response pairs** | 0 | **0** | **VERIFIED (Zero Duplicates)** |
| **Train / Val split leakage** | 0 | **0** | **VERIFIED (Strictly Disjoint)** |
| **Benchmark Contamination Rate** | 0.0% claimed | **0.0% (0 / 200)** | **VERIFIED (100% Contamination-Free)** |

## 4. Conclusion
The dataset lineage discrepancy is **fully resolved**. The reported figures of 1,419 train and 157 val examples are 100% accurate reflections of the actual file contents and actual training inputs. The filenames `conversation_train_1080.jsonl` and `conversation_val_120.jsonl` represent legacy file naming that was not updated after corpus augmentation. Training dataset reproducibility is **VERIFIED**.
