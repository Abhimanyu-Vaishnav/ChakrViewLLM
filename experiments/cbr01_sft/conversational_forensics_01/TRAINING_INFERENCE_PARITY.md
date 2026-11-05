# Training vs Inference Parity Forensic Audit

## 1. Delimiter and Format Comparison

| Feature | Training Pipeline (`sft_engine.py`) | Interactive Chat (`chat.py`) | Parity Status |
|---|---|---|---|
| BOS Token | Prepended (`bos_id=0`) in `build_causal_sequence` | Prepended (`add_bos=True`) | **MATCH** |
| EOS Token | Appended to target (`eos_id=1`) | Stops on `next_tok == 1` | **MATCH** |
| Prompt Normalization | `prompt.strip()` (strip_delimiters=True) | Raw formatted string | **POTENTIAL DIVERGENCE** |
| Prompt Prefix Template | `User: <msg>\nAssistant:` | `User: <msg>\nAssistant:` | **MATCH** |
| History Multi-turn Format | Evaluated across runs | `User: turn1\nAssistant: resp1\nUser: turn2\nAssistant:` | **MATCH** |

## 2. Prompt Samples from `cbr_sft_07_train_228.jsonl`

### Sample 0
- Raw Prompt: `'User: What is modular software design?\nAssistant:'`
- Raw Response: `'Modular design breaks a program into independent, interchangeable modules, improving maintainability, testing, and readability.'`
- Training tokens (w/o BOS): `[88, 835, 324, 90, 107, 725, 589, 707, 2325, 448, 2058, 1192, 113, 66, 13, 68, 118, 1855, 1807, 61]`
- Chat tokens (with BOS): `[0, 88, 835, 324, 90, 107, 725, 589, 707, 2325, 448, 2058, 1192, 113, 66, 13, 68, 118, 1855, 1807, 61]` (BOS=0, rest=[88, 835, 324, 90, 107, 725, 589, 707, 2325, 448, 2058, 1192, 113, 66, 13, 68, 118, 1855, 1807, 61])
- Token equivalence (rest == train): `True`

### Sample 1
- Raw Prompt: `'User: Are you a human?\nAssistant:'`
- Raw Response: `'No, I am an artificial neural language model, not a human.'`
- Training tokens (w/o BOS): `[88, 835, 324, 68, 117, 267, 124, 114, 120, 35, 303, 632, 708, 66, 13, 68, 118, 1855, 1807, 61]`
- Chat tokens (with BOS): `[0, 88, 835, 324, 68, 117, 267, 124, 114, 120, 35, 303, 632, 708, 66, 13, 68, 118, 1855, 1807, 61]` (BOS=0, rest=[88, 835, 324, 68, 117, 267, 124, 114, 120, 35, 303, 632, 708, 66, 13, 68, 118, 1855, 1807, 61])
- Token equivalence (rest == train): `True`

### Sample 2
- Raw Prompt: `'User: Can you access my personal files outside this session?\nAssistant:'`
- Raw Response: `'No, I operate within the active chat session and have no access to external private files.'`
- Training tokens (w/o BOS): `[88, 835, 324, 70, 648, 124, 114, 120, 35, 3158, 112, 301, 498, 118, 279, 327, 828, 1207, 769, 328, 486, 310, 589, 323, 118, 798, 66, 13, 68, 118, 1855, 1807, 61]`
- Chat tokens (with BOS): `[0, 88, 835, 324, 70, 648, 124, 114, 120, 35, 3158, 112, 301, 498, 118, 279, 327, 828, 1207, 769, 328, 486, 310, 589, 323, 118, 798, 66, 13, 68, 118, 1855, 1807, 61]` (BOS=0, rest=[88, 835, 324, 70, 648, 124, 114, 120, 35, 3158, 112, 301, 498, 118, 279, 327, 828, 1207, 769, 328, 486, 310, 589, 323, 118, 798, 66, 13, 68, 118, 1855, 1807, 61])
- Token equivalence (rest == train): `True`

### Sample 3
- Raw Prompt: `'User: I made a silly mistake in my code.\nAssistant:'`
- Raw Response: `'Every developer makes simple mistakes. Finding and fixing them is a natural part of programming.'`
- Training tokens (w/o BOS): `[88, 835, 324, 992, 112, 413, 267, 303, 328, 111, 418, 112, 573, 100, 424, 356, 112, 301, 321, 305, 49, 13, 68, 118, 1855, 1807, 61]`
- Chat tokens (with BOS): `[0, 88, 835, 324, 992, 112, 413, 267, 303, 328, 111, 418, 112, 573, 100, 424, 356, 112, 301, 321, 305, 49, 13, 68, 118, 1855, 1807, 61]` (BOS=0, rest=[88, 835, 324, 992, 112, 413, 267, 303, 328, 111, 418, 112, 573, 100, 424, 356, 112, 301, 321, 305, 49, 13, 68, 118, 1855, 1807, 61])
- Token equivalence (rest == train): `True`

### Sample 4
- Raw Prompt: `'User: What is an embedding in machine learning?\nAssistant:'`
- Raw Response: `'An embedding is a vector representation that maps discrete tokens into a continuous geometric space capturing meaning.'`
- Training tokens (w/o BOS): `[88, 835, 324, 90, 107, 725, 2910, 1263, 356, 1931, 1867, 113, 309, 66, 13, 68, 118, 1855, 1807, 61]`
- Chat tokens (with BOS): `[0, 88, 835, 324, 90, 107, 725, 2910, 1263, 356, 1931, 1867, 113, 309, 66, 13, 68, 118, 1855, 1807, 61]` (BOS=0, rest=[88, 835, 324, 90, 107, 725, 2910, 1263, 356, 1931, 1867, 113, 309, 66, 13, 68, 118, 1855, 1807, 61])
- Token equivalence (rest == train): `True`

## 3. Findings

1. **Bit-Exact Token Matching on Formatted Single-Turn**: When single-turn `User: ...\nAssistant:` is fed into `chat.py`, the token sequence exactly matches the training input tokens.
2. **Multi-Turn Format Saturation**: The training dataset contains only 23 multi-turn examples out of 228 (10.1%). When history is concatenated in `chat.py`, the input sequence structure diverges from the predominant 90% single-turn distribution in the training corpus.
