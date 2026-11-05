# ChakrView Conversational Failure Forensic Summary Report

## Executive Summary

A comprehensive, read-only forensic investigation was conducted on **ChakrView CBR-SFT-07 Step 50** (`experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_50.pt`, SHA-256: `0cb5df5fa840...`) to diagnose why the model exhibits:
1. Repetitive context refusal phrases (`"is not provided in the context"`).
2. Unrelated greeting/readiness collapses (`"I am ready to assist..."`).
3. 0% multi-turn conversational context recall.
4. Repetition loops and malformed responses under greedy decoding.

All findings are documented with reproducible scripts and data logs located in:
`experiments/cbr01_sft/conversational_forensics_01/`.

---

## Direct Forensic Findings

### 1. Training Pipeline & Loss Alignment: VERIFIED HEALTHY (NO CODE BUG)
- **Prompt Masking**: All prompt tokens $t < \text{len}(\text{prompt})$ are strictly masked with `-100`.
- **Causal Alignment**: `input_seq[len(prompt)]` is the final prompt token, predicting `target[len(prompt)]` (the first response token). The target sequence appends EOS (`eos_id=1`).
- No causal shift or off-by-one indexing error exists in `prompt_contract.py` or `sft_engine.py`.

### 2. Tokenizer & Vocabulary: VERIFIED HEALTHY (100% FIDELITY)
- All 8 probe phrases (including Hindi/Hinglish) encode and decode with 100% exact character fidelity.
- Zero `<unk>` tokens are generated.
- Proper nouns such as `'Abhimanyu'` tokenize into 6 subwords (`['A', 'b', 'h', 'im', 'any', 'u']`), requiring induction attention heads to copy accurately across context.

### 3. Architecture & Parameter Count: CONFIRMED 6 LAYERS, 3.44M PARAMS
- `chakrview/brain/model.py` and the checkpoint state dict confirm:
  - Layers: **6 Transformer blocks** (`layers.0` through `layers.5`).
  - $d_{model}$: **192**.
  - Heads: **6**.
  - Vocab: **4096** with strictly tied input/output embeddings.
  - Unique trainable parameters: **3,443,136**.
  - Prior mentions of "4 layers" in markdown notes were documentation typos copied from earlier toy configurations.

### 4. Primary Root Cause 1: "Context Attractor Sink" in SFT Data
- Across CBR-SFT-04, 05, 06A, and 07, abstention examples heavily reused exact phrasing: `"That information is not provided in the context."` and `"The provided context does not contain..."`.
- This created a severe loss attractor. Whenever a prompt resembles an information query or introduces past conversational context, attention weights route into this memorized refusal sink, producing malformed phrases such as `"Yes, the neural language model is a context"`.

### 5. Primary Root Cause 2: Multi-Turn Representation Scarcity
- In `cbr_sft_07_train_228.jsonl`, only **23 out of 228 records (10.1%)** were multi-turn dialogues. 90% were single-turn prompt-response pairs.
- In a 6-layer model without pretrained weights, 23 examples are insufficient for attention heads to form stable induction circuits across turns. When `chat.py` prepends past turns, the input distribution causes context dissolution (token 1 probability for `'Abhimanyu'` is merely 5%, while `'Y'`/`'N'` dominate).

### 6. Primary Root Cause 3: Repetition Trapping Under Pure Greedy Decoding
- Under pure greedy decoding (`temperature=0.0, repetition_penalty=1.0`), strong self-transition weights induce loops (`"neural neural neural language model"`).
- In `chat.py`, a repetition penalty of 1.25 suppresses repeating tokens, but when forced away from top logits, the model picks low-probability degeneracies (`"You is a contain the problem..."`).

---

## Key Recommendations for CBR-SFT-08

1. **Clean Dataset Balance (300 Pairs)**:
   - 35% Multi-turn entity copying (`User: My name is X.\nAssistant: Hello X!\nUser: What is my name?\nAssistant: Your name is X.`).
   - 20% Single-turn entity extraction without using the word `"context"`.
   - 15% Diverse natural greetings that avoid the `"ready to assist"` monoculture.
   - 15% Varied abstentions replacing `"not provided in the context"`.
   - 15% Technical explanations and Hinglish.
2. **Resume from CBR-SFT-04 Step 200**:
   - Resume from Step 200 before the context-sink bias became entrenched.
3. **Calibrated Decoding**:
   - Use `repetition_penalty=1.15` and `no_repeat_ngram_size=3` at `temperature=0.0`.

