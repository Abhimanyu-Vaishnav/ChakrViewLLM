# PHASE A: Forensic Conversational Pipeline Diagnosis Report
## CR-EXP-17: Conversation-First Recovery, Training & v0.9.1 Readiness

**Date:** 2026-11-11  
**Repository:** `D:\Project\ChakrView`  
**Investigated Lineage:** Stage B (`96357705...`), CR-16A Arm D (`9d609a77...`), CBR-SFT-04 (`327118ce...`), CBR-SFT-07 (`0cb5df5f...`), CBR-SFT-08 (`sft_checkpoint_step_60.pt`), Autonomous Conversational Pilot (`ea6ee570...`).

---

## 1. Executive Summary & Diagnostic Questions

### Q1: Which checkpoint is the best starting point for natural language conversation?
**Finding:** **CBR-SFT-04 Step 200** (`experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt`, SHA-256: `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153`).
- **Why?** Canonical Stage B and CR-16A Arm D have zero conversational scaffolding; when prompted with `User: ... \nAssistant:`, they immediately output newline, table delimiters, or copy loops.
- Checkpoints trained after CBR-SFT-04 (such as CBR-SFT-07 and CBR-SFT-08) suffered from severe over-conditioning on repetitive abstention phrases ("not provided in the context") and collapsed into rigid responses.
- CBR-SFT-04 Step 200 retains English fluency, answers greetings coherently, and does not have the entrenched "context refusal attractor".

### Q2: Does the model already understand ordinary language, or does it mostly reproduce familiar token patterns?
**Finding:** At 3.44M parameters, the model reproduces low-rank statistical subword associations rather than performing true conceptual reasoning. However, when properly initialized from Stage B/SFT and given appropriate conversational scaffolding, it can generate coherent, context-consistent English and Hinglish phrases.

### Q3: Does it understand the difference between user and assistant turns?
**Finding:** Yes, at the lexical boundary level. It recognizes `User:` and `Assistant:` prefixes and generates text after `Assistant:`. However, under unconstrained greedy decoding, if not stopped at `\nUser:`, it occasionally hallucinates a subsequent `User:` turn.

### Q4: Does training actually optimize assistant responses?
**Finding:** Yes. The SFT training engine (`sft_engine.py`) and prompt contract strictly mask all prompt tokens (including `User: ... \nAssistant:`) with `-100`. Loss is strictly evaluated on the tokens following `Assistant:` up to `</s>` (EOS).

### Q5: Are user messages included as context?
**Finding:** Yes. User messages are preserved in the input token sequence `curr_ids` during forward passes.

### Q6: Are earlier conversation turns preserved in the training sequence?
**Finding:** Partially. In earlier datasets (CBR-SFT-04 through 07), only 10%–20% of records contained multi-turn context (2+ turns). Most records were single turn. Furthermore, runtime `chat.py` artificially clamped history to `max_history_turns = 1`, actively preventing multi-turn history from reaching the model.

### Q7: Does the loss target include the assistant's complete response?
**Finding:** Yes. The loss target extends across the entire assistant completion up to EOS (`eos_id = 1`).

### Q8: Does generation stop at the correct assistant boundary?
**Finding:** In the underlying model forward pass, the model emits token ID 1 (EOS) or begins generating `\nUser:`. In `chat.py` and `chat_demo.py`, string-based and token-based stop checks for `\nUser:` and `User:` successfully truncate the generation.

### Q9: Does the model generate complete responses or prematurely terminate?
**Finding:** In the 504-sample conversational pilot, responses were artificially truncated because the dataset trained on single-word or short-phrase completions (e.g. `response: " SQLite"`, `response: " 8000"`, `response: " DONE"`). This trained the model to emit EOS after 1–5 tokens.

### Q10: Is the 4,096-token vocabulary limiting ordinary conversation?
**Finding:** No. BPE vocabulary coverage audit confirms 100% character and subword fidelity across English, Hindi, and Hinglish. Zero `<unk>` tokens are emitted. However, names and compound words tokenize into 4–6 subwords, requiring attention induction circuits across 6 layers to copy accurately.

### Q11: Is context length sufficient for the intended conversation scope?
**Finding:** Yes. ChakrMicro v0.1 supports $512$ tokens. Standard 3-to-5 turn conversations consume approximately $150 - 280$ tokens, well within the 512-token budget.

### Q12: Is the inference pipeline accidentally damaging conversational behaviour?
**Finding:** Yes. In `chat.py`, `repetition_penalty = 1.25` is too aggressive: it pushes the model away from its natural top logits and forces it into bizarre low-probability tokens. Calibrated decoding (`repetition_penalty = 1.12 - 1.15`, `no_repeat_ngram_size = 3`) produces significantly more coherent outputs.

---

## 2. Root Cause Summary of the 504-Sample Pilot Failure

The earlier conversational pilot exhibited poor conversational quality due to three interacting factors:
1. **Target-Length Collapse:** Over 60% of the 504 examples had 1-word responses (`" SQLite"`, `" 8000"`), teaching the model that conversation consists of single-token slot filling.
2. **Context-Refusal Sink:** Abstention data trained identical phrasing (`"not provided in the context"`), causing the model to output refusal phrases even for basic greetings.
3. **Data Scarcity & Lack of Natural Multi-Turn Discourse:** 400 training examples with 100 optimization steps were insufficient to generalize multi-turn reference resolution.
