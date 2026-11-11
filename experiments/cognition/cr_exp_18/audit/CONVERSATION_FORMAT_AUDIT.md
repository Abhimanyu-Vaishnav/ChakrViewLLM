# CR-EXP-18: Phase D — Conversation Formatting Audit Report

## 1. End-to-End Pipeline Path
Data flow:
```text
Original JSONL records -> ConversationFormatter -> Tokenizer -> Input IDs & Target Mask -> Model Forward -> chat.py Runtime
```

## 2. Formatting Consistency Audit
Inspection across 20 representative training and inference examples:
- **Speaker Tags:** Canonical `User: ` and `Assistant: ` boundaries are formatted consistently.
- **Prompt Termination:** Every prompt in `conversation_train_1080.jsonl` and `CONVERSATION_BENCHMARK_V1.json` terminates with `Assistant:`.
- **Response Format:** Every response begins immediately with the first word of the assistant's reply (no leading whitespace).
- **Subword Boundary Behavior:** 
  - Tokenizing `Assistant:` yields `[68, 118, 1855, 1807, 61]`, ending with colon token ID `61`.
  - When the assistant starts generating, token ID `324` (`': '`) is not emitted because the prompt already ends with colon `61`. The model predicts the first word token directly.
- **History Representation:**
  - Turn separator `\n` is consistently applied between turns.
  - Multi-turn prompts in training use the exact string structure:
    `User: {u1}\nAssistant: {a1}\nUser: {u2}\nAssistant:`
  - Inference scripts and `chat.py` assemble multi-turn prompts using the identical template: `\n`.join([f"User: {turn['user']}\nAssistant: {turn['assistant']}", f"User: {user_msg}\nAssistant:"]).

## 3. Findings
Formatting template discrepancy is **NOT SUPPORTED** as the cause of failure. The string templates at training and inference are identical.
