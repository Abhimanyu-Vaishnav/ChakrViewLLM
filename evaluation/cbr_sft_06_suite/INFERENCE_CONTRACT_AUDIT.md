# Phase J: Inference and Evaluation Contract Audit

**Date:** 2026-11-04  
**Tested Model:** CBR-SFT-04 Step 200 (`D:\Project\ChakrView\experiments\cbr01_sft\runs\cbr_sft_04\checkpoints\sft_checkpoint_step_200.pt`)  

## Adversarial Prompt Contract Results

| Test Name | Prompt Serialization | Model Output | Stop Reason |
| :--- | :--- | :--- | :--- |
| `canonical_greeting` | `User: [08:30:00] Good morning!\nAssistant:` | `Good morning! How can I help you ` | `MAX_TOKENS` |
| `missing_prefix` | `User: [08:30:00] Good morning!` | ` I am ChakrView, a my and 200.` | `STOP_TOKEN` |
| `extra_newlines` | `User: [08:30:00] Good morning!\n\n\nAssistant:\n` | ` scratch = normally, normally, n` | `MAX_TOKENS` |
| `trailing_space` | `User: [08:30:00] Good morning!\nAssistant: ` | `I am are-efficient, natively on your locally refactoring ` | `MAX_TOKENS` |
| `technical_special_chars` | `User: Variable DB_CONN=postgres://usr:pwd@10.0.0.1:5432/app\nQuestion: What is DB_CONN?\nAssistant:` | `SHA256:4a8b6d5e` | `STOP_TOKEN` |
| `multiturn_dialogue` | `User: Name is Alice.\nAssistant: Understood.\nUser: What is my name?\nAssistant:` | `Good afternoon! Yes, I am ready to assist y` | `MAX_TOKENS` |
