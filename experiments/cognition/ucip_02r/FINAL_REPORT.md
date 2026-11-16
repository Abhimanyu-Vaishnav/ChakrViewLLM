# UCIP-02R: CLI Contract Repair & Neural/Hybrid Mode Verification

**Mission ID:** UCIP-02R  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Date:** 2026-11-16  
**Auditor / Lead Systems Architect:** Autonomous Lead Systems Architect  
**Active Branch:** `exp/cbr-01-conversational-breakthrough`  

---

## 1. Executive Summary & Problem Resolution

In mission UCIP-02, the interactive CLI parameters (`--memory on|off`, `--debug-prompt`) were implemented in `chakrview/chat.py`'s `main()` block, but the package entrypoint `chakrview/__main__.py` defined a separate subparsers hierarchy for `python -m chakrview chat` that lacked these options. Consequently, executing `python -m chakrview chat --help` did not expose `--memory` or `--debug-prompt`.

### Actions Completed:
1. **Unified CLI Contract:** Connected `chakrview/__main__.py` and `chakrview/chat.py` to share the exact same CLI arguments:
   - `--checkpoint CHECKPOINT` (Default: `releases/v1.0_unified_conversational_intelligence/chakrview_v011_model.pt`)
   - `--memory {on,off}` (Explicit default: `on`)
   - `--debug-prompt` (Displays formatted prompt text and exact token count)
   - `--test-mode` (Runs non-interactive test verification)
2. **Runtime Mode Behavioral Verification:**
   - **`--memory on` Mode:** Fact extraction, working memory storage, and hybrid response integration are active. Verified on the Rahul name recall test:
     `[Hybrid Memory: Name] Your name is Rahul.`
     `/memory` reflects active stored facts.
   - **`--memory off` Mode:** Memory extraction and storage are completely disabled. No facts are tracked; neural model operates purely on causal attention:
     `/memory` displays `Memory mode is currently DISABLED (--memory off). No facts are tracked.`
     `What is my name?` yields pure neural generation without hybrid interception.
   - **`--debug-prompt` Mode:** Displays exact prompt text and token count before generation begins.
   - **Session Commands Updated:** `/reset` and `/info` accurately display the active memory mode.
3. **Automated Regression Suite:** Created `experiments/cognition/ucip_02r/tests/test_cli_contract.py` containing 4 unit tests covering `--help`, `--memory on`, `--memory off`, and `--debug-prompt`. All 4 tests passed.
4. **Canonical Lineage:** Cryptographically re-verified all canonical architecture inits, Stage B checkpoints, and protected `chakrview/runtime/session.py`. All passed bit-exact.

---

## 2. Real CLI Execution Output Verification

### A. Help Output (`python -m chakrview chat --help`)
```text
usage: chakrview chat [-h] [--checkpoint CHECKPOINT] [--memory {on,off}]
                      [--debug-prompt] [--test-mode]

options:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        Path to checkpoint .pt file
  --memory {on,off}     Enable or disable hybrid working memory (default: on)
  --debug-prompt        Display full prompt and token count before generation
  --test-mode           Run automated multi-turn verification
```

### B. Memory ON (`python -m chakrview chat --memory on`)
```text
User > My name is Rahul.
ChakrView > Your name is Sharj.

User > What is my name?
ChakrView > [Hybrid Memory: Name] Your name is Rahul.

User > /memory
--- ChakrView Active Memory ---
 - Name: Rahul (Turn 1)
```

### C. Memory OFF (`python -m chakrview chat --memory off`)
```text
User > My name is Rahul.
ChakrView > Your name is Sharj.

User > What is my name?
ChakrView > Your name is Sarhan.

User > /memory
--- ChakrView Active Memory ---
Memory mode is currently DISABLED (--memory off). No facts are tracked.
```

### D. Debug Prompt (`python -m chakrview chat --debug-prompt`)
```text
User > Hello assistant
[DEBUG PROMPT]
Token Count: 16 tokens
Formatted Text:
User: Hello assistant
Assistant:
[END DEBUG PROMPT]
ChakrView > APIt of the problem tool.
```

---

## 3. Git Delivery & Verification

- **Branch:** `exp/cbr-01-conversational-breakthrough`
- **Files Modified:**
  - `chakrview/__main__.py`
  - `chakrview/chat.py`
  - `experiments/cognition/ucip_02r/tests/test_cli_contract.py`
  - `experiments/cognition/ucip_02r/FINAL_REPORT.md`
