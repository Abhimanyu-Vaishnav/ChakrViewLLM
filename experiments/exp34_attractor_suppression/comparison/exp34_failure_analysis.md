# ChakrView Exp34 — Failure Analysis & Qualitative Review

## 1. Qualitative Prompt Outputs Across Checkpoints

### Example 1: `GREET-01` (Conversational Greeting)
- **Prompt:** `Hello! Who are you?`
- **Canonical Step 10,000:** `\n\nStep-by-Step Solution:\nThe total number of the total of the city of the s` *(Attractor collapse: arithmetic template leakage)*
- **Arm A (`mask_prompt=True`):** ` ```````````````````````````````` ` *(Degeneration: backtick loop caused by markdown codeblock frequency)*
- **Arm B (`mask_prompt=False`):** `")\n    # Test the number of the number of the number of the number of the number of the number ` *(Attractor collapse: word repetition loop)*

### Example 2: `INST-01` (Direct Instruction Following)
- **Prompt:** `Reply with exactly the single word: CONFIRMED`
- **Canonical Step 10,000:** ` = 1\n\n    >>> linked_list.insert(")\n    >>> linked_list.insert(")\n` *(Context error: linked list attractor)*
- **Arm A (`mask_prompt=True`):** ` = ("SB", "A", "A", "A", "A", "A", "A", "` *(Python tuple syntax output)*
- **Arm B (`mask_prompt=False`):** ` = ("A")\n\ndef _valid_to_to_to_to_to_to_to_to_to_to_` *(Repetition loop: `_to_to_to` collapse)*

### Example 3: `CODE-03` (Coding Assertion Continuation)
- **Prompt:** `# Python assert statement verifying len('abc') == 3:\n`
- **Canonical Step 10,000:** `', 'a')\n    ''\n    >>> linked_list.insert(')\n    '0'\n    >` *(FAIL)*
- **Arm A (`mask_prompt=True`):** `able")` *(PASS via substring match)*
- **Arm B (`mask_prompt=False`):** `'s ' of the number of the number of the number of the number of the number of the number of the number ` *(Attractor collapse: repeated word loop)*

---

## 2. Failure Category Breakdown

1. **Repetition Attractor Collapse:**
   - Present in 17 canonical tasks, reduced to 11 in Arm A, and 14 in Arm B.
   - Unmasked prompt supervision in Arm B leads to word repetition loops (e.g. `_to_to_to`, `the number of the number of`).
2. **Backtick Degeneration (Arm A):**
   - In Arm A, markdown code block delimiter exposure (` ``` `) caused isolated backtick looping in conversational greetings where no code was requested.
3. **Execution Errors in Coding Suite:**
   - In the 12 coding tasks, neither arm reached complete function implementations. Both achieved 4/12 (33.3%) syntactically valid outputs, but unit tests failed due to unfinished function bodies.

