# Phase 1: Full Autoregressive Generation Pipeline Audit

**Mission:** UCIP-05 — Neural Generation Recovery & Foundational Language Competence  
**Date:** 2026-11-16  
**Auditor:** Autonomous Lead Engineer  

---

## 1. Tokenizer Correctness & Round-Trip Fidelity

- **Artifacts:** `data/experiments/vocab_4096/vocab.json` (SHA-256 `db4d6119...`) and `merges.json` (SHA-256 `ada4b1dc...`).
- **Round-Trip Test Suite:** 95 representative strings spanning:
  - English conversational greetings, polite interactions, and idioms.
  - Technical vocabulary (CPU, RAM, compiler, operating system, photosynthesis, neural networks, backpropagation).
  - Devanagari Hindi (complex conjuncts, matras, nuktas, halant, full stop `।`).
  - Roman Hindi and natural code-switched Hinglish.
  - Exact copying strings and arithmetic expressions.
- **Result:** **95 / 95 exact byte-level matches (100.0% round-trip fidelity)**.
- **Finding:** The tokenizer is mathematically correct. Character mappings, Devanagari multi-byte UTF-8 sequences, whitespace, and numbers encode and decode cleanly. The generation defects are **not** caused by tokenizer encoding/decoding corruption.

---

## 2. Training-Target Alignment & Loss Masking Audit

- **Script Inspected:** `experiments/cognition/ucip_04/checkpoints/train_ucip04.py`
- **Collation Logic:**
  - `prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)`
  - `target_ids = tokenizer.encode(target, add_bos=False, add_eos=True)`
  - `input_ids = prompt_ids + target_ids`
  - `labels = [-100] * len(prompt_ids) + target_ids`
  - `logits = model(inp[:, :-1])`
  - `shift_logits = logits.view(-1, vocab_size)`
  - `shift_targets = tgt[:, 1:].contiguous().view(-1)`
- **Alignment Verification:**
  - Prompt tokens are masked with `-100` and do **not** contribute to loss.
  - The final token of the prompt (`:` at `pos = len(prompt_ids)-1`) predicts the first token of the target.
  - The penultimate token of the target predicts the EOS token (`1`).
  - The shift is strictly next-token prediction ($x_t \to x_{t+1}$).
- **Finding:** The training objective and loss masking mathematics are sound. However, **the target data itself contained severe repetition of narrow syntactic frames and zero multi-turn conversational examples**.

---

## 3. Attention, Masking & Positional Correctness

- **Causal Mask:** Lower-triangular causal mask verified in `ChakrMicro.forward()`.
- **Equivalence Test:** Parallel batch forward pass logits at `[0, -1, :]` were compared against step-wise sequential autoregressive forward pass logits.
  - Maximum absolute difference: **`0.0`** (bit-for-bit identical).
- **Causal Invariance Test:** Modifying a token at position $t=8$ produced an absolute difference of **`0.0`** on all output logits for positions $t < 8$.
- **Finding:** Attention causality and position handling are strictly verified and operationally intact.

---

## 4. Forward Pass and Autoregressive Decoding Loop

- **CLI Generation Loop:** In `chakrview/chat.py`, autoregressive decoding sequentially appends `next_tok` to `curr` and passes the full accumulated context to the model at each step.
- **Repetition Penalty Behavior:**
  - When repetition penalty is disabled (`1.0`), models frequently enter repeating single-token loops (`an an an an`, `is a ful is a ful`).
  - When repetition penalty is enabled (`1.15`), the penalty pushes probability off the repeating token into neighboring subwords, occasionally creating distorted word blends (e.g. `sourtheral`, `assimporital`, `Youdr`) if the base model's probability distribution is poorly calibrated.
- **Finding:** The decoding mechanics are operational. The underlying root cause is that the model's learned distribution has narrow high-loss basins for multi-turn conversational prefixes.

---

## 5. Prompt-Format Audit & Multi-Turn Catastrophic Collapse

- **Format in Training Data:**
  All training samples were single-turn:
  ```
  User: <query>\nAssistant: <response><eos>
  ```
- **Format in Interactive CLI (`ConversationalChatCLI`):**
  Turn 1:
  ```
  User: Hello ChakrView. How are you?\nAssistant:
  ```
  Turn 2:
  ```
  User: Hello ChakrView. How are you?
  Assistant: Hello! I am an independents been sourtheral learning support ke liye available hain.
  User: What does a CPU do?
  Assistant:
  ```
  Turn 3+: Continues accumulating previous conversation turns up to the prompt token budget.
- **Catastrophic Failure Diagnosis:**
  Because the model was trained exclusively on single-turn sequences starting with `BOS User:`, the presence of intermediate `User:` and `Assistant:` prefixes and previous response text in the prompt context severely derails attention. The model was never trained to attend to the latest `User:` query when historical turns precede it! As a result:
  - By Turn 3, the model output collapsed into fragments: `Youmits interconversp and ready to assimporital networking from me! Yes, I am allyour project is a computer.`
  - By Turn 4, it degenerated to: `Kaves ine.`
  - By Turn 7, it collapsed into: `Youdr.`
- **Required Solution:**
  1. The training corpus **must include diverse multi-turn conversational dialogues** with varied contexts.
  2. The model must learn to attend to the immediate user turn while maintaining coherent dialogue flow.

---

## 6. Cross-Checkpoint Architectural Lineage Comparison

A comparative evaluation was executed across the 5 relevant historical checkpoints using identical prompts:

| Checkpoint | Parameters | Architecture | Hello ChakrView | What does a CPU do? | Say exactly: The sky is blue. | Arithmetic (5 apples - 2) |
|---|---|---|---|---|---|---|
| **v0.11** | 3.44M | ChakrMicro (192d, 6L) | Project intro | `"I reciples was manager tha."` | `"Good evening! What project..."` | `"Your name is Aarah."` (attractor) |
| **UCIP-03 Micro** | 3.44M | ChakrMicro (192d, 6L) | `"You live in Neha."` | `"The mein is a neural neural hai."` | `"The planned of the bold."` | `"The capital of the horing..."` |
| **UCIP-04 Micro** | 3.44M | ChakrMicro (192d, 6L) | `"Solar panels generate in the..."` | `"A mountain ChakrView planning..."` | `"Musics the mountain Earth's..."` | `"You live in electrical area."` |
| **UCIP-04 Medium** | 12.93M | ChakrMedium (384d, 7L) | Repetition / Hinglish blend | **Accurate** (memorized single-turn) | Solar system attractor | `"A doing is calledy..."` |
| **UCIP-01R 13M Parent** | 12.93M | ChakrMedium (384d, 7L) | Attractor | `"I do not have access to that information."` | `"Paris."` | `"Your project was a dist."` |

### Key Diagnostic Insights:
1. **ChakrMedium (12.93M) possesses vastly superior linguistic capacity** compared to ChakrMicro (3.44M), being the only checkpoint that generated an accurate, coherent technical explanation for the CPU prompt in single-turn mode.
2. **However, ChakrMedium was overfitted on an ultra-narrow dataset** (375 records, 144 unique targets, 0 multi-turn records).
3. **ChakrMicro (3.44M) under-indexes on capacity** and suffers from severe interference when trained on multi-lingual tasks without sufficient parameter width.
4. **Conclusion:** Foundational recovery requires expanding the dataset to 4,000+ balanced, verified records covering single-turn and multi-turn conversations, arithmetic, exact copying, Devanagari Hindi, Roman Hindi, and Hinglish, and training both ChakrMicro and ChakrMedium under controlled autoregressive supervision.
