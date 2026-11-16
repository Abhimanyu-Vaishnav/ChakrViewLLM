# Phase 3: Foundational Corpus Audit & Recovery Plan

**Mission:** UCIP-05 — Neural Generation Recovery & Foundational Language Competence  
**Date:** 2026-11-16  

---

## 1. Audit of Existing UCIP-04 Training Corpus (`ucip04_train.jsonl`)

An audit of `experiments/cognition/ucip_04/data/ucip04_train.jsonl` revealed the exact structural dataset flaws responsible for the failures observed by the project owner:

1. **Severe Undersizing and Target Over-Concentration:**
   - Total records in training set: **375**.
   - Unique response targets: **144** (over 60% of targets were reused across different prompts).
   - Top targets were repeated up to 6 times each (e.g., *"Astronomers observe distant galaxies..."*, *"Mathematics provides a foundation..."*).
2. **Narrow Entity Attractor Syntactic Framing:**
   - 32 out of 375 records (8.5%) used fixed entity template frames (*"I reside in [City] -> You live in [City]"*, *"I am building [Project] -> Your project is called [Project]"*).
   - Combined with previous pre-training checkpoints that had similar templates, the model suffered from the persistent attractor *"Your project is a computer..."*.
3. **Complete Absence of Real Multi-Turn Conversational Dialogue:**
   - The "multi-turn" data in UCIP-04 consisted entirely of synthetic single-fact query-response pairs (`User: My name is X.\nAssistant: ...\nUser: What is my name?\nAssistant:`).
   - There was **zero** natural multi-turn conversation (e.g. Turn 1 greeting -> Turn 2 question -> Turn 3 follow-up).
   - When the interactive CLI accumulated real turns, the model encountered an out-of-distribution prompt format and collapsed into fragments (`Youmits interconversp...`, `Kaves ine.`, `Youdr.`).
4. **Total Omission of Arithmetic and Constraint-Following Data:**
   - There were 0 arithmetic word problems or calculation records.
   - When asked *"If I have 5 apples and give 2 to my friend..."*, the model hallucinated (`"A doing is calledy made to stul planets ater."`).
5. **Ultra-Sparse Hindi & Hinglish Coverage:**
   - Devanagari Hindi had only 21 training records.
   - Roman Hindi had only 16 training records.
   - Hinglish had only 16 training records.
   - This acute sparsity caused severe subword collapse under generation penalties.

---

## 2. UCIP-05 Foundational Corpus Requirements

To achieve true generalisation without memorisation or attractor collapse:

| Category | Description | Target Volume |
|---|---|---:|
| **A. Basic English Conversation** | Greetings, persona, capability boundaries, natural dialogue | 600+ |
| **B. Multi-Turn Realistic Dialogues** | 2-turn and 3-turn dialogues covering varied topics | 800+ |
| **C. English Concepts & Explanations** | Science, computing (CPU, RAM, OS), engineering, daily facts | 800+ |
| **D. English Sentence Generation** | Creative and descriptive sentences for diverse objects/actions | 500+ |
| **E. Devanagari Hindi** | Conversational, educational, technical, and cultural dialogues | 700+ |
| **F. Roman Hindi** | Conversational everyday questions, explanations, technical queries | 600+ |
| **G. Natural Hinglish** | Fluent code-switched dialogues, technical discussions, learning | 700+ |
| **H. Exact Copying & Verbatim** | Copying short words, sentences, punctuation, codes | 400+ |
| **I. Elementary Arithmetic** | Word problems, basic addition, subtraction, simple multiplication | 400+ |
| **J. Basic Instruction Following** | One-word answers, list constraints, yes/no formatting, summarisation | 300+ |
| **Total Target** | **Curated, high-quality, non-overlapping dataset** | **5,500 – 6,000+** |

---

## 3. Strict Quality Controls & Leakage Prevention
1. **Benchmark Exclusion:** All 300 probes in `baseline_benchmark.json` (SHA-256 `f74443b8...`) are strictly excluded.
2. **Train/Val/Held-out Separation:**
   - Dedicated training split: ~85% of records.
   - Validation split: ~15% of records (at least 500 records).
   - Frozen benchmark: 300 completely independent probes.
3. **Format Integrity:**
   - Validated UTF-8 encoding.
   - Exact `User: ...\nAssistant: ...` formatting.
   - Non-empty prompt and target.
   - Space prefix on assistant target (`" Hello..."`) matching canonical BPE word boundary tokenisation.
