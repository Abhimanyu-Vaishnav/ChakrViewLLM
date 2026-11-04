# CBR-SFT-01 — FORENSIC FAILURE ANALYSIS & ROOT CAUSE

---

## 1. Why Did Exp34 v3 Training Fail to Break the 0/60 Conversational Benchmark?

Through forensic output analysis comparing training examples and test probes, the precise root cause has been isolated:

### 1. Severe Serialization Format Mismatch
- **M3 Evaluation Probes Format:**
  Every task in `m3_conversational_tasks.json` uses the strict chat turn delimiter:
  ```
  User: Hi, good morning!
  Assistant:
  ```
- **Exp34 v3 Training Data Format:**
  All 900 training records in `exp34_train_split_900.jsonl` are raw single-turn instructional questions with **zero** `User:` or `Assistant:` scaffolding:
  ```
  How does BFS find the shortest path in an unweighted graph?
  Calculate the simple interest on a principal of $3100...
  ```
- **The Empirical Consequence:**
  When prompted with `'User: ...\nAssistant:'`, the model encounters tokens it was never trained on during SFT. Because `User:` and `Assistant:` never appeared in Exp34 v3, the model's attention heads associate the colon and newline with Wikipedia-style table markup from Stage C pretraining (e.g. emitting `\n! scope=&quot;background-color: #ffffff&quot; | \n`), completely bypassing its SFT instruction weights.

### 2. General Knowledge vs. Microscopic Representation Mismatch
- Exp34 v3 targets complex technical topics (BFS algorithms, simple interest calculations, Python code debugging).
- For a 3.44M parameter model with small capacity, attempting to memorize 900 complex technical paragraphs in 100 steps creates high cross-entropy confusion rather than foundational conversational behavior.
- Conversational grounding requires simple dialogue turn-taking (greetings, confirmations, role identity, and short entity copies) formatted in the exact chat schema used by evaluation.
