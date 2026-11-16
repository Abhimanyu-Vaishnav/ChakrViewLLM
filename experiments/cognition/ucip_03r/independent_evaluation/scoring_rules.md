# Independent Evaluation Rubric & Scoring Rules

## 1. Principles
- **No Length-Based Proxies:** Length alone never grants passing score.
- **Verifiable Semantic Correctness:** Output must satisfy prompt requirements (grammar, script, meaning, or mathematical truth).
- **Four-Tier Verdicts:**
  - `PASS` (Score = 1.0): Completely correct, grammatically coherent, and contextually appropriate.
  - `PARTIAL` (Score = 0.5): Relevant keywords or partial script/arithmetic match, but with minor syntactic defects or incomplete phrasing.
  - `FAIL` (Score = 0.0): Incoherent, wrong script, wrong answer, hallucinations, or random word sequences.
  - `UNSCORABLE` (Score = 0.0): Empty, degenerate single character, or garbage tokens.

## 2. Category Rubrics

### A. English Language Quality
- Must be valid English words forming a grammatically recognizable clause.
- Must address the user's prompt (relevance).
- Random word combinations (e.g. "The planned of the bold") are scored `FAIL`.

### B. Devanagari Hindi
- Must contain genuine Devanagari Unicode characters ($U+0900$ to $U+097F$).
- Must contain recognizable Hindi vocabulary and correct grammatical postpositions/verbs.
- Mixing random English words inside Devanagari fragments or emitting Devanagari garbage words is scored `FAIL`.

### C. Roman Hindi
- Must contain recognizable colloquial Hindi terms written in Roman script (e.g. *kya*, *hai*, *madad*, *namaste*, *kaise*).
- Must form a sensible statement in Roman Hindi.
- English sentences generated in response to Roman Hindi prompts are scored `FAIL`.

### D. Hinglish
- Must exhibit meaningful code-switching between English and Hindi.
- Must be contextually responsive to the prompt.

### E. Exact Copying
- Must match the target string exactly (after whitespace and case normalization).
- Partial string match is scored `PARTIAL` only if $>75\%$ of words match in order; otherwise `FAIL`.

### F. Arithmetic
- Numerical result must equal the mathematically correct value.
- Hallucinated equations (e.g. "2 + 2 is 5") are scored `FAIL`.

### G. Factual QA & Entity Recall
- Must contain the correct entity/fact without contradiction.
