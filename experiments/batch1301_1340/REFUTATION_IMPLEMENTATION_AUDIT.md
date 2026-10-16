# Refutation Implementation Audit — Batches 1301–1340
## Diagnosis of Heuristic Vulnerabilities in `chakrview/runtime/local_knowledge_assistant.py`

### 1. Executive Summary
In Batches 1261–1300, premise contradiction checks were added to `LocalKnowledgeAssistant.ask()` (lines 320–340). While this fixed the 5 specific frozen false-premise benchmark questions (raising false-premise accuracy from 0% to 80%), our Phase 1 audit proves that **the implementation relies on brittle lexical and unaligned numerical heuristics that fail on natural variations**.

### 2. Concrete Defect 1: Unaligned Global Number Sets in "Why" Queries
- **Code Location**: `chakrview/runtime/local_knowledge_assistant.py:L327-340`:
  ```python
  is_why_query = bool(re.search(r"\bwhy\s+(?:does|is|are|did|do|was|were)\b", q_clean_lower))
  if is_why_query and not is_refute:
      q_nums = set(re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", q_clean_lower.replace(",", "")))
      ans_nums = set(re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", ans_lower.replace(",", "")))
      distinct_q_nums = q_nums - common_nums
      distinct_ans_nums = ans_nums - common_nums
      if distinct_q_nums and distinct_ans_nums:
          is_refute = True
  ```
- **Flaw**: It treats *any* number in a "why" query that does not literally appear in the evidence chunk as a contradiction, regardless of the attribute or entity it refers to.
- **Reproducible Failure**:
  - Query: `"Why does ChakrMicro V0.6 have 6 query heads on 1 machine?"`
  - Expected: `ANSWER` (The text establishes 6 query heads; "1 machine" is an uncontradicted physical context).
  - Actual: `REFUTE`! The system claims a contradiction: *"Query asserts 1, but verified evidence establishes - Multi-head attention count: 6 query heads and 2 key-value heads..."*

### 3. Concrete Defect 2: Global Keyword-Triggered Prohibition Contradictions
- **Code Location**: `chakrview/runtime/local_knowledge_assistant.py:L320-325`:
  ```python
  allows_in_q = bool(re.search(r"\b(?:allow|allows|allowed|permit|permits|permitted|enable|enabled)\b", q_clean_lower))
  forbids_in_ans = bool(re.search(r"\b(?:forbidden|prohibited|disabled|denied|disallowed|never)\b", ans_lower))
  if allows_in_q and forbids_in_ans:
      is_refute = True
  ```
- **Flaw**: If the query contains any word meaning "allow" and the retrieved evidence passage contains any word meaning "forbidden" or "prohibited", the system refutes the query without verifying whether they describe the same action or subject.
- **Reproducible Failure**:
  - Query: `"Does ChakrView allow unprivileged executions in production?"`
  - Corpus Fact: `security_sandbox_isolation.md` states: *"All tool executions run in isolated unprivileged worker processes. Administrative access: Root and sudo commands are permanently forbidden."*
  - Expected: `ANSWER` (Unprivileged execution IS allowed/standard; sudo is forbidden).
  - Actual: `REFUTE`! The system claims a contradiction simply because the query has "allow" and the text has "forbidden".

### 4. Categorization of Components
1. **Lexical Heuristics**: Regular expression scans for `\bwhy does\b` and keyword co-occurrence (`allow` vs `forbidden`).
2. **Deterministic Evidence Selection**: BM25 retrieval and concise sentence extraction.
3. **Explicit Claim Comparison**: Rudimentary numerical set subtraction ($Q_{\text{nums}} \setminus E_{\text{nums}}$).
4. **Neural Component**: **ZERO**. The neural encoder does not perform or participate in contradiction detection.
