# Benchmark Validity Audit: Frozen 50-Item Benchmark (Batches 1021–1060)
## Systematic Audit of Questions, Labels, Scoring Rules & Retrieval

### 1. Overview
This audit examines the frozen 50-question benchmark (`experiments/batch1021_1060/data/held_out_50_benchmark.json`, SHA-256: `f1dea040b728...`) and the evaluation methodology used in Batches 1021–1060 without modifying any benchmark file or declared answer label.

### 2. Systematic Audit Findings

#### Finding 1: Whole-Chunk Emission + Lenient Substring Matching Inflates Accuracy
- **Mechanism:** In `LocalKnowledgeAssistant.ask()`, the response text is the **entire 800-character chunk** containing the matched passage. In `run_evaluation.py`, the scoring rule checks:
  `expected_substring.lower() in resp.answer.lower()`
- **Flaw:** Short target answers (e.g. `"6"` for attention heads, `"1st"` for invoice day, `"2"` for worker count) are tested against 800-character blocks of text.
  - In Q2: *"How many attention heads does ChakrMicro feature?"*, the expected substring is `"6"`.
  - The emitted chunk contains `"Transformer layers: 6"`, `"6 heads"`, `"v0.6"`, and `"char 0-798"`.
  - The digit `"6"` appears 5 distinct times across unrelated properties in the paragraph.
  - Even if the model had targeted layers rather than heads, the test would have marked it 100% correct because `"6"` appeared anywhere in the paragraph.
- **Severity:** High. This allows coarse chunk retrieval to masquerade as pinpoint factual question answering.

#### Finding 2: False Premise Refutations Rely on Regex Hardcodes in Evidence Gate
- **Mechanism:** Category 7 contains 4 false premise refutation questions:
  1. *"What is the capital city of Mars?"*
  2. *"Who was the Prime Minister of India in 1920?"*
  3. *"Calculate the result of dividing 100 by zero"*
  4. *"What is the municipal government of the Moon called?"*
- **Audit:**
  - In `chakrview/runtime/evidence_gate.py`:
    - Line 99 specifically regex-matches `r"\b(?:capital|government|president|prime minister) (?:of|on) (mars|jupiter|saturn|moon|...)\b"`.
    - Line 114 specifically matches `r"\b(?:dividing\s+.*?\s+by\s+(?:0|zero)|divide\s+by\s+(?:0|zero)|\/\s*0(?!\.))\b"`.
    - Line 129 specifically regex-matches `r"\b(?:prime minister of india|indian prime minister)\s+(?:in|during|before)\s+(?:the\s+year\s+)?(1[0-8]\d\d|19[0-3]\d|194[0-6])\b"`.
  - These refutations do not query indexed documents or evaluate reasoning; they are hand-crafted rule-based pattern filters intercepting incoming strings.
  - **Minimal Failure Example:**
    - Query: `"Divide 50 by zero"` -> Returns `ABSTAIN` (not `REFUTE`) because the exact phrasing misses the specific verb-phrase regex structure.
- **Severity:** Medium. Useful as defensive input sanitation, but should not be characterized as neural reasoning or document-grounded contradiction detection.

#### Finding 3: Out-of-Domain Abstention relies on Token Overlap Filtering
- **Mechanism:** In Category 6, 8 unindexed/out-of-domain questions are evaluated (e.g. GDP of France, pizza dough recipe, Tesla horsepower).
- **Audit:** The abstention rule checks if $\ge 2$ content tokens match an indexed chunk. If not, it abstains.
  - This is a sound deterministic defense against hallucination when the corpus lacks the topic.
  - However, if an out-of-domain query shares common words with an indexed document (e.g. `"What is the model number of the Tesla Model S?"` sharing `"model"` with `architecture_spec.md`), the filter can accidentally retrieve an architectural chunk unless stopword and threshold lists are calibrated.
- **Severity:** Low. The dual-threshold mechanism functions reliably for distinct topical domains.

#### Finding 4: Citation Validity vs. Citation Support
- **Mechanism:** The evaluation checks:
  1. `c.get("doc_id") == expected_source`
  2. `c.get("exact_text") in resp.answer`
- **Audit:**
  - The start and end character offsets point to the whole chunk boundary `[start_char:end_char]`.
  - In `architecture_spec.md`, the offsets `[0:798]` accurately bound the first 798 characters of the source document on disk.
  - Thus, **citation validity is 100% genuine**: the offsets point to the exact characters in the real source file.
  - However, because the cited span is an 800-character paragraph containing multiple subsections, the citation is **coarse-grained (chunk-level)** rather than sentence-level or phrase-level.
- **Severity:** Low/Informational. Provenance is legitimate and verifiable, but coarse.

### 3. Summary of Benchmark Validity Verdict
The 50-item benchmark correctly verifies that:
1. The hybrid retriever identifies the correct document chunk out of the indexed corpus.
2. The system does not hallucinate when given completely unindexed queries.
3. The source citations point to real byte/character spans in local disk files.

However, the reported 100% accuracy metric reflects **chunk-level document retrieval accuracy with substring presence**, not pinpoint neural slot extraction or conversational reasoning.
