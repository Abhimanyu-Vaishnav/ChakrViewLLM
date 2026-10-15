# Independent Evaluation Protocol: Batches 1061–1100
## Rigorous Black-Box Evaluation Standards

### 1. Principles of Independent Evaluation
1. **Public Interface Conformance:** The evaluator interacts with `LocalKnowledgeAssistant` exclusively through its documented methods:
   - `assistant.ingest_directory(corpus_path)`
   - `assistant.ask(query_str)`
   - `assistant.reset_conversation()`
2. **Byte-Level Citation Verification:**
   - Every citation emitted by the assistant is resolved against the actual file on disk.
   - The file is read from the filesystem, and the slice `doc_content[start_char:end_char]` is verified to exactly match the cited text.
   - Any citation pointing to a non-existent document, negative offset, offset exceeding file length, or text mismatch fails the validity check immediately.
3. **Four Distinct Evaluation Tiers:**
   - **Citation Validity:** The cited document exists on disk and the character bounds enclose the cited passage.
   - **Citation Relevance:** The cited document matches the expected authoritative source for the fact.
   - **Citation Support:** The cited text span explicitly contains the claim being answered.
   - **Answer Correctness:** The response contains the ground-truth factual value (or appropriate abstention / refutation).
4. **Independent Execution:** The evaluation harness does not import or depend on internal scoring variables from `run_evaluation.py`.
