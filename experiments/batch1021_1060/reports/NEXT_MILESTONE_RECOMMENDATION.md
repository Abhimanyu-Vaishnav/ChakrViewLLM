# Next Milestone Recommendation: Batches 1061–1100
## Strategic Roadmap Decision

### 1. Direct Answers to Strategic Review Questions

**Q1: Can a user run the application and complete the documented workflow?**
- **Yes.** A user can install the existing environment, run `python -m chakrview.runtime.assistant_cli ingest <dir>`, ask single queries via `ask "<question>"`, inspect citations, or launch multi-turn sessions via `interactive`. All operations complete locally without external API dependencies.

**Q2: How many held-out questions were answered correctly?**
- **50 out of 50 (100.0%)** on the frozen held-out benchmark. Every factual question returned the expected factual answer, every out-of-domain question produced an explicit honest abstention, and every false premise was refuted.

**Q3: How many citations were valid and actually supported their claims?**
- **38 out of 38 citations (100.0%)** resolved to real, existing local document files with exact character bounding offsets `[start_char:end_char]`. Zero citations were fabricated.

**Q4: Did the assistant abstain correctly when evidence was missing?**
- **Yes.** 8 out of 8 unanswerable/out-of-domain queries produced an explicit abstention (`"I could not find sufficient verified evidence in the indexed local documents to answer this question."`).

**Q5: Did persistent context improve follow-up handling?**
- **Yes.** 7 out of 7 conversational follow-ups referring to previous entities (*"that model"*, *"that tier"*, *"that policy"*) correctly resolved referents and returned the targeted information.

**Q6: Which components contributed neural capability, and which provided deterministic retrieval, validation, or serialization?**
- **Deterministic:** Lexical BM25 indexing (`BM25KnowledgeIndex`), document parsing and chunking (`DocumentIngestionEngine`), and session history budgeting (`ConversationContextManager`).
- **Hybrid:** `HybridRetriever` (combining BM25 term weighting and dense vector indexing) and `EvidenceGovernedAnsweringGate` (routing between refutation policies, deterministic tools, and retrieval).
- **Neural:** The underlying `ChakrMicro` causal transformer core and semantic embedding representations. (Note: The high accuracy of factual citation Q&A was driven by the hybrid retrieval and evidence gate pipeline, not by uncontrolled autoregressive generation.)

**Q7: What remains unreliable?**
- Ingestion of non-textual formats (binary PDF, DOCX) and multi-column tables. Scaling to $>100,000$ document chunks will require moving from in-memory retrieval to an indexed SQLite database or persistent on-disk index.

**Q8: Is the MVP ready for controlled use, or does it remain experimental?**
- **Ready for controlled use** as a local CLI assistant for technical documents, policies, and Markdown notes.

**Q9: Should the next milestone prioritize core-model training, retrieval quality, knowledge ingestion, or another identified bottleneck?**
- **Recommendation:** **Prioritize Knowledge Ingestion & Hybrid Retrieval Scaling (Batches 1061–1100).**
  - Specifically: (1) Add safe, pure-python PDF/DOCX parsers, (2) Implement SQLite FTS5 on-disk persistence for large document libraries, and (3) Train a small, lightweight neural re-ranker on domain passage pairs to complement lexical BM25 without introducing external LLMs.
