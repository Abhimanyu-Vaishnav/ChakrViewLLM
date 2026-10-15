# WP 1018–1020: Next Milestone Recommendation — Batches 1021–1060

## Milestone 1021–1060: The Local Knowledge Assistant MVP

### Objective
Stop synthetic planning research and build an end-to-end, usable, local desktop application that ingests user documents (Markdown, text, notes), answers questions with verified citations, and maintains conversation context across turns.

### Key Deliverables:
1. **Local Document Ingestion CLI**:
   - `chakrview ingest --path <docs_dir>`: Indexes local files with chunk provenance and local embedding representations.
2. **Interactive Local Assistant CLI**:
   - `chakrview chat`: Terminal interface powered by `ChakrMicro`, hybrid retrieval, and evidence-governed abstention.
3. **Verified Citation Metric**:
   - Evaluate on 50 real-world document queries, targeting >85% citation accuracy and 0% ungrounded hallucinations.
4. **Zero Plan DSL Dependency**:
   - Uses direct factual generation with source citation brackets (`[Doc1:Chunk3]`), eliminating DSL parsing overhead.
