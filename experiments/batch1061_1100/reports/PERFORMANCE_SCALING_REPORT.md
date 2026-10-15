# Performance & Scaling Audit: Batches 1061–1100
## Latency Profiling, Resource Consumption & Scaling Characteristics

### 1. Cold-Start and Baseline Resource Footprint
Measurements conducted on Windows 11 (Python 3.14.7, PyTorch 2.14.0+cpu, AVX2):
- **Assistant Cold-Start Latency:** **0.02 ms** (instantaneous initialization of in-memory indices).
- **Corpus Ingestion & Index Construction Time (9 chunks):** **18.3 ms**.
- **Process Memory (RSS):** **228.9 MB** total process baseline (only 0.45 MB consumed by indexed documents and vocabulary data structures).
- **Neural Model Loading Status:** **Not invoked**. The neural core (`ChakrMicro`) is not loaded into memory or executed during standard question answering or retrieval paths.

### 2. Query Latency Breakdown (Sample Size: 250 Invocations)
| Phase | Median Latency | P95 Latency | Notes |
| :--- | :--- | :--- | :--- |
| **BM25 + Hash-Vector Hybrid Retrieval** | **0.135 ms** | **0.160 ms** | Score computation over in-memory inverted index & vector matrix. |
| **Candidate Re-ranking & Filtering** | **0.280 ms** | **0.360 ms** | Dual-token content overlap scoring & threshold checks. |
| **Evidence Gate / Contradiction Intercept** | **0.020 ms** | **0.030 ms** | Regex evaluation against query string. |
| **Total End-to-End Query (`ask()`)** | **0.444 ms** | **0.578 ms** | Total latency from query string to cited `AssistantResponse`. |

### 3. Ingestion & Query Latency Scaling Characteristics
Evaluated on synthetic Markdown corpora scaling from 10 to 500 documents:

| Documents | Chunks | Ingest & Index Time | Query Latency | Notes |
| :---: | :---: | :---: | :---: | :--- |
| **10** | 10 | 188.27 ms | 6.73 ms (cold) | First-run module cache initialization. |
| **50** | 50 | 740.13 ms | 0.72 ms | Warm execution. |
| **100** | 100 | 1,908.53 ms | 1.24 ms | Pure in-memory inverted index. |
| **200** | 200 | 7,084.29 ms | 2.06 ms | Linear scaling of lexical search. |
| **500** | 500 | 28,210.76 ms | 4.22 ms | Ingestion remains dominated by disk I/O and index rebuilding. |

### 4. Bottleneck Analysis & Performance Insights
1. **Query Execution Bottleneck:** At $<5$ ms even at 500 chunks, query latency is dominated by Python string tokenization and set intersection operations.
2. **Ingestion Bottleneck:** Ingestion time scales linearly with chunk count ($O(N)$). The current design rebuilds lexical and vector indices (`self._rebuild_indices()`) after every file ingestion. Batching index updates during directory walk will dramatically reduce multi-file indexing overhead.
3. **Retrieval-Only vs. Full System:** Retrieval represents approximately 30% of full latency, while string token filtering and session budgeting account for the remaining 70%.
