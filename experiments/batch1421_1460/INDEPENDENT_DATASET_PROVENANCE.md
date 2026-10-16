# Independent Dataset Provenance

## Batches 1421–1460: Multi-Domain Held-Out Evaluation Corpus (30 Documents, 7,741 Targets)

### 1. Corpus Provenance & Category Scope

The independent evaluation corpus is stored under:
`experiments/batch1421_1460/data/corpus/`

The dataset comprises **30 distinct technical documents** containing **7,771 total tokens** (7,741 causal prediction targets), deliberately authored to span five diverse computer science and technical disciplines:

| Category | File Prefix | Document Count | Total Tokens | Target Tokens ($T-1$) | Domain Coverage |
|---|---|---|---|---|---|
| **OS & Computer Architecture** | `os_*.md`, `arch_*.md` | 6 documents | 1,515 tokens | 1,509 targets | Virtual memory paging, APIC interrupts, CFS scheduler, MESI cache coherence, TAGE branch predictor, PCIe Gen5 DMA. |
| **Python & Concurrency** | `py_*.md` | 6 documents | 1,544 tokens | 1,538 targets | CPython GIL mutex internals, Asyncio event loop, Pymalloc arenas, Bytecode dispatch loop, POSIX shared memory, Generational GC. |
| **Distributed Systems & Consensus** | `dist_*.md` | 6 documents | 1,540 tokens | 1,534 targets | Raft log replication, Vector clocks, Paxos 2PC, Two-Phase Commit, SWIM gossip failure detection, Consistent hashing rings. |
| **Mathematical & Numerical Computing** | `math_*.md` | 6 documents | 1,595 tokens | 1,589 targets | Matrix multiplication GEMM, SVD low-rank approximation, AdamW gradient descent, FFT Cooley-Tukey, MCMC Metropolis-Hastings, Power iteration. |
| **Explanatory Technical Prose** | `prose_*.md` | 6 documents | 1,577 tokens | 1,571 targets | Fault-tolerant design, Zero-trust networking, Distributed tracing, Immutable infrastructure, Streaming data ETL, LSM-Tree storage. |
| **Total** | | **30 documents** | **7,771 tokens** | **7,741 targets** | **Complete Multi-Domain Breadth** |

---

### 2. Independent Corpus Manifest & Hash Registry

Full machine-readable manifest is stored at [independent_corpus_manifest.json](file:///D:/Project/ChakrView/experiments/batch1421_1460/data/independent_corpus_manifest.json).

#### Hash Registry (Sample):
- `os_virtual_memory_paging.md`: `11c2b2b8835560b431776ce3aee69772c72b220bc0a133d59670d8a5716df1f5`
- `py_gil_multithreading_internals.md`: `e363a897369f88d227b6131cfa754966601b04618790be4fa377e3aa0fa2908f`
- `dist_raft_log_replication.md`: `afce9adf9eb9d592ee49320b9e8bb248f3316e174b29fbc99014eb3854199f36`
- `math_matrix_multiplication_gemm.md`: `019d96532d59ff6a9bfe4468f9a2fa0dc94a50b73c24b9a029587a8b303e9fe7`
- `prose_fault_tolerant_system_design.md`: `fe3c3cc83c05c317ff6774640d2f0932dd5a25e98bb4bca1d56cc33c5e0da278`
