# WP 993–998: Capability Matrix Against the Original ChakrView Vision

| # | Intended Vision Capability | Current Implementation Module | Mechanism Type | Weights / Data Used | Integrated in Default Route? | Empirical Evidence / Status | Missing Dependency / Blocker |
|---|:---|:---|:---:|:---:|:---:|:---|:---|
| 1 | **From-Scratch Tokenizer** | `chakrview/tokenizer/tokenizer.py` | Deterministic BPE | Vocab 4096 vocabulary | **YES** | Lossless UTF-8 roundtrip verified | Multi-digit subword splitting (`600` $\to$ `6`, `00`) |
| 2 | **Causal Neural Transformer Core** | `chakrview/brain/model.py` | Neural (ChakrMicro) | Step 10,000 checkpoint | **YES** | Pretrained step 10k loss ~2.18 | Small scale (d_model=192, 6 layers, 512 context) |
| 3 | **General Language Generation** | `chakrview/inference/` | Autoregressive sampling | Step 10,000 checkpoint | **YES** | Predicts fluent basic English sentences | Coherence decays after ~40 tokens |
| 4 | **Semantic Intent Classification** | `chakrview/runtime/neural_intent_router.py` | Neural Head | Step 250 candidate | Experimental | 65.5% calculator intent accuracy | Fails on multi-clause prompts |
| 5 | **Mathematical / Logical Reasoning** | `chakrview/cognition/` | Deterministic tool | Python `eval` sandbox | Experimental | Exact math via Python calculator | Model cannot perform multi-step mental arithmetic |
| 6 | **Multi-Step Task Planning** | `chakrview/cognition/neural_topology_planner.py` | Hybrid (Neural Span + Heuristic DAG) | Step 250 span model | Research Only | 36.4% exact match on 2-step arithmetic | Fails on variable 3-step/4-step topologies |
| 7 | **Grounded Argument Extraction** | `chakrview/brain/span_grounding.py` | Neural Span Head | Step 250 span model | Research Only | 32.9% slot correctness across frozen test | Recency attention bias towards prompt tail |
| 8 | **Persistent Episodic Memory** | `chakrview/runtime/session.py` | Deterministic / DB | SQLite / InMemory store | **YES** | Multi-turn context compaction verified | Neural anaphora resolution is rule-assisted |
| 9 | **Local Document Knowledge Ingestion**| `chakrview/cognition/federation/` | Hybrid Retrieval | Local text files | Experimental | Embeddings + BM25 keyword search | Document chunking and semantic reranking |
| 10| **Retrieval Provenance & Evidence Gate**| `chakrview/runtime/evidence_gate.py` | Deterministic rule | Token overlap / provenance | **YES** | Abstains when evidence is missing | Neural attribution verification |
| 11| **Safe Sandboxed Tool Execution** | `chakrview/runtime/tool_executor.py` | Deterministic sandbox | Subprocess / math eval | **YES** | Tool allowlist strictly enforced | Limited tool registry (calculator, text_utility) |
| 12| **Controlled Continual Learning** | `chakrview/training/continual.py` | Neural Optimizer | Experience replay buffer | Research Only | Replay mixing prevents catastrophic forgetting | Automated trigger from user corrections |
| 13| **Hardware-Adaptive Execution** | `chakrview/runtime/hardware_adaptive_engine.py` | Systems check | PyTorch CUDA/CPU | **YES** | P50 latency 6.4ms GPU, 6.1ms CPU | Quantization (INT8/FP16) not yet implemented |
| 14| **Distributed Compute Federation** | `chakrview/distributed/` | RPC / Socket | Local network cluster | Experimental | Test suite passes | Real-world cluster validation |
| 15| **Reproducible Safety & Evaluation**| `experiments/batch*/` | Python evaluation | Automated scripts | **YES** | 51/51 regression tests passed | Unified CI/CD evaluation harness |
| 16| **End-to-End Practical Application** | `chakrview/runtime/session.py` | Hybrid pipeline | Core + Tools + Memory | **PARTIAL** | Basic terminal chat with memory | Lacks polished CLI or user-facing task UI |
