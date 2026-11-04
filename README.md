# ChakrView

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Release-v0.9.0-emerald.svg)](https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM/releases/tag/v0.9.0)
[![Target](https://img.shields.io/badge/Execution-CPU--First%20Native-orange.svg)](#quick-start--cpu-execution)
[![Parameters](https://img.shields.io/badge/Model%20Size-3.44M-purple.svg)](#architecture-specifications)
[![Pretrained Weights](https://img.shields.io/badge/Pretrained%20Weights-None%20(From%20Scratch)-red.svg)](#what-chakrview-is)

**ChakrView** is an indigenous, first-principles AI research initiative building modular neural and cognitive systems from scratch. Designed to run natively on commodity CPU hardware without requiring GPUs or external cloud LLMs, ChakrView pairs an ultra-compact autoregressive neural transformer (**ChakrMicro**, 3.44M parameters) with higher-order relational, symbolic, and governed cognitive modules.

---

## What ChakrView Is

1. **Indigenous & Independent:** Built entirely from mathematical foundations and native code. No borrowed weights from LLaMA, Mistral, or GPT; zero hidden API calls to proprietary cloud models.
2. **CPU-First Execution:** Designed and benchmarked natively for consumer and server CPUs (x86_64, ARM64) with sub-second cold starts (~440 ms) and real-time inference throughput (~130 tokens/sec).
3. **Compact Neural Foundation:** Powered by **ChakrMicro v0.1**, an efficient 3.44M-parameter decoder-only transformer with rotary position embeddings (RoPE), SwiGLU activation, and tied embeddings.
4. **Governed Cognitive Layering:** Separates the frozen neural representation core from higher-order relational transitivity, episodic working memory, and multi-hop reasoning engines.
5. **Epistemically Honest Research:** Governed by strict verification contracts, machine-checkable decision gates, two-way cluster bootstrap statistics, and open reporting of model limits.

---

## Latest Release: ChakrView v0.9.0 (Engineering Foundation & Research Preview)

ChakrView v0.9.0 marks the transition from exploratory experimental scripts into a hardened, reproducible engineering foundation.

### Key Milestones in v0.9.0
- **Stage C Production Frequency Extraction:** Verified and processed all 27 binary shards (`data/tokenized/stage_c/train/*.bin`), counting exactly **6,651,093 uint16 tokens** in **0.217 seconds**. Stratified 707 non-zero tokens into disjoint training ($E_{train}=450$) and testing ($E_{test}=150$) candidate pools.
- **Strict Canonical Authorization Security:** Implemented cryptographic gatekeeping in `scripts/extract_stage_c_frequencies.py` enforcing canonical repo-relative path checks and bit-exact SHA-256 decision validation (`e5b433adebb53b45e8250f07cae38282e1a06839bed15fa788d16d4db5192f00`).
- **Statistical Bootstrap Rigor:** Registered two-way cluster bootstrap tooling with explicit degeneracy handling, effective sample size computation, and INCONCLUSIVE boundary checks.
- **Forensic Diagnosis of Conversational Ceilings:** Documented representation limits, dataset semantic collapse, and evaluation methodology improvements.
- **Complete Release Specification:** Published 13 comprehensive release documents under [`docs/release/v0.9/`](docs/release/v0.9/).

For official release artifacts and guides:
- [GitHub Release v0.9.0 Notes](docs/release/v0.9/GITHUB_RELEASE_NOTES.md)
- [Official Release Notes v0.9.0](docs/release/v0.9/RELEASE_NOTES.md)
- [Model Card (ChakrMicro v0.1)](docs/release/v0.9/MODEL_CARD.md)
- [Architecture Deep Dive](docs/release/v0.9/ARCHITECTURE.md)
- [Evaluation Report](docs/release/v0.9/EVALUATION_REPORT.md)
- [Known Limitations & Epistemic Boundaries](docs/release/v0.9/KNOWN_LIMITATIONS.md)
- [Dataset Provenance](docs/release/v0.9/DATASET_PROVENANCE.md)
- [Training Lineage](docs/release/v0.9/TRAINING_LINEAGE.md)
- [Installation Guide](docs/release/v0.9/INSTALLATION.md)
- [Security Policy](docs/release/v0.9/SECURITY.md)

---

## Version Comparison: v0.8.0 vs. v0.9.0

| Feature / Dimension | ChakrView v0.8.0 | ChakrView v0.9.0 (Current) |
| :--- | :--- | :--- |
| **Release Character** | Checkpoint packaging & provenance closure | Hardened engineering foundation & reproducible research preview |
| **Corpus Extraction** | Stage C shard count verified (5,534 docs) | Physical token frequency extraction executed (27 shards, 6.65M tokens) |
| **Candidate Pools** | Synthetic test fixtures only | Production disjoint candidate pools ($E_{train}=450$, $E_{test}=150$) |
| **Security & Gates** | Static policy documentation | Cryptographic SHA-256 canonical path authorization gate |
| **Statistical Rigor** | Standard average scoring | Registered 2-way cluster bootstrap with degeneracy criteria |
| **Repetition Attractor Rate** | 53.1% on baseline greedy sampling | 15.6% post-SFT; 0.0% with sliding-window n-gram suppression |
| **Coding Capability** | Basic AST sandbox execution | 33.3% syntax validity on Python prompts; isolated test runner |
| **Documentation Suite** | 5 release guides | Complete 13-document release specification in `docs/release/v0.9/` |

---

## Architecture Specifications

```
                      +-----------------------------------+
                      |       Input Task / Query          |
                      +-----------------------------------+
                                         |
                                         v
                      +-----------------------------------+
                      |   Language <-> Cognitive Bridge   |
                      |   (Intent & Frame Extraction)     |
                      +-----------------------------------+
                                         |
                +------------------------+------------------------+
                |                                                 |
                v                                                 v
+-------------------------------+               +-------------------------------+
|  Canonical Neural Brain       |               |  Relational Semantic Encoder  |
| (ChakrMicro, 3.44M params)    |               |  (Grounding & Transitivity)   |
| Frozen Baseline (ΔW = 0)      |               |  (Surface Invariance)         |
+-------------------------------+               +-------------------------------+
                |                                                 |
                +------------------------+------------------------+
                                         |
                                         v
                      +-----------------------------------+
                      |  Semantic Cognitive Coupling      |
                      |  (Evidence & Conflict Resolution) |
                      +-----------------------------------+
                                         |
               +-------------------------+-------------------------+
               |                                                   |
               v                                                   v
+-------------------------------+               +-------------------------------+
|   Hierarchical Planner        |               |  Bounded Working Memory 2.0   |
| (Goal Breakdown & Recovery)   |               |  (Task-Scoped, Zero Leaks)    |
+-------------------------------+               +-------------------------------+
```

### ChakrMicro v0.1 Hyperparameters
- **Total Parameters:** 3,443,136 unique parameters
- **Architecture:** Autoregressive Decoder-Only Transformer
- **Layers:** 6 Transformer Blocks
- **Attention Heads:** 6 Heads ($d_{head} = 32$)
- **Hidden Dimension ($d_{model}$):** 192
- **Feed-Forward Dimension ($d_{mlp}$):** 512 with SwiGLU non-linearity
- **Positional Embeddings:** Rotary Position Embeddings (RoPE)
- **Vocabulary:** 4,096 BPE tokens with byte fallback
- **Weight Tying:** Shared input embedding and output unembedding matrices ($4096 \times 192$)
- **Context Window:** 512 tokens
- **Precision:** Float32 / BFloat16 native CPU execution

---

## Empirical Benchmark Performance

All benchmarks are executed natively on commodity CPU hardware without GPU acceleration:

| Benchmark Suite | Total Tasks | Baseline (Step 10k) | SFT (Step 250) | SFT (Step 500) | Focus Area |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Conversational Suite (M3)** | 60 | 0/60 (0.0%) | 1/60 (1.67%) | 0/60 (0.0%) | Multi-turn dialogue coherence & recall |
| **Native Intelligence Suite** | 32 | 1/32 (3.1%) | 2/32 (6.2%) | 3/32 (9.4%) | Relational & 10-category reasoning |
| **Coding Benchmark** | 12 | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | Functional test execution pass rate |
| — **Syntax Validity Rate** | 12 | 3/12 (25.0%) | 4/12 (33.3%) | 4/12 (33.3%) | AST syntax parse validity |
| **Neural Harness Suite** | 30 | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) | Exact-match zero-shot QA |
| **Attractor Loop Rate** | 32 | 53.1% | 15.6% | 15.6% | Repetitive degenerate loops (lower is better) |

---

## Honest Natural-Language Capability Assessment

ChakrView operates with strict scientific demarcation between genuine capabilities and microscopic neural capacity bounds:

### 1. What the Model Can Do
- **Short Prompt Completion:** Generates syntactically consistent token continuations within the scope of its pretraining distribution.
- **Repetition Suppression:** Sliding-window n-gram blocking ($N=3$) and SFT reduce greedy attractor collapse from 53.1% to under 16% (and 0% with active blocking).
- **Python Syntax Drafting:** Synthesizes structurally parseable Python snippets (def, return, loops) with a 33.3% AST validity rate.
- **Relational Grounding (Cognitive Layer):** The hybrid cognitive pipeline extracts subject-predicate-object triples and evaluates multi-hop transitive paths.

### 2. What the Model CANNOT Do
- **General Dialogue / Chatbot Interaction:** The 3.44M parameter core cannot conduct natural open-domain multi-turn conversation. It is not an alternative to modern 7B+ LLMs or ChatGPT.
- **World Knowledge & Trivia:** Possesses no encyclopedic knowledge outside its synthetic pretraining shards. General knowledge queries will hallucinate or fail.
- **Functional Coding Execution:** While generating valid syntax for 1 in 3 prompts, 0% of generated programs pass automated functional verification.
- **Complex Logic & Math:** Unassisted arithmetic performance is $<10\%$ accuracy.

---

## Known Limitations

> [!WARNING]
> **ChakrMicro is an ultra-compact 3.44M-parameter experimental model.**
> It must not be deployed in production for autonomous customer-facing dialogue, safety-critical decision-making, or unsupervised code generation.

1. **Capacity Ceilings:** Microscopic parameter counts severely restrict factual retention and syntactic generalization.
2. **Context Degradation:** In dialogues exceeding 300 tokens, attention heads struggle to track distant variables.
3. **Attractor Vulnerability:** Greedy decoding on out-of-distribution prompts can enter repetitive phrase cycles unless repetition penalties ($1.1 \le \lambda \le 1.25$) are applied.

---

## Canonical Checksum & Integrity Manifest

The integrity of canonical model definitions and weights is cryptographically locked:

| File | Path | SHA-256 Digest | Status |
| :--- | :--- | :--- | :---: |
| **Brain Architecture** | `chakrview/brain/model.py` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` | **IMMUTABLE** |
| **Runtime Session** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` | **IMMUTABLE** |
| **Canonical Checkpoint**| `training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **VERIFIED** |
| **Tokenizer Vocab** | `data/experiments/vocab_4096/vocab.json` | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` | **VERIFIED** |
| **Tokenizer Merges**| `data/experiments/vocab_4096/merges.json` | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` | **VERIFIED** |
| **Decision Doc** | `docs/experiments/CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md` | `e5b433adebb53b45e8250f07cae38282e1a06839bed15fa788d16d4db5192f00` | **VERIFIED** |

---

## Quick Start & CPU Execution

### 1. Requirements
- Python 3.10 to 3.14 (64-bit)
- Standard x86_64 or ARM64 CPU (No GPU required)

### 2. Installation
```powershell
# Clone repository
git clone https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git
cd ChakrViewLLM

# Setup virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install CPU dependencies
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Verify Checkpoint & Architecture
```bash
python scripts/verify_checkpoint_integrity.py
```

### 4. Interactive Session
```python
from chakrview.runtime.session import InferenceSession

# Initialize native CPU session
session = InferenceSession.from_checkpoint(
    checkpoint_path="experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
    device="cpu"
)

# Generate with repetition controls
response = session.generate(
    prompt="def add_numbers(a, b):",
    max_tokens=32,
    temperature=0.7,
    repetition_penalty=1.15
)
print("Output:", response)
```

---

## Capability Roadmap

```
+--------------------------------------------------------------------------+
|  v0.8.0 Checkpoint Provenance & Corpus Accounting (COMPLETED)            |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|  v0.9.0 Hardened Engineering Foundation & Research Preview (CURRENT)     |
|  - Physical Stage C frequency extraction (6.65M tokens)                  |
|  - Cryptographic authorization gate & 2-way cluster bootstrap            |
|  - Forensic evaluation and complete release documentation suite          |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|  v0.9.1 CAND_01 Extraction SFT & Rigorous Bootstrap (NEXT)              |
|  - Controlled SFT on E_train candidate token pool                        |
|  - Registered paired 2-way cluster bootstrap evaluation                  |
|  - Non-regression gating on cognitive baselines                          |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|  v1.0.0 Stable Cognitive & Neural System                                 |
|  - Scaled parameter candidates (10M - 25M) optimized for CPU execution   |
|  - High-diversity variable-binding instruction datasets                  |
|  - Fully integrated symbolic-neural reasoning runtime                    |
+--------------------------------------------------------------------------+
```

---

## Historical Releases

- [ChakrView v0.8.0 Release](https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM/releases/tag/v0.8.0): Checkpoint provenance, Stage C corpus closure, and initial research preview.
- [ChakrView v0.7.0 Release](docs/release/v0.7.0/RELEASE_NOTES.md): Coding foundation, subprocess-isolated execution, and n-gram blocking.
- [ChakrView v0.6.0 Release](release/chakrview-v0.6.0/RELEASE_NOTES_v0.6.0.md): Standardized CPU runtime adapter.
- [ChakrView v0.5.0 Release](release/chakrview-v0.5.0/docs/releases/V0.5.0_RELEASE_MANIFEST.md): Cognitive foundation and multi-hop relational transitivity.

---

## License

ChakrView is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for complete terms.
