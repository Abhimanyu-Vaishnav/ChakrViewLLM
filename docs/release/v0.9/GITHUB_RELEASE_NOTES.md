# ChakrView v0.9.0 — Official GitHub Release Notes

**Tag:** `v0.9.0`  
**Release Name:** `ChakrView v0.9.0 — Engineering Foundation & Research Preview`  
**Target Architecture:** `ChakrMicro v0.1` (3,443,136 unique parameters, tied embeddings, 6L/6H/192D, RoPE, SwiGLU)  
**Canonical Baseline Checkpoint:** `training_checkpoint_step_10000.pt` (SHA-256 `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`)  
**License:** Apache 2.0  

---

## 1. Executive Summary & Core Identity

ChakrView is an indigenous, first-principles AI research initiative exploring neural and cognitive architectures designed to execute natively on commodity CPU hardware without relying on external pretrained weights or closed-source API gateways.

**ChakrView v0.9.0** marks the transition from isolated exploratory scripts into a unified, hardened engineering foundation and research preview. This release delivers:
- **Corpus Provenance & Frequency Extraction Closure:** Completed physical Stage C token extraction across 27 binary training shards ($6,651,093$ tokens processed in $0.217\text{s}$), yielding disjoint 450-token training ($E_{train}$) and 150-token testing ($E_{test}$) candidate pools.
- **Strict Canonical Authorization Security:** Implemented cryptographic gatekeeping requiring repository-relative path binding and bit-exact SHA-256 decision validation before physical dataset extractions.
- **Epistemic Honesty & Benchmark Integrity:** Transparent benchmark reporting across Conversational (M3), General Intelligence, and Coding suites, distinguishing pure engineering infrastructure progress from general conversational capabilities.
- **Zero-Dependency CPU Runtime:** Native CPU inference adapter with sub-second cold starts and zero CUDA requirements.

---

## 2. What Changed from v0.8.0 to v0.9.0

| Dimension | ChakrView v0.8.0 | ChakrView v0.9.0 (Current) |
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

## 3. Architecture & Parameter Specification

The neural core of ChakrView is **ChakrMicro v0.1**:
- **Total Unique Parameters:** 3,443,136 ($~3.44\text{M}$)
- **Layers:** 6 Transformer decoder blocks
- **Attention Heads:** 6 query/key/value heads
- **Hidden Dimension ($d_{model}$):** 192
- **Intermediate MLP Dimension:** 512 with SwiGLU activation
- **Positional Encoding:** Rotary Position Embeddings (RoPE)
- **Vocabulary Size:** 4,096 tokens (BPE tokenizer, byte-fallback)
- **Weight Tying:** Input token embeddings and final output unembedding matrix share weights ($4096 \times 192$)
- **Context Window:** 512 tokens
- **Inference Hardware:** Commodity CPU cores (x86_64 / ARM64); zero GPU or accelerator requirement

---

## 4. Empirical Evaluation & Capability Boundary

ChakrView adheres to strict scientific demarcation: negative results and microscopic capacity bounds are reported openly without prompt cherry-picking.

### Benchmark Results (Native CPU Execution)

| Benchmark Suite | Total Tasks | Baseline (Step 10k) | SFT (Step 250) | SFT (Step 500) | Metric Focus |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Conversational Suite (M3)** | 60 | 0/60 (0.0%) | 1/60 (1.67%) | 0/60 (0.0%) | Dialogue coherence & domain recall |
| **Native Intelligence Suite** | 32 | 1/32 (3.1%) | 2/32 (6.2%) | 3/32 (9.4%) | Relational & multi-category reasoning |
| **Coding Benchmark** | 12 | 0/12 (0.0%) | 0/12 (0.0%) | 0/12 (0.0%) | Functional execution pass rate |
| — **Syntax Validity Rate** | 12 | 3/12 (25.0%) | 4/12 (33.3%) | 4/12 (33.3%) | Valid Python AST generation |
| **Neural Harness Suite** | 30 | 0/30 (0.0%) | 0/30 (0.0%) | 0/30 (0.0%) | Exact-match zero-shot QA |
| **Attractor Loop Rate** | 32 | 53.1% | 15.6% | 15.6% | Degenerate repetitive state collapse |

### Evidence-Backed Capability Assessment
1. **What Works Well:**
   - Single-token continuation and short-phrase completion within pretraining distribution.
   - Low-latency native CPU inference (~440ms cold start, ~130 tokens/sec on standard laptop CPUs).
   - Deterministic syntactic generation: produces syntactically valid Python constructs for 33.3% of basic prompts.
   - Attractor loop reduction: sliding-window n-gram blocking ($N=3$) and SFT tuning suppress repetitive token cycles.
2. **What Does NOT Work (Explicit Limitations):**
   - **General Conversational Interaction:** The model cannot engage in open-domain, multi-turn, coherent chat. It is not an alternative to frontier commercial models.
   - **World Knowledge & Factual Recall:** It cannot answer general trivia or science questions not present in its compact pretraining corpus.
   - **Functional Code Execution:** Although basic syntax is generated, 0% of coding prompts pass functional unit tests.
   - **Multi-Digit Arithmetic & Logic:** Unassisted arithmetic accuracy is $<10\%$.

---

## 5. Checksum & Integrity Manifest

Every canonical file and release document is cryptographically verified:

```
# Core Model & Checkpoint Integrity
chakrview/brain/model.py:
  SHA-256: 1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361 (BIT-EXACT)
chakrview/runtime/session.py:
  SHA-256: 9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67 (BIT-EXACT)
training_checkpoint_step_10000.pt:
  SHA-256: 96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee (BIT-EXACT)
data/experiments/vocab_4096/vocab.json:
  SHA-256: db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a (BIT-EXACT)
data/experiments/vocab_4096/merges.json:
  SHA-256: ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7 (BIT-EXACT)
docs/experiments/CAND_01_STAGE_C_EXTRACTION_AUTHORIZATION_DECISION.md:
  SHA-256: e5b433adebb53b45e8250f07cae38282e1a06839bed15fa788d16d4db5192f00 (BIT-EXACT)
```

---

## 6. Roadmap: Towards v0.9.1 and v1.0

- **v0.9.1 (Candidate 01 Training & Statistical Evaluation):**
  - Execute controlled SFT on the extracted $E_{train}$ frequency-stratified candidate corpus.
  - Run paired two-way cluster bootstrap evaluation against the canonical Step 10k baseline.
  - Test for statistically significant attractor reduction and syntax improvements without cognitive degradation.
- **v1.0 (First Stable Cognitive & Neural Milestone):**
  - Scaled architecture candidate exploration ($10\text{M}$ to $25\text{M}$ parameters) while preserving native CPU feasibility.
  - High-diversity conversational and instruction-following dataset integration.
  - Full hybrid runtime pairing the neural core with relational cognitive verification and deterministic execution sandboxes.

---

## 7. How to Publish via GitHub Web Interface / CLI

If publishing via the GitHub CLI:
```bash
gh release create v0.9.0 \
  --title "ChakrView v0.9.0 — Engineering Foundation & Research Preview" \
  --notes-file docs/release/v0.9/GITHUB_RELEASE_NOTES.md
```

If publishing via GitHub Web UI:
1. Navigate to `https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM/releases/new`.
2. Select existing tag `v0.9.0`.
3. Set Title to: `ChakrView v0.9.0 — Engineering Foundation & Research Preview`.
4. Paste the content of `docs/release/v0.9/GITHUB_RELEASE_NOTES.md` into the release description.
5. Click **Publish release**.
