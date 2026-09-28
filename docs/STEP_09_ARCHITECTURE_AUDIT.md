# Step 9 — Architectural Audit & Adaptive Brain Framework Design

**Document Version**: 1.0.0  
**Milestone**: Step 9 — Establish Adaptive Brain Architecture Foundation  
**Status**: RATIFIED & ARCHITECTURALLY COMMITTED  
**Date**: 2026-09-28  
**Model Core**: ChakrMicro v0.1 (3,443,136 parameters, frozen)  
**Tokenizer**: Byte-Level BPE ($V=4096$, frozen)  
**Corpus State**: Authentic Stage C multi-domain corpus (5,534 documents, ~7.7M tokens, frozen)  
**Baseline & Pre-Training State**: Steps 7 & 8 verified and frozen (12991e1)  

---

## Executive Summary

ChakrView has completed its foundational pre-training phase:
* **Step 7** established the first reproducible real-corpus baseline (500 steps, 512,000 tokens).
* **Step 8** scaled ChakrMicro v0.1 across 1 complete epoch (6,478 optimizer steps, 6,633,472 tokens), dropping full validation loss to **4.8062** (perplexity: 122.27) and full unseen test loss to **4.7828** (perplexity: 119.44).

However, ChakrView was never conceived as a single standalone monolithic chatbot. Its long-term mandate is to serve as an **extensible, indigenous AI cognitive engine ("brain")** capable of powering:
1. Universal base intelligence
2. Domain-specific variants (legal, medical, scientific, administrative)
3. Organization- and company-specific deployments
4. User-personalized local instances
5. Specialized skill profiles (coding, formal reasoning, mathematics, structured analysis)
6. Dynamic external knowledge ingestion (RAG, documentation, manuals)
7. Offline, edge, and power-constrained computing environments

To achieve this without destabilizing the neural core, **knowledge, skills, memory, tools, and hardware adaptation must NEVER be permanently tangled into model weights**. The neural core must remain a compact, reliable, deterministic cognitive transformer surrounded by replaceable, auditable, and secure software layers.

This document presents a comprehensive forensic audit of the existing codebase, defines the layered target architecture, details threat and integrity models, and establishes the foundational contracts for the next generation of ChakrView.

---

## 1. Forensic Audit of Current Architecture

The ChakrView repository currently consists of four primary functional engines developed across Steps 0 through 8:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          CHAKRVIEW REPOSITORY                          │
├───────────────────┬───────────────────┬─────────────────┬──────────────┤
│ chakrview/brain/  │ chakrview/        │ chakrview/      │ chakrview/   │
│ (Neural Core)     │ tokenizer/        │ corpus/         │ training/    │
├───────────────────┼───────────────────┼─────────────────┼──────────────┤
│ - ModelConfig     │ - ChakrTokenizer  │ - Document      │ - Trainer    │
│ - ChakrMicro      │ - BPE Trainer     │ - Normalizer    │ - Streaming  │
│ - TransformerBlock│ - Encoder/Decoder │ - Manifest      │   Dataset    │
│ - Attention/MHA   │ - Byte Mapping    │ - Binary Sharder│ - CausalLoss │
│ - SwiGLU FFN      │ - Special Tokens  │ - Splitter      │ - Checkpoint │
│ - RMSNorm & RoPE  │ - Serialization   │ - Validators    │ - Optimizer  │
└───────────────────┴───────────────────┴─────────────────┴──────────────┘
```

### 1.1 Boundaries & Module Inspection

1. **Model / Core Boundary (`chakrview/brain/`)**:
   * *Structure*: Pure PyTorch `nn.Module` implementation adhering to strict causal transformer principles.
   * *State*: 6 decoder blocks, $d_{\text{model}}=192$, 6 heads ($d_{\text{head}}=32$), $d_{\text{ff}}=512$, $T_{\text{max}}=512$, Pre-RMSNorm, RoPE, SwiGLU, tied embeddings, bias-free.
   * *Input/Output*: Strictly receives token IDs $[B, T]$ and outputs logits $[B, T, 4096]$.
   * *Coupling*: Highly self-contained. Depends only on `chakrview/brain/config.py`.

2. **Tokenizer Boundary (`chakrview/tokenizer/`)**:
   * *Structure*: Indigenous Byte-Level BPE tokenizer with 256 byte primitives, 3 special tokens (`<BOS>=0`, `<EOS>=1`, `<PAD>=2`), and 3,837 learned merges ($V=4096$).
   * *Input/Output*: Encodes Unicode strings to token IDs; decodes token IDs to lossless UTF-8 text.
   * *Coupling*: Decoupled from the neural core. The neural core receives integers; it does not import or know about the tokenizer.

3. **Pre-Training Infrastructure Boundary (`chakrview/training/`)**:
   * *Structure*: Streaming `uint16` binary shard dataset, causal language modeling collator, atomic checkpointing with history pruning, AdamW optimizer with 2D/1D parameter splitting, cosine learning rate scheduler, and metrics tracking.
   * *Coupling*: Tightly coupled to offline pre-training across pre-tokenized binary files. Does not support interactive sequence generation, step-by-step KV-cached decoding, or runtime dynamic context injection.

4. **Corpus Pipeline Boundary (`chakrview/corpus/`)**:
   * *Structure*: Document-level ingestion, Unicode normalization, document deduplication, SHA-256 manifest generation, deterministic train/val/test splitting, and `uint16` binary shard serialization.
   * *Coupling*: Static, batch-oriented data preparation. No dynamic document parsers (PDF, DOCX, HTML) or vector indexing primitives exist.

5. **Inference / Generation Boundary (Ad-Hoc in `scripts/`)**:
   * *Current State*: Generation was implemented inside `scripts/run_stage_c_baseline_experiment.py` and `scripts/run_stage_c_full_epoch_experiment.py` as an ad-hoc function `generate_text_sample()`.
   * *Coupling*: Tightly bound to individual experiment runners. No public generation API, no KV caching, no beam search, no top-$p$/top-$k$ sampling controls, and no prompt templating engine exists in the package itself.

6. **Configuration Boundary (`configs/` and `chakrview/config.py`)**:
   * *Current State*: Split between architectural configuration (`ModelConfig`, `ChakrConfig`) and training configurations (`PretrainingConfig`, YAML/JSON files).
   * *Coupling*: Training configs explicitly declare paths to dataset directories and experiment output folders.

---

## 2. Existing Reusable Components

The following components are robust, well-tested, and ready for reuse in the broader architecture:
1. **`ChakrMicro`**: The foundational 3.44M-parameter decoder core is mathematically verified, zero-leak causal, initialization-healthy, and CPU-efficient.
2. **`ChakrTokenizer`**: The $V=4096$ byte-level BPE tokenizer is lossless, deterministic, and verified across diverse scripts (Devanagari, Latin, punctuation, code).
3. **`CheckpointManager`**: The atomic two-phase write pattern (`.tmp` $\to$ `os.replace`), state dictionary serialization, and latest-pointer management provide the foundation for safe state management.
4. **`ResourceMonitor`**: Process RSS RAM and CPU tracking utilities in `chakrview/training/monitoring.py`.
5. **`ShardWriter` & `verify_shard_integrity()`**: Fast binary reading and SHA-256 cryptographic verification of sharded data.

---

## 3. Current Coupling Points & Architectural Limitations

1. **Pre-Training Monolith**: The current `Trainer` assumes training happens on contiguous fixed-length token sequences ($T=512$) streamed sequentially from binary shards. It cannot accept dynamic prompt-response pairs, supervised fine-tuning masks, or adapter-specific gradient updates.
2. **Missing Inference Layer**: Generation logic is duplicated across one-off experiment scripts rather than living as a reusable runtime module.
3. **Absence of Memory / Context Abstractions**: The model accepts raw integer sequences. There is no concept of a "system prompt", "user turn", "retrieved knowledge segment", or "memory buffer".
4. **No Modular Hardware Abstraction**: System profiling is currently performed on an ad-hoc basis in script headers (`psutil.virtual_memory()`, `torch.get_num_threads()`) rather than informing runtime execution plans.
5. **Monolithic Experiment Checkpoints**: Checkpoints currently store the complete model weights, optimizer, and scheduler in a single `.pt` file. There is no mechanism to store lightweight adapter weights, skill configurations, or external knowledge references separately from base weights.

---

## 4. Current Technical Debt

1. **O(N^2) Autoregressive Generation**: The ad-hoc generation function repeatedly re-evaluates the entire prompt sequence on every token generation step because `ChakrMicro` lacks an autoregressive KV-cache decoding mode.
2. **Hardcoded Sequence Length Assertions**: Multiple scripts and collators hardcode sequence length to 512, making adaptive shorter context windows (e.g., $T=128$ for low-RAM devices) awkward without manual overrides.
3. **Duplicated Test Logic**: Smoke test prompt definitions and generation loops are duplicated across baseline and full-epoch runners.
4. **Scattered Manifests**: Dataset manifests live in `data/manifests/`, experiment logs live in `data/experiments/`, and checkpoints live in `checkpoints/`, with no unified top-level brain manifest linking model weights to corpus provenance and tokenizer versions.

---

## 5. Component Lifecycle Governance

To preserve the scientific integrity of ChakrView, components are classified into three governance categories:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   COMPONENT GOVERNANCE CLASSIFICATION                  │
├─────────────────────┬───────────────────────┬──────────────────────────┤
│ STRICTLY FROZEN     │ STABLE INTERFACES     │ PLUGGABLE / REPLACEABLE  │
├─────────────────────┼───────────────────────┼──────────────────────────┤
│ - ModelConfig spec  │ - Tokenizer API       │ - Knowledge Sources      │
│ - ChakrMicro v0.1   │ - Brain Model Forward │ - Document Retrievers    │
│   architecture      │ - Checkpoint Format   │ - Skill Profiles         │
│ - Tokenizer Merges  │ - Dataset Iterators   │ - Adaptation Modules     │
│   (3,837 merges)    │ - Hardware Profiling  │ - Context Formatter      │
│ - Stage C Corpus    │ - Proposal Lifecycles │ - Runtime Execution Plan │
│   (32 binary shards)│ - Health/Integrity    │ - Safety/Quarantine      │
│ - Step 7/8 Records  │   Contracts           │   Policies               │
└─────────────────────┴───────────────────────┴──────────────────────────┘
```

1. **Components That Must Remain Strictly Frozen**:
   * `chakrview/brain/`: Layer counts, dimensions, attention mechanisms, normalization, and tied embedding topology.
   * `chakrview/tokenizer/`: Byte-Level BPE merge table, special token IDs, and vocabulary size ($V=4096$).
   * `data/tokenized/stage_c/`: The 32 binary shards of authentic Stage C data.
   * Checkpoints and experiment logs for Steps 7 and 8.

2. **Components That Should Become Formal Interfaces**:
   * `BrainCore`: Abstract base definition of an autoregressive causal language model.
   * `TokenizerInterface`: Abstract contract for token serialization and decoding.
   * `KnowledgeSource`, `KnowledgeIndex`, `Retriever`: Abstract interfaces for dynamic information injection.
   * `Skill`, `SkillRegistry`: Abstract contracts for capabilities and prompt/tool policies.
   * `HardwareProfile`, `RuntimePlanner`: System-aware execution planning.
   * `HealthCheck`, `IntegrityVerifier`, `RollbackManager`: System integrity, verification, and quarantine.

3. **Components That Should Be Completely Replaceable**:
   * External document stores and vector databases.
   * Prompt templates and system policies.
   * Fine-tuning adapters (LoRA / prefix tuners).
   * Tool execution drivers and API dispatchers.
   * Hardware execution backends (CPU native, ONNX Runtime, GGML/llama.cpp, GPU).

---

## 6. Target Multi-Layer Architecture

To meet the requirements of universal intelligence, domain specialization, local offline deployment, and non-entangled knowledge, ChakrView adopts the following layered structure:

```
                        ┌──────────────────────────┐
                        │       USER / APP         │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │    CHAKRVIEW RUNTIME     │
                        │ orchestration / policies │
                        └────────────┬─────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
┌────────▼────────┐         ┌────────▼────────┐         ┌────────▼────────┐
│ Knowledge Layer │         │   Skill Layer   │         │  Memory Layer   │
│ documents / RAG │         │coding/reasoning │         │  user / session │
└────────┬────────┘         └────────┬────────┘         └────────┬────────┘
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │        BRAIN CORE        │
                        │        ChakrMicro        │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │  ADAPTATION / LEARNING   │
                        │  fine-tuning / adapters  │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │ SAFETY / INTEGRITY LAYER │
                        │ verify / rollback / heal │
                        └──────────────────────────┘
```

---

## 7. Knowledge Separation Design (Phase B)

Knowledge must never be baked irrevocably into model weights for company, domain, or user applications. 

### 7.1 Taxonomy of Knowledge
1. **Parametric Knowledge**: High-level statistical patterns, syntax, lexical grammar, and general priors acquired during base pre-training across multi-domain tokens.
2. **External Knowledge**: Heterogeneous documents that change frequently:
   * PDF (technical manuals, legal contracts)
   * DOC / DOCX (business reports, specifications)
   * TXT / MD (developer documentation, notes)
   * CSV / XLS / XLSX (tabular data, inventories)
   * JSON / HTML (APIs, web extracts, documentation dumps)
   * Local Directory Hierarchies
3. **Domain Knowledge**: Proprietary organizational data (internal company wikis, policies, engineering playbooks) requiring strict access boundaries.
4. **User Knowledge**: User-specific preferences, active session memory, and permitted user notes.

### 7.2 Non-Parametric Specialization Principle
A company-specific or user-specific ChakrView instance answers questions using company documents **WITHOUT requiring retraining or weight fine-tuning**. 

```
User Query
    │
    ▼
Retriever ──[Queries KnowledgeIndex]──► Returns Top-K KnowledgeChunks
    │
    ▼
ContextProvider ──[Injects within Token Budget]──► Structured Context Window
    │
    ▼
Brain Core (Frozen ChakrMicro v0.1) ──► Factual, Grounded Completion
```

When adaptation *is* eventually desired (e.g. style alignment or domain jargon), it is achieved via lightweight external adapters (LoRA / prefix embeddings) rather than destructive full-model retraining.

---

## 8. Skill Separation Design (Phase C)

A skill is a specialized operational capability (e.g. Python coding, GSM8K-style step-by-step arithmetic, legal document summarizing, JSON extraction) that can be activated on demand.

### 8.1 The Composition Formula
$$\text{Specialized ChakrView Instance} = \text{Base Brain} + \text{Skill} + \text{Knowledge} + \text{Memory} + \text{Tools}$$

A single frozen base brain instance can serve a software engineer, a financial analyst, or a customer support agent simply by switching the registered skill and knowledge bindings.

### 8.2 Skill Manifestation Modes
Skills can be implemented through multiple non-exclusive mechanisms:
1. **Prompt Policies**: Few-shot exemplars, structured output schemas, and domain persona framing.
2. **Tool Policies**: Restricting available external capabilities (e.g. math calculator, read-only file viewer).
3. **Contextual Adapters**: Small parameter tensors (e.g. LoRA matrices on attention projections) that adjust output distribution without touching base weights.
4. **Specialized Diagnostic Suites**: Regression test batteries tailored specifically to measure skill fidelity.

---

## 9. Controlled Self-Improvement Architecture (Phase D)

The system must **NEVER** autonomously or blindly rewrite its own neural weights or codebase. Self-improvement must operate as a formal, human-auditable engineering workflow with explicit proposal tracking.

### 9.1 The Controlled Self-Improvement Loop

```
┌─────────┐     ┌──────────┐     ┌───────────┐     ┌─────────────┐
│ Observe │ ──► │ Evaluate │ ──► │ Identify  │ ──► │   Propose   │
│  Logs   │     │  Metrics │     │ Weakness  │     │ Improvement │
└─────────┘     └──────────┘     └───────────┘     └──────┬──────┘
                                                          │
┌──────────────┐     ┌────────────┐     ┌──────────────┐  │
│ Record       │ ◄── │ Immutable  │ ◄── │   Promote    │  │
│ Provenance   │     │ Checkpoint │     │  Candidate   │  │
└──────────────┘     └────────────┘     └──────▲───────┘  │
                                               │          │
┌──────────────┐     ┌────────────┐     ┌──────┴───────┐  │
│    Reject    │ ◄── │ Comparison │ ◄── │  Regression  │ ◄┘
│ / Quarantine │     │ Gate / OK  │     │  Evaluation  │ (Isolated Sandbox)
└──────────────┘     └────────────┘     └──────────────┘
```

### 9.2 Distinction of Change Types
The system strictly segregates updates by risk and scope:
* `KNOWLEDGE`: Updating vector indexes, adding documents (Low Risk, no code/weight change).
* `SKILL`: Updating prompt templates or tool permissions (Low Risk).
* `CONFIGURATION`: Tuning context length, batch size, or sampling temperatures (Low Risk).
* `ADAPTER`: Modifying or training lightweight LoRA weights (Medium Risk).
* `WEIGHTS`: Updating or fine-tuning base neural model weights (High Risk, requires extensive regression).
* `CODE`: Modifying runtime software or execution logic (Critical Risk, strictly prohibited autonomously; requires human developer review and commit).

---

## 10. Layered Software Integrity & Self-Healing (Phase E)

ChakrView rejects the notion of "AI magic" for security. Self-healing is treated as a deterministic, layered software engineering discipline: **restoration to a verified known-good state**.

```
                [ Candidate Update / Artifact ]
                               │
                               ▼
               ┌───────────────────────────────┐
               │ 1. Cryptographic Verification │
               │    SHA-256 Digest & Schema    │
               └───────────────┬───────────────┘
                               │ PASS
                               ▼
               ┌───────────────────────────────┐
               │ 2. Isolated Sandboxed Run     │
               │    No OS / Shell Mutations    │
               └───────────────┬───────────────┘
                               │ PASS
                               ▼
               ┌───────────────────────────────┐
               │ 3. Core Health Tests          │
               │    NaN / Inf / Tensor Checks  │
               └───────────────┬───────────────┘
                               │ PASS
                               ▼
               ┌───────────────────────────────┐
               │ 4. Full Regression Suite      │
               │    Zero Broken Capabilities   │
               └───────────────┬───────────────┘
                               │ PASS
                               ▼
               ┌───────────────────────────────┐
               │ 5. Promotion & State Record   │
               │    Atomic Update to Active    │
               └───────────────────────────────┘

[ ANY FAILURE AT STAGES 1-4 ]
         │
         ▼
┌─────────────────────────────────┐
│ Quarantine Candidate Artifact   │
│ Immediate Automatic Rollback    │
│ Generate Forensic Failure Audit │
└─────────────────────────────────┘
```

### Key Pillars of Software Integrity:
1. **Artifact Hashing**: Every model weight file, tokenizer merge table, configuration, and binary shard has an authoritative SHA-256 digest.
2. **Atomic Writes**: All state mutations use two-phase writes (`.tmp` followed by `os.replace`).
3. **Strict Sandboxing**: Evaluation of candidate models or adapters occurs in memory without permissions to execute shell commands, alter files, or access the network.
4. **Automated Quarantine**: Any artifact failing a health check is immediately moved to `quarantine/` with a diagnostic report.
5. **Separation of Execution from Update Authority**: The inference runtime has zero write privileges to its own model weights or executable code.

---

## 11. Hardware Adaptation Framework (Phase F)

ChakrView is intended to execute locally across varied hardware environments without code alteration:
* Legacy 32-bit / 64-bit older PCs with limited instructions
* Modern multi-core CPUs (Intel Core i5/i7/i9, AMD Ryzen)
* ARM SBCs and embedded processors (Raspberry Pi, Apple Silicon)
* High-end workstations with dedicated GPUs

### 11.1 Probing & Planning Pipeline
```
HardwareCapabilityDetector
        │ (Probes CPU cores, physical RAM, GPU VRAM, OS)
        ▼
HardwareProfile
        │ (Structured snapshot of system capacities)
        ▼
RuntimePlanner
        │ (Computes constraints against ModelConfig)
        ▼
ModelExecutionPlan
        │
        ├── Context Length (e.g. 128 for low-RAM vs 512 for full)
        ├── Batch Size (1 for interactive vs higher for batch)
        ├── Thread Count (OMP/MKL optimization)
        ├── Precision (FP32 baseline, BF16/FP16 if supported)
        ├── Memory Ceiling (MB RSS safety limit)
        └── KV Cache Mode (Full vs Sliding Window)
```

---

## 12. Versioning & Lineage Hierarchy (Phase G)

To avoid duplicating 3.44M-parameter weight matrices for every specialized deployment, ChakrView establishes an explicit version hierarchy:

```
                      ChakrView Core v0.1.0
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
Universal v0.1.0         Coding v0.1.0           Reasoning v0.1.0
        │                       │
        ▼                       ▼
Company-A v0.1.0         Company-B v0.1.0
        │
        ▼
Company-A v0.2.0 (Updated Knowledge & Skill Policies)
```

Each derived version is declared in a `BrainVersionManifest` storing:
* `version_id`: Unique semantic identifier
* `profile_type`: BrainProfileType (UNIVERSAL, CODING, REASONING, ENTERPRISE, USER_SPECIFIC, RESEARCH)
* `base_model_version`: Immutable base model identifier (e.g. `chakrmicro-v0.1`)
* `base_model_hash`: SHA-256 digest of underlying model weights
* `tokenizer_checksum`: SHA-256 digest of tokenizer
* `parent_version_id`: Lineage parent (or `None` for root)
* `lineage`: Ordered list of ancestral version IDs
* `adapter_manifests`: Dictionary mapping adapter IDs to their respective SHA-256 hashes
* `skills`: List of registered skill identifiers
* `knowledge_indexes`: List of associated knowledge index identifiers
* `runtime_config_hash`: SHA-256 digest of runtime settings

---

## 13. Threat Model & Security Boundaries (Phase H)

A rigorous defense-in-depth threat model identifies 12 distinct attack surfaces and their engineering mitigations:

| Threat ID | Threat Name | Attack Surface | Trust Boundary | Possible Impact | Defensive Mitigation | Detection Mechanism | Recovery Action |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T-01** | Malicious Model Artifact | Checkpoint loading (`torch.load`) | Storage $\to$ Process Memory | Arbitrary code execution via pickle | Restrict `weights_only=True` or use safetensors; verify SHA-256 prior to loading | Hash mismatch; unpickler exception | Quarantine file; rollback to known-good checkpoint |
| **T-02** | Poisoned Training Data | Corpus ingestion pipeline | External data $\to$ Shard Writer | Degraded model alignment or backdoor triggers | Strict heuristic filtering, deduplication, and schema validation | Distributional divergence in validation perplexity | Discard poisoned shards; re-ingest from verified sources |
| **T-03** | Compromised Adapter | Dynamic LoRA loading | Extension $\to$ Neural Core | Subversion of model output for specific prompts | Checksum verification; evaluate against invariant regression battery | Regression failure on standard baseline test | Unload adapter; blacklist hash; alert admin |
| **T-04** | Modified Configuration | JSON/YAML config files | Filesystem $\to$ Runtime | Excessive memory allocation or unstable LR | Immutable config dataclasses; validate ranges in `__post_init__` | Invariant assertion failure during initialization | Load default configuration; fail closed |
| **T-05** | Malicious Ingested Document | Document parser / text extractor | External File $\to$ Knowledge Base | Buffer overflow or parser exploit | Safe text-only readers; strict character length and encoding limits | Parser exception; out-of-bounds token count | Quarantine document; log parsing error |
| **T-06** | Malicious Web Content | Web crawler / HTTP fetcher | Internet $\to$ Ingestion Engine | Ingestion of exploit payloads or malicious HTML | Stripping HTML tags; no JS execution; domain allowlisting | RegEx sanitation failures; unapproved scheme | Block source; log URL violation |
| **T-07** | Prompt Injection via RAG | Retrieved text chunks | Knowledge Store $\to$ Model Context | Model hijacked into ignoring user instructions | Clear demarcation delimiters; system prompt precedence; sanitize chunks | Keyword canary tokens; output format validation | Suppress injected chunk; penalize retrieval score |
| **T-08** | Tool Abuse | Model-invoked tool dispatcher | Brain Output $\to$ OS / Filesystem | Unauthorized file modification or shell execution | Strict capability restrictions; no shell access; read-only sandboxes | Tool permission violation; policy monitor | Terminate tool call; log security event |
| **T-09** | Unauthorized Self-Update | Improvement loop execution | Autonomous Agent $\to$ Production Weights | Unaudited or regressed model promoted | Mandatory approval gate; read-only production directories | Modification attempt on write-protected files | Rollback candidate; raise critical alert |
| **T-10** | Rollback Bypass | Integrity verification logic | Update Manager $\to$ File Pointer | Persisting broken or regressed system state | Atomic pointer swap; verify candidate before updating pointer | Post-swap health check failure | Atomic restoration of previous pointer file |
| **T-11** | Corrupted Checkpoint | Incomplete write during power loss | Disk $\to$ Checkpoint Directory | Process crash on restart | Atomic two-phase write (`.tmp` $\to$ `os.replace`) | Missing or truncated `.pt` file; SHA mismatch | Restore from latest valid pointer in `latest_checkpoint.json` |
| **T-12** | Privilege Escalation | Runtime host process | User $\to$ OS Kernel | Host compromise via model service | Run process with least privilege (non-root, restricted user) | OS access audit logs; process permission violations | Kill process; terminate container |

---

## 14. Testing & Verification Strategy (Phase I)

The testing framework expands into 8 structured layers:
1. **Unit Tests**: Verification of individual dataclasses, hashing functions, and validators.
2. **Integration Tests**: Verification of context injection, skill registration, and planning loops.
3. **Artifact Integrity Tests**: Verification that all model checkpoints, tokenizer merge tables, and corpus shards match their recorded SHA-256 digests.
4. **Model Regression Tests**: Invariant checks ensuring ChakrMicro parameter count ($3,443,136$), vocabulary size ($4,096$), and deterministic forward outputs remain exact.
5. **Knowledge Retrieval Tests**: Correct top-k ranking and token budget enforcement for context providers.
6. **Skill Policy Tests**: Ensuring tools and prompts obey declared permissions.
7. **Rollback & Quarantine Tests**: Simulating corrupt checkpoints and verifying automatic quarantine and recovery.
8. **Hardware Profiling Tests**: Ensuring graceful fallback on memory-constrained systems.

---

## 15. Next Allowed Engineering Phase

With Step 9 establishing the foundational design and runtime abstractions:
* **Neural Core**: ChakrMicro v0.1 remains strictly frozen.
* **Tokenizer**: $V=4096$ remains strictly frozen.
* **Corpus**: Stage C remains strictly frozen.
* **Immediate Allowed Implementation**: Implement the safe, modular, type-checked Python interfaces under `chakrview/runtime/` with comprehensive unit tests in `tests/`.
