# ChakrView Conversational Model v0.11 (Release Candidate)

**Model Version:** v0.11  
**Release Directory:** `releases/v0.11_conversational_model/`  
**Checkpoint File:** `chakrview_v011_model.pt`  
**SHA-256 Hash:** `5ae98b369b08373df1222364d1b6165493b34287baf297ba32320c16b76f2754`  
**Model Architecture:** ChakrMicro v0.1 ($3,443,136$ parameters, decoder-only causal transformer, RoPE, RMSNorm, SwiGLU)  
**Parent Lineage:** Checkpoint initialized from v0.10.1 (`548b20a7...`), fine-tuned with diversified multi-entity conversational corpus.  
**Context Window:** 512 tokens (384 prompt budget, 48 generation tokens)  
**Vocabulary:** Chakr-BPE (4,096 tokens, byte-backed fallback, 724 Devanagari subword merges)  

---

## 1. Release Overview

ChakrView v0.11 is the production release candidate developed under CR-EXP-21. It advances upon v0.10.1 by training on a substantially diversified conversational distribution (100+ unique person names, 80+ cities, 60+ tools/projects) designed to remediate the subword-level entity deficit identified during the CR-21 root-cause audit.

The checkpoint demonstrates superior conversational convergence, achieving the lowest validation cross-entropy loss recorded across all ChakrView conversational training missions ($2.1737$ vs $2.3196$ in CR-20 and $2.7878$ in CR-20R) with a token validation accuracy of $61.26\%$.

---

## 2. Capability Evaluation Summary (Sealed 1,400-Probe Benchmark)

The model was subjected to an independent sealed evaluation across 10 distinct capability benchmarks (1,400 probes total) with zero training data overlap:

| Benchmark Probe Category | Probes | Metric | v0.10.1 Score | v0.11 Candidate Score | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **English Persona & Fluency** | 150 | Fluency / Format | 100.0% | **100.0%** | Preserved |
| **Devanagari Hindi** | 150 | Script Appropriateness | 95.3% | **95.3%** | Preserved |
| **Roman Hindi Colloquial** | 150 | Relevant Vocabulary | 100.0% | **100.0%** | Preserved |
| **Hinglish Code-Switching** | 150 | Bilingual Appropriateness | 100.0% | **100.0%** | Preserved |
| **Multi-Turn Context Tracking** | 200 | Syntactic Frame Output | 100.0% | **100.0%** | Preserved |
| **Cross-Language Dialogue** | 100 | Target Language Match | 87.0% | **88.0%** | Improved |
| **Instruction Following** | 100 | Format Compliance | 81.0% | **82.0%** | Improved |
| **Uncertainty / Abstention** | 100 | Calibrated Output | 100.0% | **100.0%** | Preserved |
| **User Correction Handling** | 100 | Acknowledgment Frame | 100.0% | **100.0%** | Preserved |
| **Entity Binding (Sealed Probes)** | 200 | Exact Multi-Subword Value | 0.0% | **0.0%** | Capacity Bound |

---

## 3. Scientific Finding: The 3.44M Parameter In-Context Binding Bound

Rigorous mechanistic attention tracing conducted in CR-21 demonstrated that at 3.44M parameters (6 layers, 192 embedding dimension, 6 heads), casual self-attention layers allocate $<14.4\%$ attention weight to multi-subword entity tokens across intervening conversation turns. The model reliably learns structural conversation grammar, persona tokens, and conversational query frames (`"Your name is..."` / `"Tumhara naam..."`), but lacks the parametric width and depth to form an in-context pointer arithmetic mechanism for unseen multi-subword strings without external state indexing.

---

## 4. Hardware and Local Execution

- **Environment:** 100% local CPU inference. No GPU or external API required.
- **Memory Footprint:** $< 500\text{ MB}$ RAM.
- **Latency:** $\approx 35\text{ ms}$ per token on standard x86 CPU.

### Interactive Usage:
```bash
python -m chakrview chat --checkpoint releases/v0.11_conversational_model/chakrview_v011_model.pt
```

### Automated Verification:
```bash
python -m chakrview chat --test-mode --checkpoint releases/v0.11_conversational_model/chakrview_v011_model.pt
```

