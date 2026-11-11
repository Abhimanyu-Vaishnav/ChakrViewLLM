# CR-EXP-16 Experiment Design
## Foundational Algorithmic Capability Acquisition

**Mission Identifier:** CR-EXP-16  
**Objective:** Investigate whether ChakrMicro can acquire foundational algorithmic capabilities (exact sequence copying, key-value retrieval, variable reassignment, and state tracking) through controlled synthetic pretraining before conversational fine-tuning.  
**Auditor / Maintainer:** Autonomous Scientific Research & Engineering Team  
**Date:** November 2026  

---

## 1. Frozen Architecture Specification

ChakrMicro's architecture remains strictly fixed at canonical parameters:
- **Vocabulary Size:** 4,096 tokens (BPE)
- **Model Hidden Dimension ($d_{\text{model}}$):** 192
- **Transformer Layers ($N_{\text{layers}}$):** 6
- **Attention Heads ($N_{\text{heads}}$):** 6
- **Head Subspace Dimension ($d_{\text{head}}$):** 32
- **FFN Hidden Dimension ($d_{\text{ff}}$):** 512
- **Positional Encoding:** RoPE ($\theta = 10000.0$)
- **Context Length:** 512 tokens
- **Total Parameters:** 3,443,136
- **Tie Word Embeddings:** `True` (embedding weights shared with LM head)

---

## 2. Experimental Arms and Training Regimes

Five controlled arms will be investigated to isolate capability acquisition:

| Arm ID | Arm Description | Initial Checkpoint | Training Curriculum |
|---|---|---|---|
| **Arm A** | Frozen Pretraining Control | Canonical Stage B Step 10,000 | None (Frozen Zero-Shot Baseline) |
| **Arm B** | Sequence Copying Arm | Canonical Stage B Step 10,000 | Stage A: Sequence & Token Copying |
| **Arm C** | Key-Value Retrieval Arm | Canonical Stage B Step 10,000 | Stage B: Key-Value Lookup |
| **Arm D** | Sequential Copying + Retrieval | Canonical Stage B Step 10,000 | Stage A $\to$ Stage B Sequential |
| **Arm E** | Sequential Full Algorithmic | Canonical Stage B Step 10,000 | Stage A $\to$ Stage B $\to$ Stage C (Reassignment) |

---

## 3. Discrepancy Closures from CR-15 / CR-15A

1. **Resolution of `Dev` Token ID Discrepancy:**
   - In canonical tokenizer `data/experiments/vocab_4096/`, the string `"Dev"` maps directly to token ID `1302`.
   - Token ID `342` in `vocab_4096` decodes to the subword `"le"`. The earlier textual reference in CR-15 was a typo in documentation; both CR-15 and CR-15A used `vocab_4096` where `Dev` is single-token `[1302]`.
2. **Abstention Defect Ground Truth:**
   - On the CR-15 benchmark, unanswerable queries asked `What is the current location of Vikas?`.
   - Because models lacked abstention training, they suffered 100% false positive hallucination, emitting memorized cities rather than epistemic phrases.

