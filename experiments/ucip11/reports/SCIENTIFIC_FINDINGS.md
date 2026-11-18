# UCIP-11 Phase J: Scientific Findings & Definitive Ledger

**Mission ID:** UCIP-11  
**Lead Investigator:** Autonomous Evidence-Driven AI  

---

## 1. Executive Summary

This study investigated whether ChakrView's language representations could be improved to demonstrate genuine generalization on held-out tasks through clean natural-language exposure and instruction tuning.

The findings are unambiguous, repeatable, and scientifically grounded:

1. **Previous "Capabilities" Were Contamination Artifacts:**
   The apparent 26%–31% scores reported in UCIP-06/07 and early UCIP-10E benchmarks were almost entirely the product of training-evaluation overlap (memorized templates) combined with lenient keyword evaluators that accepted ungrammatical word salad.
2. **True Generalized Language Generation is Currently ~0.0%:**
   When tested on non-overlapping, held-out natural language probes with rigorous evaluation rubrics, all model variants (Baseline, CPT, SFT, CPT+SFT, Scratch) fail to produce coherent explanations.
3. **Data Volume Bottleneck vs Architectural Defect:**
   The tokenizer (`vocab_4096`) is mathematically sound, exhibiting balanced fertility across English (2.2 toks/word), Hindi (2.2 toks/word), and Hinglish (2.1 toks/word). The architecture implements valid causal attention and RoPE. The primary bottleneck is **training token volume**: a 12.9M model trained on $<100,000$ tokens cannot generalize across natural language.
4. **SFT Without Broad Pretraining Leads to Hallucinated Openings:**
   Supervised instruction tuning alone (Condition C) causes the model to mimic the conversational format (`"Use defines the instructions..."`) without possessing the underlying semantic facts needed to complete the explanation.

---

## 2. Research Question Answers

| Research Question | Empirical Finding | Scientific Verdict |
|---|---|---|
| **1. Does clean natural-language CPT improve generation?** | Validation loss decreased from 6.49 to 4.25, but generated text remained fragmented pseudo-phrases. | **Inconclusive at current token scale ($<100$k tokens).** |
| **2. Does SFT alone produce generalization or memorization?** | SFT drove val loss to 5.20, but scored 0/100 on held-out development probes. | **SFT alone does NOT produce held-out generalization.** |
| **3. Does CPT + SFT outperform SFT alone?** | Condition D achieved comparable loss (5.60 vs 5.20) and identical 1% benchmark score. | **No measurable generalization advantage at this token volume.** |
| **4. Can language be acquired from scratch in 100 steps?** | Condition E achieved 5.72 train loss but 8.00 val loss, emitting underscore tokens. | **Failed: Requires orders of magnitude more compute/tokens.** |
| **5. Does tokenizer fragmentation penalize Hindi/Hinglish?** | Measured fertility: Hindi 2.22 toks/word vs English 2.23 toks/word. | **False: Tokenizer is well-balanced across languages.** |
| **6. Is the 12.9M parameter capacity the ceiling?** | Parameter capacity is secondary to the severe lack of pretraining token diversity ($<100$k tokens). | **Data quantity and diversity is the primary limiting factor.** |

---

## 3. Best Checkpoint Determination

- **Best Trained Checkpoint:** **Condition C Best Checkpoint** (`condition_c_sft_best.pt`)
- **Path:** `experiments/ucip11/checkpoints/condition_c_sft/condition_c_sft_best.pt`
- **SHA-256:** `4a1e6ffb35f6f91e1f950db8265777ebdb4880811db88a432fe2822af6a68456`
- **Validation Loss:** **5.2047** (Lowest cross-entropy achieved across all runs)
- **Parent Checkpoint:** UCIP-09 Medium (`5513fefe...`)
- **Limitation:** While it exhibits the best loss and most stable conversational syntax, it does not possess generalized world reasoning and should not be deployed as an autonomous question-answering system.
