# Step 460: Generation Attractor & Repetition Investigation Report

**Wave**: 460  
**Target Milestone**: Scientific Diagnosis of Phrase Loops and Generation Degeneration  
**Candidate Evaluated**: `master_candidate_step1500.pt` (SHA: `1c1d126caa4b168831fa15138b225c1f2d0fcc58906273e3ea89f06c5b021ffa`)  
**Status**: COMPLETE / VERIFIED  

---

## 1. Objective

To scientifically determine the root cause of the "phrase-loop attractor" phenomenon (e.g. `"of the most of the most..."`) observed in intermediate language candidates, and isolate **training distribution effects** from **decoding effects**.

---

## 2. Controlled Decoding Sweep Across 10 Prompts

We evaluated six decoding regimes across standard bilingual prompts (`The sky is`, `India is`, `Python is a`, `The capital of India is`, `Two plus two equals`, `ChakrView is`, `मेरा नाम`, `भारत एक`, `दिल्ली भारत की`, `The sun rises in the`):

| Decoding Strategy | Consecutive Repetition Rate | Distinct-2 ($D_2$) | Distinct-3 ($D_3$) | Phrase-Loop Attractor Rate | Mean Generation Entropy |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Greedy ($\text{temp}=0.0$)** | 0.9% | 47.4% | 51.8% | **70.0%** | 5.12 nats |
| **Pure Sampling ($\text{temp}=0.7$)** | 0.4% | 93.0% | 95.0% | **20.0%** | 5.38 nats |
| **Nucleus ($\text{temp}=0.7, p=0.9$)** | 1.7% | 98.3% | 99.5% | **0.0%** | 5.41 nats |
| **Nucleus + Penalty ($\text{temp}=0.7, p=0.9, \text{rep}=1.2$)** | **0.0%** | **99.1%** | **99.5%** | **0.0%** | 5.48 nats |
| **Nucleus + Strong Penalty ($\text{temp}=0.7, p=0.9, \text{rep}=1.5$)** | 0.4% | 100.0% | 100.0% | **0.0%** | 5.56 nats |
| **Greedy + Penalty ($\text{temp}=0.0, \text{rep}=1.3$)** | 1.7% | 95.7% | 97.3% | **0.0%** | 5.12 nats |

---

## 3. Scientific Root Cause Findings

1. **Separation of Training Effect vs Decoding Effect**:
   - **Training Effect**: The model has learned that high-frequency unigrams/bigrams (`the`, `of the`, `most of`) have the lowest cross-entropy loss in natural text. Under cross-entropy $\sim 3.6$ nats, these high-frequency functional words dominate top logits.
   - **Decoding Effect**: In greedy argmax decoding ($T=1$, $\text{temp}=0$), once an attractor bigram is generated (`of the`), its highest-likelihood continuation is often itself (`most of the`), creating a deterministic recurrent cycle.
2. **Attractor Elimination**:
   - Applying nucleus sampling ($p=0.9$) or modest repetition penalties ($\text{rep}=1.2$) **completely abolishes phrase loops (0.0% loop rate)** while lifting Distinct-2 from 47.4% to 99.1%.
   - **Conclusion**: Phrase loops are not a structural failure of the neural weights, but a known property of greedy decoding on un-instruction-tuned base language models.
