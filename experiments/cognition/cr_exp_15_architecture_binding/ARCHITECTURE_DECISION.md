# CR-EXP-15 Architecture Decision
## Definitive Architectural and Methodological Directives for ChakrView

**Mission ID:** CR-EXP-15  
**Audience:** Neural Research Scientist, Core Architecture Maintainer, v0.9.1 Release Team  

---

## 1. Core Architectural Conclusions

Based on the empirical findings of CR-EXP-15 across 7 distinct models and 8,400 evaluated probe outputs:

### 1. Rejection of the Subword Tokenizer Hypothesis
Subword fragmentation is **not** the barrier preventing entity tracking. Single-token entities (names like `Dev`) fail copying at identical 0% rates as 4-token entities (`Digvijay`). Changing tokenizers or adding byte-level entity prefixes without changing induction circuits will yield zero benefit.

### 2. Rejection of Naive Attention Head Scaling ($d_{\text{head}}=64$)
Doubling head dimension while halving heads ($6 \times 32 \to 3 \times 64$) without induction-head pretraining leads to optimization instability and mode collapse (`Assistant: Assistant:`). The current 6-head configuration with $d_{\text{head}}=32$ is sufficient in representation rank, but lacks learned key-query routing circuits.

### 3. Rejection of Naive Parameter Expansion ($3.44\text{M} \to 5.77\text{M}$)
Increasing hidden dimension to 256 ($5.77\text{M}$ params) without foundational pretraining results in repetition loops (`AAAAAAAA...`). Scaling capacity at the fine-tuning stage does not create new reasoning primitives.

---

## 2. Binding Capability Bottleneck Identified: Induction Circuit Deficit

The fundamental structural flaw of ChakrMicro is **the absence of in-context induction heads**. 

When an autoregressive transformer generates the answer to `Where is Rajnath currently located?`, it must:
1. Match the token sequence `Rajnath` in the prompt.
2. Attend to the location token following `lives in` or `moved to` for that entity.
3. Copy that location token into the output.

Currently, ChakrMicro's attention heads have only learned to attend to dialogue markers (`Assistant:`) and syntactic keywords (`is`, `in`). The feed-forward network fills in the entity slot using parametric weights memorized during training (`Amitabh`, `Gwalior`).

---

## 3. Concrete Directives for Future Training & CR-EXP-16

1. **Retain Canonical Architecture:**
   - Keep $d_{\text{model}} = 192$, $N_{\text{layers}} = 6$, $N_{\text{heads}} = 6$, $d_{\text{head}} = 32$. This architecture has proven stable and computationally viable on CPU/low-resource hardware.

2. **Implement Synthetic Induction-Head Pretraining (Phase 0):**
   - Before any conversational fine-tuning, the base model must undergo a dedicated pretraining stage on synthetic pointer routing tasks:
     - Repeated random token sequence replication ($A B C D \dots \to A B C D \dots$).
     - Variable assignment and lookup ($x = 42; y = 19; \text{return } x \to 42$).
     - In-context dictionary retrieval ($K_1: V_1, K_2: V_2 \implies \text{Query } K_1 \to V_1$).
   - This explicitly forces the attention heads in Layers 2–4 to form induction circuits before dialogue fine-tuning.

3. **Separate Abstention Calibration From Tracking Benchmarks:**
   - Evaluator code must enforce strict isolation: never bundle abstention true-negatives into state-tracking accuracy.

