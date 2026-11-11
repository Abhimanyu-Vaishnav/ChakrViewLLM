# CR-EXP-15 Failure Analysis
## Qualitative Error Taxonomies and Mechanistic Root Cause

**Mission ID:** CR-EXP-15  
**Focus:** Diagnostic qualitative breakdown of generation failures across all 7 evaluated architectures.

---

## 1. Qualitative Failure Taxonomy

Inspection of 8,400 total probe generations (1,200 probes $\times$ 7 models) reveals three distinct failure regimes:

### Regime 1: Parametric Substitution (Candidate D / Dynamic Updates)
* **Manifestation:** Candidate D produces fluent, grammatical sentences following the exact target template, but substitutes entities and values from training data rather than reading from the prompt context.
* **Examples:**
  - *Prompt:* `Repeat the exact name: Mukesh.` $\to$ *Output:* `Manom is currently in Gwalior.`
  - *Prompt:* `Turn 1: User: Rajnath lives in Rourkela. Turn 2: User: Rajnath moved to Saharanpur. Turn 3: User: Where is Rajnath currently located?` $\to$ *Output:* `Amitabh is currently in Aurangabad.`
  - *Prompt:* `Turn 1: User: Badri lives in Belgaum. Turn 2: Tarun lives in Vellore. Turn 3: What is the current location of Vikas?` (Abstention probe) $\to$ *Output:* `Ankush is currently in Aurangabad.`
* **Mechanistic Cause:** The model minimizes cross-entropy loss by projecting high-frequency entity/location bigrams from its static feed-forward weights. It behaves as an autoregressive *prior generator* rather than a *context-conditioned transducer*.

### Regime 2: Conversational Sycophancy / Assistant Dialogue Attractor (CR-12 Parent Baseline & Candidate A)
* **Manifestation:** Baseline models prepend polite assistant greetings or conversational boilerplate, completely ignoring the probe instruction.
* **Examples:**
  - *Prompt:* `Repeat the exact name: Somesh.` $\to$ *Output:* `Good morning! I am here to assist you with pass?`
  - *Prompt:* `Turn 1: User: ...` $\to$ *Output:* `Hello! How can I assist you with your day?`
* **Mechanistic Cause:** Strong conversational pretraining weights in early layers create massive attention sinks on dialogue tokens (`User:`, `Assistant:`, `Hello`), suppressing context retrieval heads.

### Regime 3: Modal & Repetitive Collapse (Arch Arms B & C)
* **Manifestation:** Models trained with altered architectures from scratch (Arm B: 3 heads, $d_{\text{head}}=64$; Arm C: 256-dim, 4 heads, $d_{\text{head}}=64$) quickly degenerate into repeating fixed token loops.
* **Examples:**
  - *Arch Arm B:* `Assistant: Assistant: Assistant: Assistant: Assistant:`
  - *Arch Arm C:* `AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA`
* **Mechanistic Cause:** Standard random weight initialization with small-scale fine-tuning data (3,000 steps) fails to break symmetry or form functional self-attention circuits. Transformer models require large-scale pretraining across broad corpora to develop stable key-query subspace alignments.

---

## 2. Mechanistic Root Cause: The Absence of In-Context Induction Circuits

Why does even Candidate B (which was specifically trained on 500 pure entity-copying examples) fail at direct copying?

1. **Subword Tokenizer Is Not The Blame:** In Phase 4 (`TOKENIZER_BINDING_DIAGNOSTICS.md`), single-token names (`Dev` = `[342]`, `Kavita` = `[810]`) had 0% copying accuracy. Fragmentation does not explain the failure.
2. **Attention Head Dimension Is Not The Blame:** Doubling head dimension to 64 (Arch Arm B) does not automatically generate copy heads.
3. **Model Width Is Not The Blame:** Increasing width to 256 (Arch Arm C) does not solve copy routing.
4. **The Real Culprit: Lack of Induction Head Pretraining.**
   In transformer mechanistic interpretability (Elhage et al., 2021), in-context copying and binding require a two-layer circuit:
   - **Layer $L$ (Previous-Token Head):** Attends from token $n$ to token $n-1$, writing the identity of $n-1$ into the residual stream.
   - **Layer $L+k$ (Induction Head):** Attends from current token $A$ back to a prior occurrence of $A$, and reads out the token that followed $A$ ($B$) to predict $B$.
   
   In ChakrMicro's small 6-layer architecture, these induction circuits never formed during historical pretraining. Fine-tuning on a few hundred or thousand examples only adjusts final layer logits or biases towards memorized training tokens, but cannot synthesize induction heads from scratch without dedicated pretraining tasks (e.g. prefix matching, repeated string prediction).

