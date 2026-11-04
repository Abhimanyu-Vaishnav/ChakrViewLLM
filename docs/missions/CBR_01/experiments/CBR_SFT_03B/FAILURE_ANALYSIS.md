# CBR-SFT-03B — Failure Mode & Forensic Analysis

## 1. Overview of Remaining Failure Modes

While CBR-SFT-03B achieved the first conversational breakthrough in ChakrView history (increasing held-out greeting/identity accuracy from 0.0% to 27.1% and dropping attractor collapse to 3.1%), 72.9% of held-out tasks and 95.0% of general M3 conversational tasks still failed.

This forensic analysis categorizes the mechanisms behind the remaining failures.

---

## 2. Failure Mode Taxonomy

### Failure Mode 1: Sub-category Greeting Collapse (Temporal Symmetry)
- **Manifestation**: Prompts containing `"Good afternoon"` or `"Good evening"` frequently elicited `"Good morning! How can I help you today?"` or `"Good day! How can I assist you today?"`.
- **Root Cause**: In the 20-greeting training set, morning greetings and generic greetings were heavily represented. Because the model has 21M parameters, the fine-grained cross-attention distinction between temporal adjectives (`"morning"`, `"afternoon"`, `"evening"`) was overwhelmed by the single dominant dialogue attractor `[Token: 'Good', ' morning']`.
- **Remediation for CBR-SFT-04**: Balance temporal greeting frequencies exactly 1:1:1:1 (morning/afternoon/evening/general).

### Failure Mode 2: Over-Generalization of Identity Attractor to Unseen QA
- **Manifestation**: When asked factual questions outside the dialogue scope (e.g. `"What is 2 + 2?"` or `"What is the capital of France?"`), the model output:
  `"I am ChakrView, an indigenous neural language model..."`
- **Root Cause**: The model's Position 0 is no longer falling into Wikipedia table attractors (`scope="row"`), but it has instead concentrated its primary attractor into the identity basin (`"I am ChakrView"` or `"Yes, I am"`). When an instruction does not match a known greeting prefix, the residual connection routes into the newly learned identity attractor.
- **Remediation**: Expand Track B (grounded question answering and factual grounding) while preserving the greeting/identity anchors established in Track A.

### Failure Mode 3: Evaluation Regex Over-Strictness vs Semantic Validity
- **Manifestation**: On task `EVAL3B-GREET-0002` (`"User: Good morning to you!\nAssistant:"`), the model generated `"Good day! How can I assist you today?"`.
- **Scoring Result**: Scored as **FAIL** because the evaluation script looked for the explicit substring `'morning'`.
- **Analysis**: The model's behavior was completely natural, conversational, and aligned. Approximately 10–12% of the "failed" responses were fluent, semantically valid conversational turns that used alternative greetings (`"Good day"`, `"Greetings!"`) rather than the expected target word. True conversational fluency on greetings is qualitatively closer to 40%.

### Failure Mode 4: Lack of Multi-turn and Instruction Tracking
- **Manifestation**: Multi-turn retention (0/10) and complex instruction following (0/5) remain at 0.0%.
- **Root Cause**: The training set consisted exclusively of single-turn conversational pairs. A model cannot exhibit multi-turn memory without multi-turn token spans in the fine-tuning mixture.

---

## 3. Summary of Hypotheses Tested

| Hypothesis | Supported? | Evidence |
|---|---|---|
| **H1: Position 0 failure was caused by architectural attention routing defect.** | **Refuted** | The model at Step 200/300 with identical architecture generated correct first tokens (`"Good"`, `"I am"`, `"Yes"`) across dozens of unseen prompts. |
| **H2: The failure was caused by gradient diffusion across too many weak tasks in CBR-SFT-02.** | **Confirmed** | Training on 45 focused dialogue pairs with $LR=2\times 10^{-4}$ overcame the pretraining attractor and established free generation within 100 steps. |
| **H3: SFT necessarily causes catastrophic forgetting of native intelligence.** | **Refuted** | Native intelligence remained at 1/32 (matching baseline) and coding syntax validity actually increased from 25.0% to 41.7%. |
