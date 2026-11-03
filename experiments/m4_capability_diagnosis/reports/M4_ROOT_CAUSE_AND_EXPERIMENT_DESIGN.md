# CHAKRVIEW — MILESTONE M4 ROOT-CAUSE ANALYSIS & EXPERIMENT DESIGN
## Ranked Evidence-Based Diagnosis and Protocol for M5 Controlled Grounding Experiment

**Project:** ChakrView  
**Milestone:** M4 — Capability Diagnosis & Learning Strategy Research  
**Date:** 2026-10-30  

---

## 1. RANKED ROOT-CAUSE TABLE

Based on rigorous cross-examination of model code, training mechanics, dataset structures, and evaluation traces, the top five causes explaining ChakrView's lack of conversational intelligence are ranked below:

| Rank | Root Cause Hypothesis | Supporting Evidence | Contradicting Evidence | Confidence | Impact |
|---|---|---|---|---|---|
| **1** | **Dataset Semantic Collapse (Low Response Diversity)** | 11/15 categories share only 5 unique responses repeated 8x across prompt variants. 125 total responses in 600 records (20.8% diversity). Model memorized static answer tokens instead of dynamic prompt-to-response mapping. | Arithmetic category has 40 distinct responses, but still produced invalid calculations in evaluation. | **CONFIRMED** | **CRITICAL** |
| **2** | **Lack of Variable-Binding & Dynamic Extraction Training Data** | Multi-turn training examples feature 5 identical facts (`"Your favorite fruit is mango."`, `"You live in Bangalore."`). No examples exist where an arbitrary name or variable in the prompt is mapped to an answer. | Context length (512) and RoPE rotary embeddings are structurally capable of sequence attention. | **STRONGLY INDICATED** | **HIGH** |
| **3** | **Evaluator False Positives Masked True Intelligence Baseline** | `CODE-03` passed via AST syntax check on fragment `dp.`. `M3-EVAL-GQA-04` passed via substring `"no"` in `"non-positive"`. True conversational intelligence across all checkpoints is 0.0%. | Neural harness correctly captured physical generation stability (EOS, repetition). | **CONFIRMED** | **HIGH** |
| **4** | **Pretraining Representation Breadth (Pretrained Base Checkpoint Ceiling)** | Pretrained base checkpoint (Canonical Step 10,000) was trained on a small synthetic corpus (10k steps) with limited factual world knowledge. SFT cannot retrieve facts the base model never learned. | The model successfully learned syntax tokens, EOS token distribution, and prompt role labels from SFT alone. | **STRONGLY INDICATED** | **MEDIUM-HIGH** |
| **5** | **Superficial Prompt Masking without Bidirectional Conditioning Scaffolding** | Causal mask with response-only loss allows the model to predict high-frequency response tokens without strong cross-entropy pressure on intermediate reasoning tokens. | Standard causal SFT with prompt masking is the industry-standard training formulation. | **PLAUSIBLE** | **MEDIUM** |

### Hypotheses Ruled Out / Unsupported:
- *Hypothesis: "3.44M parameter model cannot fit dialogue format."* **REFUTED.** The model learned turn taking, role tokens (`User:`, `Assistant:`), and `<EOS>` termination flawlessly.
- *Hypothesis: "Attention context window is too short."* **REFUTED.** All prompts were $<120$ tokens, well within the 512-token RoPE window.

---

## 2. COMPARISON OF CANDIDATE DIRECTIONS FOR M5

| Direction | Description | Feasibility | Expected Value | Selection |
|---|---|---|---|---|
| **Direction A: Dynamic Prompt-Conditioned Extraction & Variable Binding** | Replace collapsed 125-response mixture with 1,000 human-verified, high-diversity examples focusing on dynamic span extraction, simple slot filling, and direct answer conditioning. | High (CPU SFT, < 150s) | Directly addresses Root Causes #1 and #2. Tests whether 3.44M can learn dynamic input-to-output mapping. | **SELECTED (PRIMARY)** |
| **Direction B: Expanded Multi-Task Pretraining / CPT** | Pretrain for 20,000 additional steps on expanded multilingual/code corpora before SFT. | Low (High CPU training time, multi-day run) | Addresses Root Cause #4, but premature before verifying SFT data efficiency. | Deferred to M6 |
| **Direction C: Architectural Expansion (Increase layers/width)** | Expand from 3.44M to 15M parameters. | Medium | Prohibited by project constraints; unscientific to expand size before fixing data diversity. | Rejected |

---

## 3. RECOMMENDED EXPERIMENT: EXP35 — DYNAMIC PROMPT-CONDITIONED EXTRACTION (DPCE)

### 3.1 Research Question
Can ChakrMicro (3.44M parameters) learn dynamic, prompt-dependent entity extraction and question answering when trained on a dataset with 100% unique response strings and explicit variable-binding supervision?

### 3.2 Hypothesis
If the fine-tuning mixture eliminates duplicate responses and explicitly pairs diverse contexts with dynamic target spans (e.g. `Context: Alice has 3 apples. Question: Who has apples? Answer: Alice`), the model will begin generating prompt-relevant entities rather than falling back to static memorized planet/Linux tokens.

### 3.3 Experimental Specification
- **Baseline Checkpoint:** M3 Step 250 (`91b0bae2...`) or M2 Arm A Step 50 (`150f6214...`).
- **Dataset Requirements:**
  - 1,000 curated records (900 train / 100 val).
  - **100% Unique Responses** (1,000 distinct completions).
  - Explicit task types:
    1. Direct Copy / Extractive QA (Context contains the exact answer token).
    2. Slot Filling & Variable Binding (e.g. `X = 5, what is X?`).
    3. Closed-world entity disambiguation.
    4. Clean conversational greetings and polite termination.
- **Training Configuration:**
  - Steps: 500
  - Batch Size: 4 (with gradient accumulation = 2, effective batch = 8)
  - Learning Rate: $3.0 \times 10^{-4}$ with cosine decay to $3.0 \times 10^{-5}$
  - Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay = 0.01)
  - Prompt Masking: Strict response-only loss (`mask_prompt=True`, `ignore_index=-100`)
- **Evaluation Plan:**
  - Run repaired 60-task Conversational Benchmark (with word-boundary scoring).
  - Run 30-prompt Neural Harness for repetition, unique tokens, and EOS.
- **Measurable Success Criteria:**
  1. Conversational Benchmark Pass Rate $> 10.0\%$ (at least 6/60 tasks genuinely passed with word-boundary match).
  2. Extractive Grounded QA Pass Rate $> 25.0\%$.
  3. Harness Repetition Ratio $< 0.05$.
  4. Genuine EOS Termination Rate $> 85.0\%$.
- **Failure Conditions:**
  - Conversational benchmark remains $\le 3.3\%$ (2/60), or responses continue generating unconditioned static tokens (`largest planet`, `Linux`).

---

## 4. ESTIMATED RESOURCE UTILIZATION (CPU-FIRST)
- Training hardware: Standard x86_64 CPU (AVX2/AVX-512)
- Training duration: $\approx 120 - 180$ seconds for 500 steps.
- Peak RAM: $< 800\text{ MB}$.
- Storage footprint: $< 150\text{ MB}$ total for checkpoints and logs.

