# Exp20 Context Extraction Comparison Report

**Experiment ID**: `exp20_single_task_contextual_binding_cpu_400`  
**Task Family**: Single-Task `context_extraction`  
**Test Set**: 20 frozen held-out records (`EXP18_TEST_CONTEXT_EXTRACTION_0000` to `0019`)  
**Baseline**: Isolated canonical Step 10,000 checkpoint copy  
**Trained Candidate**: Best validation checkpoint (Step 240, $L_{\text{val}} = 1.7447$)  

---

## 1. Quantitative Comparison

| Metric / Attribute | Starting Baseline (Step 10,000) | Exp19 Multi-Task (Step 340) | Exp20 Single-Task (Step 240) | Delta vs Baseline | Exp20 Target | Gate Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Task Accuracy** | **0.0%** (0/20) | **0.0%** (0/20) | **0.0%** (0/20) | **0.0%** | $\ge 20.0\%$ | **UNMET** |
| **EOS Completion Rate** | **0.0%** (0/20) | **55.0%** (11/20) | **100.0%** (20/20) | **+100.0%** | $\ge 80.0\%$ | **SURPASSED** |
| **Syntax Frame Accuracy** | 0.0% (0/20) | 25.0% (5/20) | **95.0%** (19/20) | **+95.0%** | $\ge 80.0\%$ | **SURPASSED** |
| **Average Tokens Generated** | 45.0 tokens | ~33.5 tokens | **10.65 tokens** | **-34.35 tokens** | $\le 25.0$ | **SURPASSED** |
| **Average Latency (CPU)** | 425.20 ms | 315.40 ms | **108.62 ms** | **-316.58 ms** | $\le 250$ ms | **SURPASSED** |
| **Best Validation Loss** | 4.4120 | 2.8786 (multi) | **1.7447** (single) | **-2.6673 nats** | - | **OPTIMIZED** |

---

## 2. In-Depth Error & Behavioral Analysis

### What Improved Radically: Syntax & Termination
1. **100% EOS Termination**: Every single one of the 20 test generations terminated cleanly with EOS. There were zero runaway loops, zero repetition crashes, and zero `MAX_TOKENS` cutoffs.
2. **Inference Acceleration**: Latency dropped to an astonishing **108.62 ms per query on pure CPU**, generating exactly the concise sentence needed (~10.6 tokens).
3. **Syntax Frame Mastery**: 19 out of 20 generations perfectly matched the official syntax structure:
   `"<Name> is assigned as the <Role>."`

### What Failed: Entity Hallucination vs Contextual Binding
When asked:
- **Context**: `Official expedition roster dispatch: Zachary Taylor is assigned as the principal systems engineer for Orbital Expedition TEST-11.`
- **Prompt**: `In Orbital Expedition TEST-11, who is assigned as the principal systems engineer?`
- **Expected**: `Zachary Taylor is assigned as the principal systems engineer.`
- **Model Generated**: `'Hanna Lin is assigned as the principal systems engineer.'`

### The Mechanism of Failure
- The model generated `'Hanna Lin'` (a blend of training names `"Hanna Marin"` and `"Maya Lin"`).
- In another test record with context `Fatima Zahra`, the model generated `'Handry Lindry'` (a blend of training names `"Caleb Landry"` and `"Henrik Lindqvist"`).
- In another test record with context `Victoria Sterling`, the model generated `'Carlo Gemma'` (a blend of training names `"Gabriel Santos"` and `"Gemma Chan"`).

**Conclusion**: The model did **not** attend to the novel entity name in the `Context:` prompt. Instead, it memorized the entity distribution of the 50 training names and sampled plausible name tokens from its internal weights, filling the subject slot before generating the role.

---

## 3. Hypothesis Resolution

- **Primary Hypothesis ($H_1$)**: *Isolating context extraction will reduce cross-task interference and improve exact semantic extraction on unseen held-out examples.*
  - **RESULT**: **REJECTED**. Task isolation eliminated cross-task interference, cut validation loss to 1.74, and yielded 100% EOS termination, but exact task accuracy on novel held-out entities remained **0/20 (0.0%)**.
- **Null Hypothesis ($H_0$)**: *Single-task SFT will not produce a meaningful improvement over the starting checkpoint on the frozen context-extraction evaluation set.*
  - **RESULT**: **SUPPORTED**.

### Root-Cause Finding
ChakrMicro (3.44M parameters, 6 layers, 384 embedding dim, 6 attention heads) lacks dedicated **induction heads** / copy-paste circuits capable of dynamic cross-token binding from the prompt context into the generation stream. It functions as an associative syntax generator rather than an attention-driven factual extractor.
