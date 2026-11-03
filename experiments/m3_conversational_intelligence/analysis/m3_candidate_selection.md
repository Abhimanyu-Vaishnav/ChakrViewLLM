# M3 Checkpoint Candidate Selection & v0.9 Readiness Evaluation

**Milestone:** M3 — Conversational Intelligence Development  
**Target Gate:** November 2026 v0.9 Conversational Intelligence Preview  
**Evaluated Candidates:** M3 Step 50, Step 100, Step 250, Step 500  

---

## 1. Candidate Comparison & Scoring Matrix

| Evaluation Dimension | Weight | Step 50 | Step 100 | Step 250 (Selected) | Step 500 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Repetition Suppression (4-gram rep ratio)** | 25% | 0.1253 (Fair) | 0.1241 (Fair) | **0.0223 (Optimal)** | 0.0276 (Optimal) |
| **Unique Token Diversity** | 20% | 0.7660 | 0.7939 | **0.9036 (Peak)** | 0.8781 |
| **Natural EOS Termination Rate** | 20% | 63.3% (19/30) | 66.7% (20/30) | **93.3% (28/30)** | 83.3% (25/30) |
| **Attractor Loop Suppression (32 tasks)** | 15% | 31.2% (10/32) | 25.0% (8/32) | **15.6% (5/32)** | **15.6% (5/32)** |
| **Intelligence Regression (32 tasks)** | 10% | 2/32 (6.2%) | 2/32 (6.2%) | 2/32 (6.2%) | 3/32 (9.4%) |
| **Coding Regression (Syntax Validity)** | 10% | 3/12 (25.0%) | 4/12 (33.3%) | **4/12 (33.3%)** | **4/12 (33.3%)** |
| **Overfitting Risk ($L_{\text{val}}$ vs Generalization)**| Gate | Low ($L=4.11$) | Moderate ($L=3.17$) | **Optimal ($L=2.60$)** | High ($L=1.95$, sub-word drift) |
| **Overall Scientific Score (0 - 100)** | 100% | **61.5** | **68.2** | **88.5** | **81.0** |

---

## 2. Best Checkpoint Selection

### Selected Milestone Candidate: **M3 Checkpoint Step 250**
- **File:** `experiments/m3_conversational_intelligence/runs/m3_sft_500steps/sft_checkpoint_step_250.pt`
- **File Size:** 41,421,538 bytes
- **SHA-256:** `91b0bae2a8c737c8db4079c8ac3645f8ff8436b12501f687660f4f9ab7b83433`

### Why Step 250 Was Chosen Over Step 500:
1. **Lowest Repetition Rate Across All Runs:** Step 250 achieved an all-time low 4-gram repetition ratio of **0.0223** (compared to 0.0276 at Step 500, 0.1862 at M2 Arm A, and 0.2540 on Canonical Step 10k).
2. **Highest Vocabulary Diversity:** Step 250 achieved a unique token ratio of **0.9036** (surpassing Step 500's 0.8781).
3. **Peak EOS Stopping Reliability:** 28 out of 30 test prompts (93.3%) terminated cleanly with genuine EOS markers, avoiding truncation or runaway loops.
4. **Preservation of General Knowledge:** At Step 500, over-optimization toward short memorized completions caused sub-word boundary fragmentation on unknown words, whereas Step 250 maintains balanced syntactic fluency.

---

## 3. Honest v0.9 Readiness Gate Assessment

### Assessment Rule:
> Maintain scientific truth. Do not declare a checkpoint "conversational" or "production-ready" when the data shows failure.

### Evaluation of 8 Required Capabilities:
1. **Natural Dialogue & Greetings:** **PARTIAL** (Generates natural declarative sentences, but greeting intent is not consistently matched).
2. **Direct Instruction Following:** **FAIL** (Cannot reliably obey negative constraints or exact word demands).
3. **Context Retention Across Turns:** **FAIL** (Multi-turn entity tracking fails to extract prior user tokens).
4. **Basic Reasoning & Arithmetic:** **FAIL** (Arithmetic expressions fail to resolve to correct numerical answers).
5. **Information Extraction / QA:** **FAIL** (Grounded QA does not reliably extract target entities from context).
6. **Format Adherence:** **FAIL** (Cannot strictly enforce JSON, list delimiters, or exact word counts).
7. **Appropriate EOS Termination:** **PASS** (Achieved **93.3%** clean `<EOS>` termination rate).
8. **Repetition / Attractor Suppression:** **PASS** (Suppressed attractor collapse from **53.1%** down to **15.6%**; 4-gram repetition reduced by 91%).

---

## 4. Formal Gate Decision

**Status:** **NOT READY FOR GENERAL CONVERSATIONAL v0.9 RELEASE (CONDITIONAL EXPERIMENTAL CANDIDATE ONLY)**

### Recommendation:
- Promote **M3 Step 250** as the **Internal Research Baseline for Attractor Suppression & EOS Control**.
- **Do NOT release as a user-facing conversational assistant.**
- For true conversational intelligence, the architecture requires scaling from 3.44M parameters ($d=192$) to $\ge 15\text{M}$ parameters ($d=384$ or $512$) and larger context pretraining before instruction tuning.

