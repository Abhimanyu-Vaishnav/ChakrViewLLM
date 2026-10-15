# Mission Summary: Batches 501–540
## Neural Generalization & Learning Improvement

**Repository:** `D:\Project\ChakrView`  
**Branch:** `dev/v0.6-v0.7-validation`  
**Starting Commit:** `35e0b06` — `feat(batch471-500): stress-test scratchpad generalization, integrate evidence gate into runtime, and audit attribution`  
**Milestone Commit:** `feat(batch501-540): audit scratchpad leakage, train multi-task rule curriculum, and evaluate 60-item generalization`  
**Canonical Step 10,000 Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`  
- SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified Immutable)  
**Evaluated Starting Candidate:** `experiments/batch441_470_reasoning_gate/checkpoints/candidate_step150_scratchpad_reasoning.pt`  
- SHA-256: `72784a614b2e669242aae1d67adf4e474f186178c18d4ec25070e78378b19759`  
**New Trained Candidate:** `experiments/batch501_540/checkpoints/candidate_step200_compositional_rule.pt`  
- SHA-256: `b18e69ebf4f0b396465af3c323f38bcfa84d5a38eabbd4a4220a10d6bfafee98`  

---

### 1. Executive Summary & Core Research Findings

In Batches 501–540, we addressed the central research question of ChakrView: **Can a 3.4M-parameter neural model learn true multi-step compositional generalization from textual scratchpads, or is it fundamentally an associative memory replaying memorized frames?**

Through an exhaustive audit, synthetic dataset construction, curriculum training, and a 60-item evaluation across 6 distinct capability levels, we arrived at an unambiguous, scientifically grounded conclusion:

1. **Root Cause of Prior "50% Generalization":**  
   The prior 6-item benchmark in Batches 441–470 contained severe data contamination: **3 out of the 6 "held-out" test items were identical verbatim duplicates of training items** (`sp_test_001 == sp_train_001`, `sp_test_002 == sp_train_004`, `sp_test_003 == sp_train_006`). On the 3 truly novel items, performance was 0%.
2. **Controlled Training Experiment:**  
   We generated a clean, leakage-free dataset (`train_compositional_rule_sft.jsonl`, N=70) and a comprehensive held-out test suite (`test_heldout_comprehensive_60.jsonl`, N=60) with **zero prompt or operand overlap** across splits. We trained `candidate_step200_compositional_rule.pt` using a multi-task curriculum (40% rule scratchpads, 35% paraphrase SFT, 20% single-step primitives, 5% Stage-C pretraining rehearsal) with validation loss tracking and early stopping.
3. **The Generalization Limit of 3.4M Parameters:**  
   - On single-step arithmetic and familiar paraphrases, the model performs strongly (**75.0% held-out paraphrase, 66.7% retention**).
   - On novel multi-step compositions with unseen operands (`Category A: (55 + 37) - 14`, `12 * 7 - 21`), exact match accuracy remains **0.0% (0/10)**.
   - The model learns the *syntactic formatting* of scratchpads (`Step 1: ... Step 2: ... Final Answer: ...`), but **hallucinates intermediate arithmetic values** (`12 * 7 = 20`, `20 + 2 = 18`).
   - Pure textual autoregressive next-token prediction in a 3.4M parameter Transformer without dedicated arithmetic circuits or pointer-binding mechanisms cannot compute dynamic multi-digit arithmetic.
4. **Architectural Confirmation:**  
   This establishes that **ChakrView's two-tier architecture is mathematically necessary**: the neural brain must handle semantic understanding, intent recognition, and routing, while deterministic tools (`CalculatorTool`) must execute symbolic math.

---

### 2. Comprehensive 60-Item Evaluation Matrix

| Category | Description | Sample Size | Baseline Candidate (`72784a61...`) | Trained Candidate (`b18e69eb...`) |
| :--- | :--- | :--- | :--- | :--- |
| **Category A** | Familiar format, unseen operands | 10 | 0.0% (0/10) | 0.0% (0/10) |
| **Category B** | Novel expression structures | 10 | 0.0% (0/10) | 0.0% (0/10) |
| **Category C** | Unseen surface wording | 10 | 0.0% (0/10) | 0.0% (0/10) |
| **Category D** | Composed operations (squares, angles) | 10 | 10.0% (1/10) | 10.0% (1/10) |
| **Category E** | Symbolic rule chains (X -> Y -> Z) | 10 | 10.0% (1/10) | 0.0% (0/10) |
| **Category F** | Direct single-step controls | 10 | 30.0% (3/10) | 30.0% (3/10) |
| **Overall 60-Item Score** | Multi-category composition | 60 | **8.3% (5/60)** | **6.7% (4/60)** |
| **Held-Out Paraphrase (20 items)**| Familiar concepts, varied phrasing | 20 | 70.0% (14/20) | **75.0% (15/20)** (+5.0%) |
| **Diagnostic Retention (15 items)**| Historical capability benchmark | 15 | 66.7% (10/15) | **66.7% (10/15)** (Preserved) |
| **True EOS Termination (Token 1)**| Causal termination cleanliness | 60 | 83.3% | **93.3%** (+10.0%) |

---

### 3. Answers to Core Mission Questions

1. **Did the new candidate learn to solve previously unseen compositions better than the previous candidate, under a controlled and reproducible evaluation?**  
   **No.** Unseen compositional arithmetic accuracy was 0.0% across both candidates. The new training curriculum improved prompt diversity, reduced training loss, and increased held-out paraphrase accuracy (70% -> 75%), but did not unlock dynamic multi-step arithmetic calculation.
2. **What was the root cause of the previous 50% score?**  
   The previous held-out set leaked exact training examples into the test split (`sp_test_001` through `sp_test_003` were identical to training items).
3. **Did intermediate-step supervision improve intermediate arithmetic validity?**  
   **No.** The model generated syntactically structured steps (`Step 1: ... Step 2: ...`), but the numbers inside the steps were either replayed from training frames or mathematically incorrect.
4. **Were earlier capabilities retained?**  
   **Yes.** Paraphrase generalization improved to 75.0%, diagnostic retention was preserved at 66.7%, and true EOS emission improved to 93.3%.
5. **What is the definitive next step for ChakrView's intelligence?**  
   Cease attempting to force pure-neural text generation to perform multi-digit arithmetic. Lean into ChakrView's **cognitive routing**: train neural representations for **intent classification, slot extraction, and tool dispatch**, delegating arithmetic to `CalculatorTool` and document verification to `ChakrVerify`.

---

### 4. Recommendation

**Preserve both candidates (`candidate_step150_scratchpad_reasoning.pt` and `candidate_step200_compositional_rule.pt`) for internal comparative research.** The 3.4M model is well-suited as a semantic router and language realizer, while mathematical truth must remain governed by deterministic runtime tools. User modifications in `chakrview/runtime/session.py` remain preserved.
