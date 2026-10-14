# Initial Repository, Architecture & Capability-Provenance Audit (Batch 261)
**Location:** `experiments/batch261_270_open_ended_generalization/reports/b261_270_initial_audit.md`  
**Milestone:** Batches 261–270 — Open-Ended Question Generalization, Randomized Evaluation & Neural Capability Auditing  
**Date:** 2026-10-14  

---

## 1. Repository State & Checkpoint Invariants

| Invariant Item | Expected Specification | Actual Verification | Integrity Verdict |
|---|---|---|---|
| **Git Branch** | `dev/v0.6-v0.7-validation` | `dev/v0.6-v0.7-validation` | **VERIFIED** |
| **Previous Commit** | `7cd91e8` | `7cd91e8fdab246c828a198f5442db5c924c391e9` | **VERIFIED** |
| **Parent Commit** | `9cc3795` | `9cc3795d50429f37df91cff205081726a330cf93` | **VERIFIED** |
| **Protected File** | `chakrview/runtime/session.py` | Local modifications preserved untouched | **PROTECTED** |
| **Canonical Checkpoint** | `experiments/batch16_21_combined/.../training_checkpoint_step_10000.pt` | SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` | **IMMUTABLE** |
| **Parameter Count** | 3,443,136 parameters | 3,443,136 ($\Delta W = 0$) | **VERIFIED** |
| **Interactive Suite** | 30 test cases in real founder console | 30/30 passed (100.0%) | **REPRODUCED** |
| **3-Hop Baseline** | `frozen_threehop_manifest_100.json` | SHA-256: `6bf47a2baf12...` (51% EM, 0 confusions) | **REPRODUCED** |

---

## 2. Capability Provenance Audit: What Is Neural vs Assisted?

In accordance with user instructions, ChakrView's intelligence must never be misrepresented. The 30/30 result achieved in Batch 260 was produced through the following distinct component mechanisms:

```
                  ┌───────────────────────────────────────────────┐
                  │                 User Query                    │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │          CognitiveRouter            │
                       │  (Rule-based Regex & Precedence)    │
                       └──────────────┬──────────────────────┘
                                      │
         ┌───────────────┬────────────┴───┬──────────────────┬─────────────────┐
         │               │                │                  │                 │
         ▼               ▼                ▼                  ▼                 ▼
   CALCULATOR_TOOL  CODE_GEN      CONCEPT_DEFINITION    USER_MEMORY       NEURAL CORE /
   (ast.eval math) (Template)     (Dictionary Lookup) (Discourse Stack)    POINTER HEAD
         │               │                │                  │                 │
         ▼               ▼                ▼                  ▼                 ▼
    Deterministic  Deterministic    Deterministic      Deterministic     Neural-Trained
      Execution       Output           Lookup             Memory          Cross-Attn
```

### Route-by-Route Breakdown:

1. **`CALCULATOR_TOOL` (e.g., "What is 25 + 37?"):**
   - **Mechanism:** Deterministic regex parsing of mathematical operators into a restricted AST evaluator (`CalculatorTool`).
   - **Provenance:** **100% Deterministic Tool-Assisted**. No neural generation occurs.
   - **Limitation:** Word problems ("If I have 25 apples and buy 37 more...") fail routing unless explicitly pre-extracted.

2. **`CODE_GENERATION` (e.g., "Write a Python program to add two numbers..."):**
   - **Mechanism:** Procedural code template substitution with syntactic AST parsing.
   - **Provenance:** **100% Deterministic Code Synthesizer**.
   - **Limitation:** Does not generalize to arbitrary programming specifications outside known templated tasks.

3. **`CONCEPT_DEFINITION` (e.g., "What is an ALU?", "Explain CPU", "Value of pi"):**
   - **Mechanism:** Dictionary lookup inside `GroundedSessionAdapter.execute_turn()` mapping extracted concept keys (`alu`, `cpu`, `pi`, `ram`, `e_constant`, `photosynthesis_hindi`) to curated responses.
   - **Provenance:** **100% System Knowledge Lookup**.
   - **Limitation:** Unfamiliar concepts (e.g., "What is an FPGA?", "Explain GPU tensor cores") fail lookup and fall back to honest abstention.

4. **`USER_MEMORY` (e.g., "My name is Ramesh", "What is my name?"):**
   - **Mechanism:** Regex slot extraction (`has_fact`, `set_fact`) into `UserProfileMemory` and bounded `DiscourseEntity` stack.
   - **Provenance:** **100% Structured Memory Management**.
   - **Limitation:** Lexical extraction only; indirect self-identification requires conversational parsing.

5. **`THREE_HOP_GROUNDING` (Cross-Attention Pointer Head):**
   - **Mechanism:** Trained PyTorch module (`ChakrCrossAttentionPointerHead`) using scaled dot-product cross-attention between context tokens and query tokens, combined with `TokenTrie` constrained decoding.
   - **Provenance:** **Neural Learned Representation + TokenTrie Decoding**.
   - **Demonstrated Bound:** 51.0% exact match on held-out 3-hop entity clusters. Spurious distractor confusions successfully eliminated via intermediate supervision loss.

---

## 3. Mission Strategy for Batches 261–270

To avoid building a brittle "childlike if-else program", Batches 261–270 will:
1. Build a randomized open-ended question generator covering 12 varied domains with sealed, un-tuned held-out test splits.
2. Formally audit brittleness across paraphrases, misspellings, unseen entity combinations, and counterfactuals.
3. Decouple neural-only evaluation from tool-assisted execution.
4. Implement root-cause semantic representation repairs in classification, evidence extraction, and answer verification.
5. Prototype permission-gated internet research with strict user authorization and provenance tracking.
