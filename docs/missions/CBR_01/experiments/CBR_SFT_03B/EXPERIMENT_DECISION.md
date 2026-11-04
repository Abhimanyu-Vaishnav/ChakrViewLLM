# CBR-SFT-03B — Experiment Decision & Next Step Recommendations

## 1. Executive Decision

**Decision:** **SELECT CBR-SFT-03B STEP 200 (`297dcc60...`) AS THE NEW EXPERIMENTAL CONVERSATIONAL PILOT CHECKPOINT.**

Canonical Baseline Checkpoint (`96357705...`) remains permanently preserved and unmodified.

---

## 2. Decision Rationale

1. **Empirical Breakthrough Achieved:**
   - First non-zero conversational score on held-out prompts in ChakrView history (**27.1%** vs **0.0%** baseline).
   - First non-zero conversational score on the canonical M3 benchmark (**5.0%** vs **0.0%** baseline).
   - Repetitive pretraining attractor collapse extinguished from **53.1%** down to **3.1%**.
2. **Capability Non-Regression:**
   - Native intelligence preserved (1/32).
   - Coding AST syntax validity increased from 25.0% to 33.3% (Step 200) and 41.7% (Step 300).
3. **Step 200 vs Step 300:**
   - Step 200 achieved the best balance of short QA generalization (20.0%) and identity recognition (35.0%) without token repetition.
   - Step 300 is preserved as an alternative high-syntax candidate (41.7% coding syntax validity).

---

## 3. Recommended Next Steps for CBR-01 Mission

1. **Do NOT release CBR-SFT-03B as a public general conversational model.**
   - While 27.1% on held-out greetings is a massive internal breakthrough, general question answering, multi-turn dialogue, and arithmetic are not yet capable for production use.
2. **Proceed to CBR-SFT-04 (Track B Expansion):**
   - Design a balanced 150–200 example mixture containing:
     - 40 temporally balanced greetings (10 morning, 10 afternoon, 10 evening, 10 general).
     - 30 identity & role definitions.
     - 50 short entity-copying and dynamic grounding examples (Track B).
     - 40 basic instructional multi-turn pairs.
   - Maintain the established optimization hyperparameters ($LR=2 \times 10^{-4}$ with cosine decay).
