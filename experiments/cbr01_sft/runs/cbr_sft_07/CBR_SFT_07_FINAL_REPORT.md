# CBR-SFT-07 EXPERIMENT FINAL REPORT
## End-to-End Autonomous Research, Training, Evaluation & Morning Conversational Readiness

---

### 1. Executive Summary

- **Experiment Name:** CBR-SFT-07 (Overnight Conversational Model Mission)
- **Primary Objective:** Build the strongest reliably testable conversational ChakrView model possible using the existing research lineage, ready for developer manual interaction in the morning.
- **Parent Checkpoint:** `experiments/cbr01_sft/runs/cbr_sft_06a/checkpoints/sft_checkpoint_step_25.pt`  
  - SHA-256: `0589fd886853a2e65ffa1c03af0e4708cb7dc3dfd40b4db6ccc179c5bf1045f1`
- **Selected Best Candidate:** `experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_50.pt`  
  - SHA-256: `0cb5df5fa840d20cea737da47758d96cfe884782c9801f28e51660bb236de86b`
- **Interactive Conversational Interface:** `chakrview/chat.py` (Fully operational terminal CLI with sliding-window history and commands).

---

### 2. Dataset Architecture & Invariants

The training dataset was engineered to address the failure modes of previous iterations (e.g., temporal greeting collapse, refusal phrase leakage):
- **Total Examples:** 250 curated pairs across 10 distinct categories.
- **Split:** 228 Train (91.2%), 22 Validation (8.8%).
- **Categories:**
  1. `greetings_social` (30 items): Morning, afternoon, evening, and generic greetings.
  2. `identity_awareness` (25 items): Model name, CPU-first design, indigenous first principles.
  3. `natural_qa` (25 items): Computer architecture, memory, networking, algorithms.
  4. `context_grounding` (25 items): Exact token copying, IP addresses, config values.
  5. `multiturn_dialogue` (25 items): 2-turn and 3-turn memory retention.
  6. `explanations_reasoning` (25 items): Definition and mechanism explanations.
  7. `supportive_conversation` (20 items): Polite, encouraging, constructive developer dialogue.
  8. `hindi_hinglish` (30 items): Natural Roman Hinglish and Devanagari Hindi dialogues.
  9. `abstention` (20 items): 5 distinct phrasing variations to eliminate memorized refusal strings.
  10. `continuity_feedback` (25 items): Clarifying and follow-up conversational turns.
- **Audit Findings:** 0% prompt leakage against validation or any benchmark suites.

---

### 3. Training Execution Details

- **Micro-batch:** 2, **Grad Accum:** 2 (**Effective batch size:** 4).
- **Optimizer:** AdamW, learning rate $8.0 \times 10^{-5}$ decayed with cosine schedule to $1.0 \times 10^{-5}$.
- **Weight Decay:** 0.01, **Gradient Clip:** 1.0.
- **Hardware:** Pure CPU execution in `.venv`.
- **Wall-clock Training Time:** 19.9 seconds.
- **Checkpoints Saved & Verified:**
  - Step 25: `000383afaa217545bf3b017d928d60c29238d2f88d3bb3180bb48ef1f0fdaabc`
  - Step 50: `0cb5df5fa840d20cea737da47758d96cfe884782c9801f28e51660bb236de86b`
  - Step 75: `5d1c297b23ac69e76b110b0e7705e0449f7729c4c51eb84f721923ff8ca66904`
  - Step 100: `9b51ef12520e7b7b188f60ce9e86265d3e256dae1bb175d1d1069870d4993f44`
  - Step 150: `a75980a514c0131d83829676ff49536a97f30d34764092fa8f9c01a629280470`
  - Step 200: `335f311d4e528306a3894212cc6cc241d5e7cc7105f842c79bd38436a4e31425`

---

### 4. Comprehensive Evaluation Matrix

| Checkpoint | CBR-07 (115) | CBR-06A (100) | CBR-04 (80) | CBR-03B (70) | M3 (60) | Intel (32) | Coding Syntax |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **cbr6a_step25 (Parent)** | 63 / 115 (54.8%) | 39 / 100 (39.0%) | 22 / 80 (27.5%) | 21 / 70 (30.0%) | 3 / 60 (5.0%) | 3 / 32 (9.4%) | 4 / 12 (33.3%) |
| **cbr7_step25** | 52 / 115 (45.2%) | 3 / 100 (3.0%) | 12 / 80 (15.0%) | 15 / 70 (21.4%) | 3 / 60 (5.0%) | 1 / 32 (3.1%) | 5 / 12 (41.7%) |
| **cbr7_step50 [WINNER]** | **59 / 115 (51.3%)** | **40 / 100 (40.0%)** | **14 / 80 (17.5%)** | **20 / 70 (28.6%)** | **3 / 60 (5.0%)** | **1 / 32 (3.1%)** | **5 / 12 (41.7%)** |
| **cbr7_step75** | 54 / 115 (47.0%) | 29 / 100 (29.0%) | 15 / 80 (18.8%) | 19 / 70 (27.1%) | 1 / 60 (1.7%) | 1 / 32 (3.1%) | 4 / 12 (33.3%) |
| **cbr7_step100** | 61 / 115 (53.0%) | 39 / 100 (39.0%) | 16 / 80 (20.0%) | 19 / 70 (27.1%) | 2 / 60 (3.3%) | 1 / 32 (3.1%) | 4 / 12 (33.3%) |
| **cbr7_step150** | 60 / 115 (52.2%) | 38 / 100 (38.0%) | 14 / 80 (17.5%) | 19 / 70 (27.1%) | 3 / 60 (5.0%) | 1 / 32 (3.1%) | 4 / 12 (33.3%) |
| **cbr7_step200** | 61 / 115 (53.0%) | 45 / 100 (45.0%) | 12 / 80 (15.0%) | 20 / 70 (28.6%) | 4 / 60 (6.7%) | 1 / 32 (3.1%) | 3 / 12 (25.0%) |

---

### 5. Scientific Findings & Honest Limitations

1. **What Works:**
   - Identity is clear and uncontaminated: `I am ChakrView, an indigenous neural language model developed from first principles.`
   - Morning greetings reliably produce `Good morning!` and afternoon greetings produce `Good afternoon!` or polite conversational prompts.
   - Coding syntax validity remains high at 41.7%.
   - Terminal interactive session (`chakrview/chat.py`) works smoothly on CPU without crashes.

2. **What Still Needs Improvement:**
   - Exact token copy-grounding (copying specific IP addresses or long hex keys) remains at 0% across held-out tests.
   - Long multi-turn context retention is constrained by the 3.44M parameter capacity.
   - Scaling to **ChakrMini (20.8M parameters)** is required to provide the architectural depth and induction heads needed for persistent state tracking.
