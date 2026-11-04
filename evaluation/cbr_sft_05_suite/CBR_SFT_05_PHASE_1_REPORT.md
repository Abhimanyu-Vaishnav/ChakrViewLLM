# CBR-SFT-05 PHASE 1 REPORT
## Context Binding, Induction & Conversational Memory Research, Postmortem & Controlled Experiment Preparation

**Mission:** CBR-SFT-05 Phase 1 (Research, Dataset Engineering, Architectural Audit & Protocol Formulation)  
**Date:** 2026-11-04  
**Repository:** `D:\Project\ChakrView`  
**Active Branch:** `exp/cbr-01-conversational-breakthrough` (`257801b4c0fcc3017bb450b104baee5971ad530a`)  
**Designated Parent Checkpoint:** CBR-SFT-04 Step 200 (`327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153`)  
**Alternate Baseline Comparison Checkpoint:** CBR-SFT-03B Step 200 (`297dcc60f39cadd6bae9a5bd969b1a7d831845d7a67823b30c194a67ef75b5ac`)  
**Execution Gate Status:** **`READY FOR PHASE 2`**

---

## 1. Executive Summary

In CBR-SFT-04, ChakrView successfully eliminated temporal greeting bias (achieving 60% accuracy on afternoon and 60% on evening greetings with zero regressions on M3, intelligence, or coding benchmarks). However, **exact entity grounding (0/20)** and **multi-turn context retention (0/20)** remained completely unlearned.

Phase 1 of **CBR-SFT-05** conducted an in-depth forensic diagnosis of these two failure modes, audited the canonical architecture's capacity for attention routing, and authored an isolated **400-example SFT dataset** (360 train, 40 validation) alongside an independent **120-task held-out evaluation suite**.

### Core Diagnoses:
1. **The Architecture is Sound:** The 6-layer ChakrMicro model with Rotary Position Embeddings (RoPE) and causal attention masking has full mathematical access to past context tokens. In layer 4, attention from the query token concentrates up to 18.0% on salient context tokens.
2. **The Bottleneck was Data Density & Pattern Mismatch:** CBR-SFT-04 contained only 40 grounding and 40 two-turn examples embedded in 80 greeting and identity examples. The model learned conversational style (first-token greetings and epistemic abstentions), but never encountered enough paired variable bindings to form robust induction circuits.
3. **CBR-SFT-05 Strategy:** Provide a dense, progressive curriculum covering exact entity extraction (100), key-value bindings and updates (100), multi-turn context retention across 2, 3, and 4 turns (100), and natural conversational transfer (100).

---

## 2. Invariant & Safety Verification

All project safety rules and non-destructive constraints were strictly verified:
- **Active Git Branch:** `exp/cbr-01-conversational-breakthrough`
- **Current HEAD Commit:** `257801b4c0fcc3017bb450b104baee5971ad530a`
- **Zero Destructive Actions:** No `git reset`, `git clean`, or history modifications.
- **Protected File Hashes (100% Bit-Exact Match):**
  - `chakrview/brain/model.py`: `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` — **PASS**
  - `chakrview/runtime/session.py`: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` — **PASS**
  - `Canonical Step 10k Checkpoint`: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` — **PASS**
  - `Parent CBR-SFT-02 Step 50`: `592553c3e27b654997e8f5df61e94fe1de25bbf577391df6c1a87f974730bf9d` — **PASS**
  - `Parent CBR-SFT-03B Step 200`: `297dcc60f39cadd6bae9a5bd969b1a7d831845d7a67823b30c194a67ef75b5ac` — **PASS**
  - `Parent CBR-SFT-04 Step 200`: `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` — **PASS**
- **Isolation Boundaries:** `experiments/overnight_runs/` unaccessed and unmodified. `SmartProjectMemory` untouched.
- **Phase 1 Limitation:** Zero training steps run; zero model weights created.

---

## 3. CBR-SFT-04 Postmortem & Root Cause Analysis

Inspection of raw model generations (`raw_model_outputs.jsonl`) revealed the exact mechanisms behind the 0/20 scores:
- **Entity Grounding Failures:** When prompted with `Context: In laboratory 1000, researcher Aarav supervises operations.\nQuestion: Who supervises laboratory 1000?\nAssistant:`, the model generated `" 250."` or `" 4a8b6c6d5e"` or generic boilerplate (*"The provided context does not..."*). The model recognized the *style* of an entity answer or abstention, but lacked the associative binding to copy the specific token `Aarav` from the context.
- **Multi-Turn Retention Failures:** When prompted with `User: My name is Arjun...\nAssistant: Hello Arjun!...\nUser: What is my name?\nAssistant:`, the model generated pre-training identity fragments (*"Your first principles."* or *"You are testing the neural language model develop."*). The model attended almost exclusively to the final turn (`User: What is my name?`) and generated generic responses rather than querying Turn 1.

### Root-Cause Classification Table

| Defect / Observation | Classification | Evidence & Impact | Remediation in CBR-SFT-05 |
| :--- | :--- | :--- | :--- |
| **Out-of-Distribution Entity Copying** | Confirmed Data & Induction Deficit | Model output generic tokens because training only contained 40 entity examples—insufficient for a 3.44M model to learn copy routing. | 100 dedicated Track A copying tasks + 100 Track B key-value bindings. |
| **Multi-Turn State Recall** | Confirmed Data Sparsity & Context Bias | Model defaulted to single-turn responses because only 40 multi-turn examples existed in training. | 100 Track C multi-turn examples scaled from 2 to 4 turns with explicit distractor tasks. |
| **Abstention Hallucinations** | Plausible Model Limitation / Threshold | Model over-applied abstention phrases to valid grounding queries when uncertain. | Balanced 80% answerable vs 20% abstention supervision across all tracks. |
| **Tokenizer Subword Boundary Shifts** | Confirmed Evaluation Artifact (Legacy Regex) | Dot-delimited IPs (`10.0.4.12.`) failed legacy word-boundary regexes. | Addressed by DualScorer in `evaluator.py`. |
| **Attention Mask & Serialization Integrity** | Verified Valid (Not a Defect) | Full prompt context is passed; causal mask and RoPE offsets are mathematically bit-exact. | Maintained canonical architecture and serialization contract. |

---

## 4. Architecture and Training Pipeline Diagnosis

A comprehensive source-level audit of the ChakrMicro neural core and Stage D training engine confirmed:
1. **Earlier Turn Attention:** Permitted by design. In `chakrview/brain/attention.py`, `mask = (k_pos > q_pos)` allows any token at position $T$ to attend to all tokens from $0$ to $T$.
2. **RoPE Offsets:** Rotary embeddings are computed over the full sequence length $T$, ensuring relative positional distances are preserved across dialogue turns.
3. **Loss Masking Parity:** In `chakrview/training/sft_engine.py`, prompt tokens are masked with `-100`, and supervision starts precisely on the first token of the assistant's response.
4. **Tokenizer Invertibility:** Technical strings (IPs, hashes, paths, codes) decode 100% losslessly through the 4096-vocab BPE tokenizer.
5. **Mechanistic Layer Analysis:** Attention distributions show that by Layer 4, the model attends with significant weight (up to 18%) to numerical and semantic tokens in the context prompt. The capacity for attention routing exists natively; SFT must train the projection weights to transfer this attention into next-token logits.

---

## 5. CBR-SFT-05 Dataset Design & Manifest

An isolated, reproducible dataset of **400 examples** was created in `data/sft/cbr_sft_05/`:
- **Track A (100 examples): Exact Copy & Entity Extraction**  
  IP addresses, ports, git commit hashes, file paths, technical IDs, punctuation boundaries, and candidate selection.
- **Track B (100 examples): Key-Value & Variable Binding**  
  Single key-values (`PROJECT=ChakrView`), multi-key configuration blocks, sequential value updates, and unknown key abstentions.
- **Track C (100 examples): Multi-Turn Context Retention**  
  Two-turn context recall (30), three-turn context recall (30), four-turn dialogues with distractor turns (40).
- **Track D (100 examples): Natural Conversational Transfer**  
  Natural technical dialogues, developer preferences, architectural specifications, and rich scenarios.

### Partitioning & Stratification:
- **Total Records:** 400
- **Training Set (`cbr_sft_05_train_360.jsonl`):** 360 examples (90%)
- **Validation Set (`cbr_sft_05_val_40.jsonl`):** 40 examples (10%)
- **Leakage Audit:** 0 prompt collisions, 0 cross-split entity collisions (`duplicate_audit.json`).
- **Context Lengths:** Min 51 tokens, Mean 91.7 tokens, Max 182 tokens (well within the canonical 512 context window).

---

## 6. Independent Held-Out Evaluation Suite (120 Tasks)

Authored in `evaluation/cbr_sft_05_suite/held_out_cbr_sft_05_120.json`:
- **Track A (30 tasks):** Exact copying of novel IP addresses, ports, commit hashes, file paths, and technical IDs.
- **Track B (25 tasks):** Key-value retrieval on novel configuration keys, latest-value updates, and explicit abstentions.
- **Track C (35 tasks):** Multi-turn context recall across 2-turn, 3-turn, and 4-turn dialogues with distractor messages.
- **Track D (30 tasks):** Natural conversational transfer across novel facts, preferences, and system parameters.
- **Evaluator:** [`evaluation/cbr_sft_05_suite/evaluator.py`](file:///D:/Project/ChakrView/evaluation/cbr_sft_05_suite/evaluator.py) implementing character-level exact matching, normalized comparison, and abstention cluster detection.

---

## 7. Curriculum, Ablation & Mechanistic Diagnostic Plan

### Curriculum Stages:
- **Stage A (Steps 1–100):** Exact copy and key-value binding acquisition.
- **Stage B (Steps 101–200):** Recency updates and 2-to-3 turn context retention.
- **Stage C (Steps 201–300):** 4-turn distractor resistance and natural conversational transfer.

### Proposed Ablation Experiments:
1. **Ablation 1 (Flat Training):** Train on all 400 examples uniformly for 300 steps from CBR-SFT-04 Step 200.
2. **Ablation 2 (Synthetic Key-Value Only):** Train exclusively on Tracks A and B to measure whether synthetic key-value binding transfers to multi-turn dialogue without explicit multi-turn data.
3. **Ablation 3 (Direct from Canonical 10k Baseline):** Train on CBR-SFT-05 directly from Canonical Step 10,000 to determine whether the intermediate CBR-SFT-02/03B/04 checkpoints were necessary prerequisites.

### Mechanistic Diagnostic Battery:
1. **Entity Permutation Test:** Keep prompt structure constant while swapping entity values (`PORT=8080` vs `PORT=9090`) to verify whether output tracks the context or hallucinated priors.
2. **Context Distance Decay Test:** Move the target entity from position 10 to position 150 to measure attention degradation over token distance.
3. **Distractor Insertion Test:** Insert 1 to 3 irrelevant conversational turns between the entity declaration and query to measure resistance to dialogue interference.

---

## 8. Checkpoint Lineage & Artifact Hashes Checked

| Artifact Description | Path | SHA-256 Hash |
| :--- | :--- | :--- |
| **Model Architecture** | `chakrview/brain/model.py` | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` |
| **Runtime Session** | `chakrview/runtime/session.py` | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` |
| **Canonical Base (Step 10,000)** | `experiments/batch16_21_combined/.../training_checkpoint_step_10000.pt` | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| **CBR-SFT-02 Parent (Step 50)** | `experiments/cbr01_sft/runs/cbr_sft_02/sft_checkpoint_step_50.pt` | `592553c3e27b654997e8f5df61e94fe1de25bbf577391df6c1a87f974730bf9d` |
| **CBR-SFT-03B Parent (Step 200)** | `experiments/cbr01_sft/runs/cbr_sft_03b/sft_checkpoint_step_200.pt` | `297dcc60f39cadd6bae9a5bd969b1a7d831845d7a67823b30c194a67ef75b5ac` |
| **CBR-SFT-04 Parent (Step 200)** | `experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt` | `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153` |
| **CBR-SFT-05 Full Dataset** | `data/sft/cbr_sft_05/dataset.jsonl` | `5cc7297504ac1608e9b7138dff29b1fb34259db8bfc8d09a7f1638d93c383cad` |
| **CBR-SFT-05 Train Split (360)** | `data/sft/cbr_sft_05/splits/cbr_sft_05_train_360.jsonl` | `0329e7a75c6ad69f887901dc0ad608753b10e2bd0021409545be832feb5ac79c` |
| **CBR-SFT-05 Val Split (40)** | `data/sft/cbr_sft_05/splits/cbr_sft_05_val_40.jsonl` | `1640b77fe4d64ef68fbe0b73d26da9e3adf0d51850016cb2ceaf86230fb30cc9` |
| **CBR-SFT-05 Held-Out Eval (120)** | `evaluation/cbr_sft_05_suite/held_out_cbr_sft_05_120.json` | `94463f431a9449524040c3a415937c34b93979eee98c57ef65f1cff0649acc10` |
| **Dataset Generator Script** | `scripts/build_cbr_sft_05_dataset.py` | `0225d30fba8c950d8b58a1bb0dfc4a242c7322927ff77f72295cb18751532050` |
| **Evaluator Engine** | `evaluation/cbr_sft_05_suite/evaluator.py` | `d70b67362df50058b8f36c533e4b77918a2212d1c68f1c841e21b7952e4ea830` |

---

## 9. Comprehensive Classification of Phase 1 Invariants

| Invariant / Check | Classification | Evidence & Status |
| :--- | :---: | :--- |
| **Working Tree & Protected Files Unchanged** | **PASS** | Exact hashes verified for `model.py`, `session.py`, Step 10k, CBR2, CBR3B, CBR4. |
| **Isolated Target Directories** | **PASS** | `data/sft/cbr_sft_05/` and `evaluation/cbr_sft_05_suite/` created in isolation. |
| **Zero Model Training in Phase 1** | **PASS** | No training steps executed; no optimizer states created. |
| **400-Example Dataset Verification** | **PASS** | Exactly 100 per track across Tracks A, B, C, D. |
| **Stratified Split Integrity** | **PASS** | 360 train, 40 val; all conversation threads kept intact. |
| **Zero Prompt / Entity Leakage** | **PASS** | 0 collisions between train, val, and held-out 120 tasks. |
| **Canonical Context Window Fit** | **PASS** | Maximum sequence length 182 tokens (fits within 512 context window). |
| **Independent 120-Task Evaluation Suite** | **PASS** | Fully authored with novel entities, keys, and dialogue turns. |
| **Mechanistic Attention Audit** | **PASS** | Verified attention concentration up to 18% in Layer 4. |

---

## 10. Execution Gate Verdict

### Verdict: **`READY FOR PHASE 2`**

All dataset engineering, split auditing, evaluation suite preparation, tokenizer verification, architectural diagnostics, and parent checkpoint validations are complete. The project is fully prepared for controlled Phase 2 training upon human review and approval.

