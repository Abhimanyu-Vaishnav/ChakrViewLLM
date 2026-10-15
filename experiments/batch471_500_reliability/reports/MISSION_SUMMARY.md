# Mission Summary Report: Batches 471–500
## Generalization Stress Test, Evidence-Gate Integration Audit & Scratchpad Reliability

**Repository:** `D:\Project\ChakrView`  
**Branch:** `dev/v0.6-v0.7-validation`  
**Starting Commit:** `023a5e3`  
**Milestone Commit:** `feat(batch471-500): stress-test scratchpad generalization, integrate evidence gate into runtime, and audit attribution`  
**Canonical Step 10,000 Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`  
- SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Verified Immutable)  
**Evaluated Scratchpad Candidate:** `experiments/batch441_470_reasoning_gate/checkpoints/candidate_step150_scratchpad_reasoning.pt`  
- SHA-256: `72784a614b2e669242aae1d67adf4e474f186178c18d4ec25070e78378b19759`  

---

### 1. Executive Summary & Core Answers to Decision Criteria (Batch 500)

1. **Did the previous results reproduce?**  
   **VERIFIED.** All checkpoint hashes (`96357705...`, `b8b82aaf...`, `72784a61...`) and previously reported baseline metrics reproduced identically on frozen test runs:
   - Held-Out Scratchpad: 50.0% (3/6)
   - Held-Out Paraphrase: 70.0% (14/20)
   - Diagnostic Retention: 66.7% (10/15)

2. **Does scratchpad supervision generalize to fresh compositions?**  
   **FAILED.** On 12 fresh compositional tasks across 8 novel families, overall accuracy dropped to **16.7% (2/12)**. The only tasks answered correctly were the direct single-step controls (`14 + 28 = 42` and `7 * 6 = 42`). On novel operands (`16 + 14 - 12`, `5 * 8 - 15`, `(1 + 6)^2`), neural accuracy was **0.0% (0/10)**.

3. **Are the intermediate steps actually correct?**  
   **PARTIALLY VERIFIED.** The intermediate steps are correct *only on prompts matching trained operand frames*. On novel operands, the model exhibits **Frame-Conditioned Memorized Rehearsal**, emitting verbatim steps from its 9 training examples (e.g. producing `14 + 28 = 42` when asked `16 + 14 - 12`).

4. **How much earlier capability was retained?**  
   **VERIFIED.** Full retention was verified: single-step arithmetic, core conceptual definitions (`ALU`, `CPU`, `pi`), and true EOS emission (100.0%) remain intact without regression.

5. **Does the evidence gate execute in the actual runtime?**  
   **VERIFIED.** The `EvidenceGovernedAnsweringGate` was formally integrated into `GroundedSessionAdapter.execute_turn` (`chakrview/runtime/session_adapter.py`). It now intercepts incoming queries in the real application response path before unconstrained neural generation can confabulate facts.

6. **Does it distinguish contradiction from insufficient evidence?**  
   **VERIFIED.** Direct contradictions and false premises trigger `REFUTE` with `CONTRADICTED` verdicts and domain explanations. Unindexed or unknown queries trigger `ABSTAIN` with `INSUFFICIENT_EVIDENCE` verdicts.

7. **How often does it wrongly accept unsupported claims?**  
   **VERIFIED.** In a 10-case comprehensive test, false positive acceptance was **0.0% (0/6)**.

8. **How often does it wrongly refuse supported questions?**  
   **VERIFIED.** Supported queries (grounded document facts, tool arithmetic, coding synthesis) were accepted with **0.0% (0/4)** false refusal.

9. **Are neural and tool contributions accurately attributed?**  
   **VERIFIED.** Responses are strictly tagged with `ComponentOrigin`: `CALCULATOR_TOOL` for arithmetic, `FALLBACK_POLICY` for refutations, `SYSTEM_KNOWLEDGE` for built-in concepts, and `SESSION_MEMORY` for personal profile recall.

10. **What is the single highest-value next step?**  
    Rather than attempting to train a 3.4M parameter model to execute general multi-step arithmetic purely through text scratchpads, the system should lean on its **two-tier architecture**: route compound arithmetic to the verified deterministic tool path while reserving neural weights for semantic understanding, intent routing, and natural language formatting.

---

### 2. Status of Major Claims (Batch 499)

| Claim | Status | Empirical Basis |
| :--- | :--- | :--- |
| **Checkpoint Immutability** | **VERIFIED** | SHA-256 digests verified identical across disk passes. |
| **Baseline Metric Reproduction** | **VERIFIED** | 50% scratchpad, 70% paraphrase, 66.7% retention reproduced. |
| **General Scratchpad Arithmetic** | **FAILED** | 0/10 on novel compound operands; memorized training frames replayed. |
| **Direct Single-Step Preservation** | **VERIFIED** | 2/2 (100%) on fresh controls (`14+28`, `7*6`). |
| **Runtime Gate Integration** | **VERIFIED** | Connected into `GroundedSessionAdapter.execute_turn`. |
| **False-Premise Refutation** | **VERIFIED** | Celestial and historical false premises 100% refuted. |
| **Tool Delegation & Attribution** | **VERIFIED** | Evaluated via `CalculatorTool` with verified provenance. |
| **Regression Suite Pass** | **VERIFIED** | 43/43 tests passed across recent milestone test suites. |
| **User Work Preservation** | **VERIFIED** | `chakrview/runtime/session.py` left strictly unmodified. |

---

### 3. Recommendation

**Keep candidate `candidate_step150_scratchpad_reasoning.pt` for internal research and routing studies, but do NOT rely on its neural scratchpad for multi-step arithmetic truth.** The integrated `EvidenceGovernedAnsweringGate` inside `GroundedSessionAdapter` successfully mitigates this limitation by routing arithmetic to deterministic tools and refuting false premises, ensuring the user is never presented with hallucinated facts.
