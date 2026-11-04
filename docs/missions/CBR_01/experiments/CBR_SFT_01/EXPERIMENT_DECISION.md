# CBR-SFT-01 — EXPERIMENT DECISION & SCIENTIFIC ROADMAP

---

## 1. Candidate Selection Decision

**Decision: REJECT CANDIDATES AS CONVERSATIONAL BASELINE; RETAIN EXPERIMENTAL CHECKPOINTS FOR LINEAGE**

- **Candidate Checkpoints:** Steps 25, 50, 75, 100 in `experiments/cbr01_sft/runs/cbr_sft_01/`.
- **Reason:** The candidate checkpoints did not pass the mandatory $\ge 10.0\%$ (6/60) Conversational Acceptance Gate defined in `CONVERSATIONAL_RELEASE_GATES.md`.
- **Reference Checkpoint:** Canonical Step 10,000 remains the official, immutable baseline.

---

## 2. Positive Scientific Findings from CBR-SFT-01

1. **Attractor Collapse Suppressed:** Greedy attractor loops dropped significantly from **53.1%** to **34.4%** (Step 25).
2. **Intelligence Non-Regression:** Native intelligence pass rate increased from 3.1% (1/32) to **6.2%** (2/32).
3. **Coding Syntax Validity:** Increased from 25.0% to **33.3%**.
4. **Stable CPU Execution:** Verified that 100 steps of fine-tuning execute reliably on CPU in under 15 seconds without gradient anomalies.

---

## 3. Recommended Next Experiment: CBR-SFT-02 (Unified Chat Scaffolding & Dynamic Grounding)

To break the 0/60 floor, the next experiment must eliminate the serialization mismatch and combine diverse responses with explicit chat scaffolding:
1. **Chat Format Parity:** Prepend `User: {prompt}\nAssistant: {response}` across all training records so the model learns that `Assistant:` predicts the answer turn.
2. **Balanced Conversational Curriculum:** Mix 300 chat-scaffolded conversational pairs (greetings, role questions, conversational follow-ups, and clarification abstentions) with 300 dynamic single-entity copy pairs.
3. **Hypothesis:** Eliminating the chat delimiter mismatch while preserving response diversity will yield the first non-zero pass rate ($\ge 10\%$) on the M3 conversational benchmark.
