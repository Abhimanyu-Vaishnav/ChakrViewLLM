# CR-EXP-21: Capability Regression Report

**Experiment ID:** CR-EXP-21  
**Candidate Model:** ChakrView v0.11 (`5ae98b36...`)  
**Baseline Model:** ChakrView v0.10.1 (`548b20a7...`)  
**Parent Model:** `cbr_sft_04` (`327118ce...`)  
**Date:** 2026-11-15  

---

## 1. Regression Audit Objectives

Whenever fine-tuning is performed on an expanded or diversified corpus, small language models face severe risks of:
1. **Catastrophic Forgetting:** Loss of core persona or canonical greeting behaviors.
2. **Attractor Collapse:** Degeneration into repetitive single-phrase responses.
3. **Script Drift / Language Bleed:** Emitting Latin characters when prompted in Devanagari Hindi or vice versa.
4. **Decoding Degeneracy:** Triggering repetitive n-gram loops or early EOS termination.

---

## 2. Regression Analysis Across Evaluated Dimensions

### 2.1 Persona & English Identity Retention
- **Metric:** Exact presence of ChakrView self-identification on identity probes.
- **Baseline v0.10.1:** $150/150$ ($100.0\%$).
- **Candidate v0.11:** $150/150$ ($100.0\%$).
- **Regression:** **0.0%** (No regression).

### 2.2 Multilingual Script Integrity
- **Metric:** Devanagari Unicode character ratio on Devanagari prompts.
- **Baseline v0.10.1:** $143/150$ ($95.33\%$).
- **Candidate v0.11:** $143/150$ ($95.33\%$).
- **Regression:** **0.0%** (Identical performance; no script bleed observed).

### 2.3 Roman Hindi & Hinglish Appropriateness
- **Metric:** Lexicon matching for colloquial Hindi terms (`hai`, `bhai`, `kya`, `aap`, `theek`, etc.).
- **Baseline v0.10.1:** $150/150$ ($100.0\%$).
- **Candidate v0.11:** $150/150$ ($100.0\%$).
- **Regression:** **0.0%** (Zero regression).

### 2.4 Context Tracking & Syntactic Frame Generation
- **Metric:** Production of appropriate multi-turn anaphora and dialogue continuation frames.
- **Baseline v0.10.1:** $200/200$ ($100.0\%$).
- **Candidate v0.11:** $200/200$ ($100.0\%$).
- **Regression:** **0.0%** (Zero regression).

### 2.5 Instruction Compliance & Constraint Adherence
- **Metric:** Adherence to output formatting instructions.
- **Baseline v0.10.1:** $81/100$ ($81.0\%$).
- **Candidate v0.11:** $82/100$ ($82.0\%$).
- **Delta:** **+1.0%** improvement.

---

## 3. Runtime & Verification Parity

Both models were evaluated under identical CLI runtime conditions:
- `python -m chakrview chat --test-mode --checkpoint releases/v0.11_conversational_model/chakrview_v011_model.pt`
- **Result:** Automated verification test PASS.
- No increase in token latency ($34.8\text{ ms/token}$).
- Memory footprint unchanged ($<500\text{ MB}$ RAM).

---

## 4. Formal Regression Verdict

**VERDICT: ZERO REGRESSIONS DETECTED.**  
ChakrView v0.11 fully preserves all operational, persona, multilingual, and syntactic conversational capabilities demonstrated by v0.10.1, while delivering lower cross-entropy validation loss ($2.1737$) and marginal improvements on instruction-following benchmarks.

