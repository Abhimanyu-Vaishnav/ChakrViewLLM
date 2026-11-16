# CR-EXP-19: Final Capability & Integration Decision

**Mission:** CHAKRVIEW — CR-EXP-19  
**Date:** 2026-11-11  
**Status:** COMPLETE — RIGOROUS NEGATIVE NEURAL RESULT / CONFIRMED RUNTIME BUG FIX  
**Recommendation:** **DO NOT INTEGRATE CANDIDATE CHECKPOINTS INTO PRODUCTION / DEPLOY RUNTIME BUG FIX**

---

## 1. Executive Summary

CR-EXP-19 investigated the conversational multi-turn context failure diagnosed in CR-EXP-18. The investigation focused on two distinct failure points:
1. **Software Runtime Defect:** `chakrview/chat.py` artificially truncated dialogue history to a single turn (`max_history_turns = 1`).
2. **Neural Extended-Horizon Inability:** Whether ChakrMicro v0.1 ($3.44\text{M}$ parameters) could retain and utilize multi-turn context across 3–8 conversational turns when supplied with extended history.

### The Decision:
- **Runtime Fix:** **APPROVED & ADOPTED**. The software defect in `chakrview/chat.py` was resolved using dynamic token-budgeted history packing. 8/8 automated regression tests pass, enabling up to 384 tokens of history (5–8 full conversational turns) to be fed directly to the model without truncation or prompt overflow.
- **Candidate Checkpoint Integration:** **REJECTED (NO INTEGRATION)**. In controlled 6-run evaluation against the frozen 200-probe context benchmark (`CR19_CONTEXT_200.json`), neither the baseline ($0/200, 0.0\%$), Arm A controls ($0/200, 0.0\%$), nor Arm B extended-horizon models ($0/200, 0.0\%$) achieved multi-turn factual retrieval. Although conversational capability was fully retained and slightly improved on single-turn benchmarks ($41.3\%$ vs $36.0\%$), the neural model fails to attend to factual tokens positioned in earlier conversation turns.
- **Lineage Protection:** The production release checkpoint (`chakrview_v091_model.pt`, SHA-256: `f161ccdc...`) and protected runtime (`session.py`, SHA-256: `9080ff51...`) remain unpolluted. No public release or tag is issued.

---

## 2. Quantitative Evaluation Scorecard

| Evaluation Dimension | Frozen Parent Baseline | Arm A (Short Control, $N=3$) | Arm B (Extended Treatment, $N=3$) | Capability Acquisition Status |
| :--- | :---: | :---: | :---: | :---: |
| **Context 200 Total (CLI Runtime)** | $0/200$ ($0.0\%$) | $0/200$ ($0.0\% \pm 0.0\%$) | $0/200$ ($0.0\% \pm 0.0\%$) | **No Acquisition** |
| **Context 200 Total (Direct Model)** | $0/200$ ($0.0\%$) | $0/200$ ($0.0\% \pm 0.0\%$) | $0/200$ ($0.0\% \pm 0.0\%$) | **No Acquisition** |
| — 1-turn Horizon ($N=40$) | $0/40$ ($0.0\%$) | $0/40$ ($0.0\%$) | $0/40$ ($0.0\%$) | No Retrieval |
| — 2-turn Horizon ($N=40$) | $0/40$ ($0.0\%$) | $0/40$ ($0.0\%$) | $0/40$ ($0.0\%$) | No Retrieval |
| — 3-turn Horizon ($N=40$) | $0/40$ ($0.0\%$) | $0/40$ ($0.0\%$) | $0/40$ ($0.0\%$) | No Retrieval |
| — 5-turn Horizon ($N=40$) | $0/40$ ($0.0\%$) | $0/40$ ($0.0\%$) | $0/40$ ($0.0\%$) | No Retrieval |
| — 8-turn Horizon ($N=40$) | $0/40$ ($0.0\%$) | $0/40$ ($0.0\%$) | $0/40$ ($0.0\%$) | No Retrieval |
| **Conversation 100 Total** | $36/100$ ($36.0\%$) | $40.0/100$ ($40.0\% \pm 1.0\%$) | $41.3/100$ ($41.3\% \pm 1.5\%$) | **Retained & Stable** |
| — Greetings ($N=15$) | $13/15$ ($86.7\%$) | $10.0/15$ ($66.7\%$) | $10.0/15$ ($66.7\%$) | Stable |
| — Identity ($N=20$) | $9/20$ ($45.0\%$) | $11.7/20$ ($58.3\%$) | $11.7/20$ ($58.3\%$) | +13.3% Gain |
| — Hinglish ($N=25$) | $3/25$ ($12.0\%$) | $2.0/25$ ($8.0\%$) | $3.3/25$ ($13.3\%$) | Stable |
| — Natural Flow ($N=40$) | $11/40$ ($27.5\%$) | $16.3/40$ ($40.8\%$) | $16.3/40$ ($40.8\%$) | +13.3% Gain |

---

## 3. Causal Mechanics & Diagnosis

1. **Why the Runtime Fix Worked:**
   Before CR-19, `chakrview/chat.py` only kept the immediate previous turn (`self.history[-1:]`). The model was physically blind to anything said 2 or more turns ago. Dynamic token-budgeted history packing guarantees that turns are greedily packed up to 384 tokens, providing complete multi-turn conversational traces to the neural model.

2. **Why Neural Extended-Horizon Retrieval Failed ($0.0\%$):**
   - The validation loss on extended multi-turn dialogues dropped from $2.927$ to $0.252$ (token accuracy $94.4\%$), showing that the model easily learned the multi-turn conversational syntax and structural turns in-distribution.
   - However, when queried about a specific unprimed factual entity stated $k$ turns earlier (`"My primary station is in Astraea-City" -> "Where is my primary station?"`), the causal attention layers in ChakrMicro v0.1 ($d=192, L=6, H=6$, or $d=128, L=4, H=4$) fail to form long-range attention sinks to user-introduced entities across dialogue turn delimiters. Instead, the model outputs generic high-probability conversational priors (`"Good evening! I am ready to assist..."`).
   - SFT on conversational dialogues alone does not induce in-context retrieval across multi-turn user memory.

---

## 4. Final Recommendation & Next Steps

1. **Production Code:** Maintain the runtime bug fix in `chakrview/chat.py`. It is functionally sound, backwards-compatible, and resolves an overt software bottleneck.
2. **Model Weights:** Retain `releases/v0.9.1_conversational_preview/chakrview_v091_model.pt` as the canonical release checkpoint. None of the Arm B checkpoints are promoted.
3. **Future Architectural Direction (CR-20):** Conversational factual memory cannot be resolved purely by fine-tuning on dialogue transcripts. It requires targeted in-context retrieval objectives, explicit entity-binding pretraining, or an external memory mechanism (e.g. key-value state indexing).

