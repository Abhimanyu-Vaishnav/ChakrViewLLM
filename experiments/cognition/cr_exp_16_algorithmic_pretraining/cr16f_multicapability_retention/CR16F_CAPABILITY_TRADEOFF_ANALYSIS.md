# CR-EXP-16F: Catastrophic Forgetting & Multi-Capability Trade-Off Analysis

## 1. Longitudinal Lineage Comparison (Parent vs. CR-16D-R vs. CR-16E-R2 vs. CR-16F)

To understand capability acquisition, retention, and forgetting, we compare the current CR-16F models against their direct ancestral lineage:
- **Parent Checkpoint:** CR-16A Arm D (Copying acquisition champion)
- **CR-16D-R:** Multi-task consolidation study (replay without abstention tuning)
- **CR-16E-R2 G1:** Full negative loss ($\lambda=1.00$, no replay)
- **CR-16E-R2 G2:** Zero negative loss ($\lambda=0.00$, no replay)
- **CR-16F Arm A:** 25% Replay, $\lambda=1.00$
- **CR-16F Arm B:** 50% Replay, $\lambda=1.00$
- **CR-16F Arm C:** 25% Replay, $\lambda=0.25$
- **CR-16F Arm D:** 50% Replay, $\lambda=0.25$

### Capability Progression Table

| Capability Metric | Parent CR-16A Arm D | CR-16D-R Arm C | CR-16E-R2 G1 ($\lambda=1.0$) | CR-16E-R2 G2 ($\lambda=0.0$) | CR-16F Arm A (25%, 1.0) | CR-16F Arm B (50%, 1.0) | CR-16F Arm C (25%, 0.25) | CR-16F Arm D (50%, 0.25) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Copying (/100)** | **66/100** | 83/100 | 0.33/100 | 1.00/100 | 0.67/100 | 1.00/100 | 0.67/100 | 0.33/100 |
| **CR-16C Answerable (/1,400)** | 0/1,400 | ~763/1,400 | 101.67/1,400 | **208.33/1,400** | 65.33/1,400 | 67.33/1,400 | 145.67/1,400 | 127.33/1,400 |
| **CR-16C False Abstention (/1,400)** | 0/1,400 | ~100/1,400 | 699.00/1,400 | **0.00/1,400** | 1045.67/1,400 | 960.67/1,400 | 596.33/1,400 | 591.33/1,400 |
| **CR-16C Unans CAT9 (/100)** | 0/100 | 18/100 | 64.00/100 | **0.00/100** | 92.33/100 | 80.67/100 | 47.33/100 | 46.00/100 |
| **B1 Position Invariance (/600)** | 0/600 | — | 143.00/600 | 163.67/600 | 136.33/600 | 133.00/600 | **168.00/600** | 146.33/600 |
| **B2 Permutation Pairs (/150)** | 0/150 | — | — | — | 0.33/150 | 1.00/150 | **10.00/150** | 9.00/150 |
| **B2 Counterfactual Pairs (/150)** | 0/150 | — | — | — | 0.67/150 | 2.67/150 | **13.67/150** | 10.00/150 |
| **B3 Multi-Token Spans (/500)** | 0/500 | — | 136.00/500 | 203.67/500 | 67.67/500 | 80.67/500 | 148.00/500 | 122.67/500 |
| **B4 Abstention Recall (/300)** | 0/300 | — | 227.00/300 | **0.00/300** | **289.33/300** | 270.67/300 | 201.00/300 | 180.33/300 |
| **CAT7 Unseen-Value (/150)** | 0/150 | 0/150 | 0/150 | 0/150 | 0/150 | 0/150 | 0/150 | 0/150 |

---

## 2. In-Depth Trade-Off Findings

### A. The Irrecoverable Sequence Copying Collapse
In CR-16A Arm D, the parent model learned algorithmic sequence copying (66/100 exact matches). In CR-16D, when training on single-template simple retrieval paired with 50% copying replay, copying was retained (83/100).
However, once training data incorporated **shuffled position invariance, variable length spans, and explicit unanswerable negative abstentions (`[UNKNOWN]`)**, simple batch replay of copying completely failed to preserve copying capability ($0 - 1 / 100$).
- Qualitative inspection of copying errors shows that rather than copying the Greek letter sequences, the models output fractured subwords, military alphabet phonetic tokens, or retrieval entity fragments (e.g., prompt `Echo sequence: delta beta omicron zeta` yielded `delta silron zulu` or `delta echo`).
- This confirms that **3.4M parameter capacity is subject to fundamental representational interference** between position-invariant retrieval routing and verbatim token transmission.

### B. The Abstention-Retrieval Trade-Off Dilemma
Comparing $\lambda = 1.00$ vs $\lambda = 0.25$:
- **$\lambda = 1.00$ (Arms A & B):** The model aggressively learns to predict `[UNKNOWN]`. While CAT9 recall is $80 - 92\%$ and B4 recall is $90 - 96\%$, the model suffers catastrophic false abstentions on answerable queries ($960 - 1045$ out of 1,400 answerable probes are falsely aborted with `[UNKNOWN]`), resulting in dismal answerable retrieval ($65 - 67 / 1,400$).
- **$\lambda = 0.25$ (Arms C & D):** Attenuating the abstention gradient relieves suppression of answerable keys, cutting false abstentions nearly in half ($591 - 596 / 1,400$) and doubling answerable retrieval ($127 - 146 / 1,400$). However, abstention accuracy collapses: CAT9 recall drops below 50% ($46 - 47\%$) and B4 recall drops to $60 - 67\%$.
- Neither setting yields a balanced, calibrated agent. ChakrMicro v0.1 lacks sufficient capacity or routing separation to independently gate answerability without distorting feature representations for answerable retrieval.

### C. Persistent Zero-Shot Failure on Unseen Values (CAT7)
Across all 12 models in all 4 arms and all 3 seeds, CAT7 unseen-value accuracy was identically **$0/150$ ($0.0\% \pm 0.0\%$)**.
ChakrMicro v0.1 does not perform general algorithmic pointer retrieval or symbolic variable binding; it retrieves facts by memorizing low-rank associative pathways between known key tokens and known value subwords. When presented with values never encountered during pretraining/fine-tuning, it cannot bind or generate them.

