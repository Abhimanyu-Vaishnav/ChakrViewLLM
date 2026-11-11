# CR-EXP-17: Baseline Evaluation Report

## 1. Frozen Benchmark Specification

The held-out evaluation suite **`CONVERSATION_BENCHMARK_V1.json`** (`d2472e0b2a72f2555bea68511ac7373c5f2983f785bb9a7d60da4dd9ac2f1a5f`) contains exactly 200 independently authored multi-turn and single-turn conversational probes across 12 distinct capabilities:
- Basic conversation (20)
- Follow-up understanding (18)
- Context retention (18)
- Multi-turn continuity (18)
- Reference resolution (16)
- Topic transition (16)
- Clarification (16)
- Instruction following (16)
- Naturalness (16)
- Consistency (16)
- Language consistency / Hindi & Hinglish (16)
- Response completion (14)

A strict leakage audit verified that **0 out of 200 probes** overlap with any existing training corpus in `data/` or `experiments/`.

---

## 2. Comparative Baseline Evaluation Across 6 Checkpoints

All candidates were evaluated under identical calibrated decoding parameters (`repetition_penalty = 1.15`, `no_repeat_ngram_size = 3`, `max_new_tokens = 35`):

| Candidate Checkpoint | Lineage Origin | SHA-256 Digest | Keyword Hits (/200) | Non-Empty Responses | Repetition Cleanliness |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Stage B Baseline** | Pretraining Step 10,000 | `96357705...` | 6/200 (3.0%) | 174/200 (87.0%) | 100.0% |
| **CR-16A Arm D** | Algorithmic Copy Parent | `9d609a77...` | 14/200 (7.0%) | 200/200 (100.0%) | 94.5% |
| **Conversational SFT Parent** | CBR-SFT-04 Step 200 | `327118ce...` | **27/200 (13.5%)** | 200/200 (100.0%) | 100.0% |
| **Conversational Pilot Best** | Autonomous 12H Pilot | `ea6ee570...` | **29/200 (14.5%)** | 200/200 (100.0%) | 100.0% |
| **CBR-SFT-07 Step 50** | SFT Context Over-trained | `0cb5df5f...` | 19/200 (9.5%) | 199/200 (99.5%) | 100.0% |
| **CBR-SFT-08 Step 60** | Universal Capability Run | `sft_checkpoint...` | 17/200 (8.5%) | 198/200 (99.0%) | 100.0% |

---

## 3. Analysis & Checkpoint Selection for Phase E Initialization

1. **Failure of Raw Base & Algorithmic Models:** Canonical Stage B (3.0%) and CR-16A Arm D (7.0%) completely lack conversational turn awareness, producing raw pretraining table tokens or copying Greek letter fragments.
2. **Context-Sink Degradation in Late SFT Models:** CBR-SFT-07 (9.5%) and CBR-SFT-08 (8.5%) suffered regressions due to the "context refusal attractor sink" (`"not provided in the context"`).
3. **The Strongest Legitimate Starting Points:**
   - **Conversational SFT Parent (CBR-SFT-04 Step 200):** Best basic conversational responsiveness (12/20 hits on greetings/persona) and highest consistency (10/16 hits).
   - **Conversational Pilot Best (Step 100):** 29/200 hits (14.5%), but has target-length collapse on multi-turn queries.

**Decision:** We will train primarily from **Conversational SFT Parent (CBR-SFT-04 Step 200)**, which retains broad English language capability without the severe refusal sink of CBR-SFT-07/08.
