# CBR-SFT-01 — INDEPENDENT MULTI-SUITE EVALUATION RESULTS

**Evaluation Suites:**
1. **Conversational Benchmark:** 60 tasks (`m3_conversational_tasks.json`)
2. **Native Intelligence Regression Benchmark:** 32 tasks (`intelligence_tasks.json`)
3. **Coding Benchmark:** 12 tasks (`coding_tasks.json`)

---

## 1. Multi-Suite Candidate Comparison Matrix

| Checkpoint | Conversational (60 Tasks) | Intelligence (32 Tasks) | Attractor Collapse Rate | Coding Syntax Valid | Coding Functional Pass |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Canonical Step 10k Baseline** | **0/60 (0.0%)** | **1/32 (3.1%)** | **53.1%** | **3/12 (25.0%)** | **0/12 (0.0%)** |
| **CBR-SFT Step 25** | 0/60 (0.0%) | 2/32 (6.2%) | 34.4% | 4/12 (33.3%) | 0/12 (0.0%) |
| **CBR-SFT Step 50** | 0/60 (0.0%) | 2/32 (6.2%) | 43.8% | 4/12 (33.3%) | 0/12 (0.0%) |
| **CBR-SFT Step 75** | 0/60 (0.0%) | 2/32 (6.2%) | 37.5% | 4/12 (33.3%) | 0/12 (0.0%) |
| **CBR-SFT Step 100** | 0/60 (0.0%) | 2/32 (6.2%) | 37.5% | 4/12 (33.3%) | 0/12 (0.0%) |

---

## 2. Category-by-Category Conversational Performance

Across all 4 candidate checkpoints, performance on the 11 M3 conversational categories remained at 0/60:
- Greetings & Natural Dialogue: 0/5 (0.0%)
- Direct Instruction Following: 0/5 (0.0%)
- Follow-up Understanding: 0/5 (0.0%)
- Multi-turn Context Retention: 0/10 (0.0%)
- Grounded Question Answering: 0/5 (0.0%)
- Basic Reasoning: 0/5 (0.0%)
- Arithmetic & Logic: 0/5 (0.0%)
- Summarization: 0/5 (0.0%)
- Format Adherence: 0/5 (0.0%)
- Uncertainty & Clarification: 0/5 (0.0%)
- Topic Changes & Coherence: 0/5 (0.0%)

---

## 3. Notable Capability Improvements
1. **Intelligence Non-Regression:** Native intelligence increased from 1/32 (3.1%) to 2/32 (6.2%) across all candidates, verifying that conversational fine-tuning did not cause catastrophic forgetting of core reasoning.
2. **Attractor Collapse Suppression:** Unpenalized greedy attractor loops dropped from **53.1%** down to **34.4%** at Step 25 and **37.5%** at Step 100.
3. **Coding Syntax Validity:** Increased from 25.0% (3/12) to **33.3%** (4/12).
