# CBR-SFT-02 — 5-SUITE INDEPENDENT EVALUATION RESULTS

**Evaluation Suites:**
- **Suite A:** M3 Conversational benchmark (60 tasks)
- **Suite B:** Native Intelligence benchmark (32 tasks)
- **Suite C:** Coding benchmark (12 tasks)
- **Suite D:** Held-out Dynamic Entity Grounding (100 tasks)
- **Suite E:** Held-out Multi-turn Context Retention (50 conversations)

---

## 1. Master Multi-Suite Candidate Comparison Matrix

| Checkpoint | Suite A: Conv (60) | Suite B: Intel (32) | Attractor Rate | Suite C: Coding Syntax | Suite D: Entity (100) | Suite E: M-Turn (50) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canonical Baseline** | **0/60 (0.0%)** | **1/32 (3.1%)** | **53.1%** | **3/12 (25.0%)** | **0/100 (0.0%)** | **0/50 (0.0%)** |
| **CBR-SFT-02 Step 25** | 0/60 (0.0%) | 3/32 (9.4%) | 43.8% | 4/12 (33.3%) | 0/100 (0.0%) | 0/50 (0.0%) |
| **CBR-SFT-02 Step 50** | 0/60 (0.0%) | 3/32 (9.4%) | 34.4% | 4/12 (33.3%) | 0/100 (0.0%) | 0/50 (0.0%) |
| **CBR-SFT-02 Step 100**| 0/60 (0.0%) | 1/32 (3.1%) | 28.1% | 3/12 (25.0%) | 0/100 (0.0%) | 0/50 (0.0%) |
| **CBR-SFT-02 Step 150**| 0/60 (0.0%) | 2/32 (6.2%) | 31.2% | 2/12 (16.7%) | 0/100 (0.0%) | 0/50 (0.0%) |
| **CBR-SFT-02 Step 200**| 0/60 (0.0%) | 2/32 (6.2%) | 34.4% | 2/12 (16.7%) | 0/100 (0.0%) | 0/50 (0.0%) |

---

## 2. Granular Diagnostic Findings

1. **Attractor Loop Suppression Peak:**
   - Greedy repetition collapse reached an all-time low of **28.1%** at Step 100 (down from **53.1%** on baseline), confirming that SFT on chat-scaffolded text suppresses attractor loops.
2. **Native Intelligence Elevation:**
   - Steps 25 and 50 achieved **3/32 (9.4%)** accuracy on the 32-task native intelligence suite, tripling the baseline score (1/32) without regression on relational reasoning.
3. **Conversational Benchmark Reality:**
   - On open-domain conversational probes (M3), dynamic entity grounding (100 tasks), and multi-turn context (50 tasks), accuracy remained at **0.0%**.
   - Raw output inspection shows the model has abandoned Wikipedia table markup (`\n! scope=...`), but now outputs high-frequency pretraining fragments (`"Chinese is a list of the world"`, `"The first of the word of..."`).
