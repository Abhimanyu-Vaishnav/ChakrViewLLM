# CR-EXP-19: Dataset Quality & Contamination Audit Report

## 1. Composition Summary
- **Arm B (Extended-Horizon Treatment):** 960 total examples (720 extended-horizon 3–6 turn dialogues + 240 replay examples, exact 25.0% replay).
- **Arm A (Short-Horizon Control):** 960 total examples (720 short 1–2 turn dialogues + 240 replay examples, exact 25.0% replay).
- **Validation Split:** 80 held-out extended dialogues.

## 2. Turn Distribution in Arm B Extended-Horizon Corpus
{3: 188, 6: 182, 5: 177, 4: 176, 1: 193, 2: 44}

## 3. Disjointness and Contamination Checks
- **Train vs Validation Overlap:** 0 (Strictly Disjoint)
- **Arm B vs CR19_CONTEXT_200 Benchmark Leakage:** 0 (100% Leak-Free)
- **Arm B vs CR19_CONVERSATION_100 Benchmark Leakage:** 0 (100% Leak-Free)
- **Arm A vs Benchmarks Leakage:** 0 (100% Leak-Free)

Audit Status: **PASS (100% Contamination-Free)**
