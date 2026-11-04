# CBR-SFT-07 DATASET AUDIT REPORT

## Executive Summary
The CBR-SFT-07 curated conversational dataset contains 250 high-quality balanced examples across 10 functional categories.
It was designed to transform ChakrView into an interactive, grounded, and polite conversational assistant without causing catastrophic forgetting or temporal greetings collapse.

## Dataset Integrity & Hashes
- **Full Dataset:** data/sft/cbr_sft_07/dataset.jsonl
  - Total items: 250
  - SHA-256: 923e17640f54a3d0d34c016f67e31400aaead836688a340850af718e19d49d57
- **Train Split (91.2%):** data/sft/cbr_sft_07/splits/cbr_sft_07_train_228.jsonl
  - Total items: 228
  - SHA-256: 148f112c355a6e0942f438ef1cb21ad6d93d5171547a8309801a61f6b0ae3add
- **Validation Split (8.8%):** data/sft/cbr_sft_07/splits/cbr_sft_07_val_22.jsonl
  - Total items: 22
  - SHA-256: 099ebd9a82d328beb972e74efd39f07c56299617e8ab922b9ea2b8c215fe8d8c

## Category Distribution
- greetings_social: 30 items (12.0%) - Morning, Afternoon, Evening, Generic
- identity_awareness: 25 items (10.0%) - Architecture, CPU nature, creators, capabilities
- 
atural_qa: 25 items (10.0%) - OS, RAM, CPU, networking, computing
- context_grounding: 25 items (10.0%) - Exact entity copying, IP addresses, names
- multiturn_dialogue: 25 items (10.0%) - Context tracking, 2-turn and 3-turn dialogues
- explanations_reasoning: 25 items (10.0%) - Comparisons, step-by-step logic
- supportive_conversation: 20 items (8.0%) - Encouraging, empathetic, structured
- hindi_hinglish: 30 items (12.0%) - Roman Hinglish & Devanagari Hindi
- bstention: 20 items (8.0%) - 5 distinct abstention templates to prevent phrase collapse
- continuity_feedback: 25 items (10.0%) - Follow-up questions, clarifying explanations

## Contamination & Leakage Audit
- **Train vs Validation Overlap:** 0 prompts (0.0%).
- **Evaluation Contamination:**
  - Audited across all benchmark suites (m3_conversational_tasks.json, held_out_cbr_sft_04_80.json, held_out_cbr_sft_05_120.json, held_out_cbr_sft_06a_100.json, held_out_greeting_identity_70.json, 
ative_intelligence_suite.json, coding_tasks.json).
  - Total overlaps detected: 0. Contamination is **0.0%**.
