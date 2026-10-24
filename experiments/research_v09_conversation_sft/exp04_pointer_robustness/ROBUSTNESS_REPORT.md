# ChakrView Exp04 — Independent Pointer Robustness Report

**Status**: `ROBUSTNESS THRESHOLD NOT MET`
**Date**: 2026-10-22
**Benchmark SHA-256**: `11c0b4bfc33410e113c9c112f35e50c8a9b4446ff93fb930f0f837b0ee9d9748`

## 1. Executive Summary

- Evaluated **120 independent conversational examples** across 10 categories.
- Overall Exact-Span Accuracy: **42/120 (35.00%)**.
- Pre-registered threshold ($\ge 75.0\%$): **FAILED**.\n\n## 2. Category Performance Breakdown

| Category | Field ID | Total | Correct | Accuracy | Abstentions Correct |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `dynamic_entity_binding` | - | 12 | 2 | 16.7% | 2/2 |
| `turn_dependent_retrieval` | - | 12 | 0 | 0.0% | 0/2 |
| `anaphora_reference_resolution` | - | 12 | 8 | 66.7% | 8/8 |
| `superseded_instruction_adherence` | - | 12 | 2 | 16.7% | 0/2 |
| `user_correction_handling` | - | 12 | 2 | 16.7% | 2/2 |
| `distractor_resistance` | - | 12 | 0 | 0.0% | 0/2 |
| `ambiguity_clarification` | - | 12 | 2 | 16.7% | 2/8 |
| `missing_context_abstention` | - | 12 | 12 | 100.0% | 12/12 |
| `factual_uncertainty_handling` | - | 12 | 12 | 100.0% | 12/12 |
| `constrained_code_synthesis` | - | 12 | 2 | 16.7% | 0/2 |

## 3. Abstention Analysis

- True Positives: 38
- False Positives: 28
- False Negatives: 14
- True Negatives: 40
- Abstention Precision: **57.58%**
- Abstention Recall: **73.08%**

## 4. Diagnostics by Context Length

- `short (<100)`: 36/57 (63.2%)
- `medium (100-200)`: 6/63 (9.5%)
- `long (>200)`: 0/0 (0.0%)
