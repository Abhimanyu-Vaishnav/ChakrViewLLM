# ChakrView Exp04 — Dataset Specification

- **Benchmark Name**: Independent Pointer Robustness Challenge v1.0
- **Total Examples**: 120 (12 examples across 10 categories)
- **Benchmark SHA-256**: `11c0b4bfc33410e113c9c112f35e50c8a9b4446ff93fb930f0f837b0ee9d9748`
- **Abstention Ratio**: 52/120 (43.3%)
- **Alignment Method**: Fail-Closed Character-to-Byte-to-Token Adapter
- **Corrupted/Silent Fallbacks**: 0 (Every answerable span is verified by exact decoded reconstruction)

## Category Distribution

- `dynamic_entity_binding`: 12 examples (Field ID: 0)
- `turn_dependent_retrieval`: 12 examples (Field ID: 1)
- `anaphora_reference_resolution`: 12 examples (Field ID: 2)
- `superseded_instruction_adherence`: 12 examples (Field ID: 3)
- `user_correction_handling`: 12 examples (Field ID: 4)
- `distractor_resistance`: 12 examples (Field ID: 5)
- `ambiguity_clarification`: 12 examples (Field ID: 6)
- `missing_context_abstention`: 12 examples (Field ID: 7)
- `factual_uncertainty_handling`: 12 examples (Field ID: 8)
- `constrained_code_synthesis`: 12 examples (Field ID: 9)
