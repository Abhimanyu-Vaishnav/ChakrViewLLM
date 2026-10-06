# STEP 241: Generalized Randomized Binding Episode Generator

## Mission
Build a reusable, fully randomized contextual episode generator for key-value associations without fixed mappings, template shortcuts, or data contamination.

## Architecture & Specification
Implemented in [`chakrview/cognition/generalized_binding_episodes.py`](file:///d:/Project/ChakrView/chakrview/cognition/generalized_binding_episodes.py):
- **Dynamic Pool Sampling**:
  - Training Keys Pool: `["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]`
  - Training Values Pool: `["1", "2", "3", "4", "5", "6", "7", "8", "9"]`
  - Strictly Disjoint Evaluation Keys: `["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]`
  - Strictly Disjoint Evaluation Values: `["0", "#", "@", "$", "%", "&", "!", "?"]`
- **Association Counts**: Configurable support for 1, 2, 3, 5, and 8 association pairs per episode.
- **Syntactic Formats**: Supports 4 distinct surface layouts representing identical semantic operations ($K \to V$):
  - `format_a`: Canonical `map |A| -> |1| and |B| -> |2| query |B| -> |`
  - `format_b`: Semicolon natural `A maps to 1; B maps to 2; query B: |`
  - `format_c`: Tuple compact `pairs (A,1) (B,2) query=B -> |`
  - `format_d`: Assignment `assoc: A=1, B=2 / retrieve B = |`
- **Distractor Noise**: Configurable inclusion of distractor key-value pairs or irrelevant syntax noise.
- **Independent Hashing & Anti-Contamination**: Each episode generates an independent deterministic SHA-256 hash. `verify_no_contamination()` verifies that zero overlap exists between train and evaluation episode hashes or key identity sets.

## Verification
- Verified 1, 2, 3, 5, 8 association generation.
- Verified zero hash and key overlap between training and disjoint evaluation splits.
- Preserved strict candidate isolation; neural model receives prompt tokens only, no metadata leaks.
