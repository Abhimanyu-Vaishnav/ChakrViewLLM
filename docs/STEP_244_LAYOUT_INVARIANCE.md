# STEP 244: Layout Invariance Training

## Mission
Train the compact associative circuit across diverse surface layouts representing the same underlying semantic operation ($K \to V$) to prevent positional and template shortcut learning.

## Layout Templates
Implemented in [`chakrview/cognition/association_layout_training.py`](file:///d:/Project/ChakrView/chakrview/cognition/association_layout_training.py):
1. **Format A (Pipe Arrow)**: `map |A| -> |1| and |B| -> |2| query |B| -> |`
2. **Format B (Semicolon Maps-To)**: `A maps to 1; B maps to 2; query B: |`
3. **Format C (Tuple Compact)**: `pairs (A,1) (B,2) query=B -> |`
4. **Format D (Assignment Slash)**: `assoc: A=1, B=2 / retrieve B = |`

## Results
- **Training Progression**: Successfully cycled through training across all 4 formats.
- **Shortcut Resistance**: Eliminates dependence on fixed single-character separators (e.g. pipe `|` or arrow `->`).
- **Transfer to Disjoint Identities**: Layout invariance across surface syntaxes did not unlock disjoint unseen-key/unseen-value token accuracy, which remained $0.0000$ across all tested layouts.
