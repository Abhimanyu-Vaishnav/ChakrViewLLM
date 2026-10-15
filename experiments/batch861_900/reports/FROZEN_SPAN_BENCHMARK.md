# WP 872–877: Frozen Span Benchmark & Dataset Lineage Audit

## 1. Frozen Benchmark Verification
- **Frozen Test Dataset**: `experiments/batch781_820/data/test_plan_dsl_frozen.jsonl`
- **Total Test Items**: 70 held-out scenarios
- **Verified SHA-256**: `cabf7daa3c6e453de1ad06e99c801fbba9b669f2130746a7b34bf022cd21ec2a`
- **Integrity Status**: Strictly immutable across Batches 781–900. Zero test items were modified or used in training.

---

## 2. Span Dataset Provenance
To train the neural span-grounding candidate (`ChakrSpanGroundingModel`), a leakage-resistant span-annotated curriculum dataset was constructed:
- **Generation Script**: [`experiments/batch861_900/build_span_dataset.py`](file:///d:/Project/ChakrView/experiments/batch861_900/build_span_dataset.py)
- **Train Split**: `experiments/batch861_900/data/train_span_grounding.jsonl` ($N=27$)
  - SHA-256: `85cec08cfcdb0171f04b79d2142ad5c6d17ce0cf0bf0742c4ec483d95c296c8d`
- **Validation Split**: `experiments/batch861_900/data/val_span_grounding.jsonl` ($N=10$)
  - SHA-256: `5557ea6a5b561cee39d98d17e0a342df9bed3972535533bb96790e76f1cb42f5`

---

## 3. Coverage Characteristics
1. **Multi-token BPE Numbers**: Includes operands like 600 (`['6', '00']`), 605 (`['6', '0', '5']`), 1024, 2048, and 2500.
2. **Text Phrase Strings**: Long multi-token entities with spaces and title capitalization (`'Decentralized Consensus Ledger Unique 6'`).
3. **Repeated Numbers**: Multiple distinct occurrences of identical digits (e.g. `14` as first operand vs `14` as second occurrence).
4. **Clarification Cases**: Prompts with missing required parameters explicitly annotated with start/end index -1 and type `NONE`.
