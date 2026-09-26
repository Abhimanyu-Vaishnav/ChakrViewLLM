# Project Status: ChakrView

## Project Overview
- **Project**: ChakrView
- **Current phase**: Step 2 — Tokenizer Research & Specification
- **Status**: Tokenizer Specification Completed & Audited (Strict Technical Audit Applied; Awaiting Step 3 Approval)

---

## Status Summary

### Implementation Notice
> **IMPORTANT**: No model architecture code or tokenizer code has been written, and no neural weights, tokenizers, or pretrained components (such as Llama, Qwen, Mistral, Gemma, GPT, etc.) have been implemented or downloaded. ChakrView is being developed strictly incrementally from the ground up as an indigenous AI research initiative. Step 1 delivered the foundational neural core specification, and Step 2 has delivered the comprehensive tokenizer research and specification (with strict technical audit applied).

### Progress by Module
- `brain/`: Empty / Skeleton initialized (neural core specification complete; no code implemented yet)
- `data/`: Directory structure created (`raw`, `processed`, `validation`)
- `training/`: Directory structure created (optimization strategy specified; no code implemented yet)
- `inference/`: Directory structure created (runtime contracts specified; no code implemented yet)
- `memory/`: Directory structure created (decoupled; no memory system implemented)
- `skills/`: Directory structure created (decoupled; no skill systems implemented)
- `tools/`: Directory structure created (decoupled; no tools implemented)
- `runtime/`: Directory structure created (no runtime implemented)
- `evaluation/`: Directory structure created (evaluation criteria and test suites specified)
- `configs/`: Directory structure created (configurations defined in specifications)
- `tests/`: Directory structure created (test suite specifications defined, including adversarial tokenizer test cases)
- `checkpoints/`: Directory structure created (empty, no weights stored)
- `scripts/`: Directory structure created (no operational scripts yet)
- `docs/`: Active documentation initialized & audited
  - `docs/PROJECT_STATUS.md`
  - `docs/STEP_01_NEURAL_CORE_SPEC.md` (Step 1 Core Specification)
  - `docs/ARCHITECTURE_DECISIONS.md` (Step 1 Architecture Decision Records)
  - `docs/STEP_02_TOKENIZER_SPEC.md` (Step 2 Tokenizer Specification — Audited)
  - `docs/TOKENIZER_DECISIONS.md` (Step 2 Tokenizer Architecture Decision Records — Audited)

### Step 2 Technical Audit Summary
- **Core Invariant**: *"Tokenizer correctness takes precedence over compression."*
- **Paradigm**: BBPE is provisionally recommended; requires empirical validation before freezing implementation.
- **Claims Qualified**: Speed and fertility figures labeled strictly as hypotheses/planning targets, not measured ChakrView results.
- **Lossless Invariant**: `decode(encode(text)) == text` strictly required for all supported inputs; no irreversible normalization.
- **Unfrozen Numeric Strategy**: 3-way benchmark experiment specified (individual digits vs two-digit chunks vs normal BPE).
- **Candidate Vocabularies**: 2048, 4096, 8192, 16384 retained; final selection driven by 11 measured empirical metrics.
- **Adversarial Suite**: 15 explicit stress tests defined (Indic conjuncts, ZWJ/ZWNJ, emojis, mixed scripts, code indentation, paths, malformed sequences).
- **Frozen Decisions**: Byte-level fallback (256 primitives), no `<UNK>`, lossless round-trip requirement, causal tokenizer output, special-token IDs (`<BOS>`: 0, `<EOS>`: 1, `<PAD>`: 2).
- **Unfrozen Decisions**: Final vocabulary size, numeric strategy, exact grapheme pre-tokenization rules, corpus composition, merge ranking, tokenizer implementation strategy, optimization/trie implementation.

### Next Phase
- Awaiting Step 3 instructions from research leadership.

