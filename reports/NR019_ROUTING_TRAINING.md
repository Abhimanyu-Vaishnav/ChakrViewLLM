# NR-019: Balanced Routing Dataset & Training Specification

## 1. Dataset Generation Methodology
All datasets were generated via deterministic local procedural rules with full isolation and strict reproducibility:
- **Seed**: `19042`
- **Output Files**:
  - `data/stage_i_019/routing_train_3000.json` (3,000 samples)
  - `data/stage_i_019/routing_val_500.json` (500 samples)
  - `data/stage_i_019/sealed_benchmark_800.json` (800 unique tasks)

---

## 2. Benchmark Composition (800 Sealed Tasks, 0 Duplicates)
- **English Subset (200 tasks)**:
  - 100 in-distribution, 100 out-of-distribution parameter regimes.
  - Diverse syntax structures across 20 operation families.
- **Hindi Subset (200 tasks)**:
  - 100 in-distribution, 100 out-of-distribution.
  - Multi-format numerals: Devanagari numerals (`१५`), Devanagari number words (`पंद्रह`), and ASCII numerals.
- **Hinglish Subset (200 tasks)**:
  - 100 in-distribution, 100 out-of-distribution.
  - Transliterated Hindi number words (`pandrah`, `nau`, `bees`) and mixed phrasing.
- **Compositional, Ambiguous & Adversarial Subset (200 tasks)**:
  - 100 adversarial/unsupported injection probes (`os.system`, `eval`, arbitrary queries).
  - 100 multi-step state accumulators and conditional logic branches.

---

## 3. Grounded Architecture Comparison
In Phase 6, five architectures will be evaluated on the identical 800-task sealed benchmark:
1. **System A**: Pure Neural Baseline (ChakrGenesis-17M Stage-D direct generation).
2. **System B**: Deterministic Rule Parser + Restricted Executor (NR-018 regex baseline).
3. **System C**: Neural Intent Router (NR-018 free-form JSON generation).
4. **System D**: Neural Router with Exact Source-Span Grounding + Normalization.
5. **System E**: Cascaded Hybrid Router (Deterministic Parser fast-path $\to$ Neural Source Grounding fallback).
