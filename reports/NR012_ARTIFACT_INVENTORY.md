# CHAKRVIEW RESEARCH REPORT: NR-012 ARTIFACT INVENTORY
**Mission ID:** CV-GENESIS-NR-012  
**Date:** 2026-11-25  

---

## 1. Newly Created Runtime & Application Components
1. `chakrview/runtime/inference_17m.py`: Production streaming 17M inference engine with incremental subword decoding.
2. `chakrview/runtime/conversation_session.py`: Multi-turn dialogue manager with session export and mode switching.
3. `chakrview/brain/cognitive_architecture.py`: Enhanced working memory with fact revision, abstention gate, and deterministic math.
4. `chakrview/chat_cli.py`: Interactive command-line chat application (`/mode`, `/memory`, `/clear`, `/export`).
5. `chakrview/ui/app.py`: Standalone dark-themed local web console (standard-library HTTP server, zero external framework dependencies).
6. `tests/test_cognitive_architecture_nr012.py`: Unit test suite verifying fact revision, math routing, and abstention.

---

## 2. Experimental Data & Quantization Assets
1. `experiments/quantization_012/run_quantization_benchmarks.py`: Benchmark runner for FP32, FP16, and dynamic INT8.
2. `experiments/quantization_012/genesis_17m_fp16.pt`: Half-precision checkpoint (32.6 MB).
3. `experiments/quantization_012/genesis_17m_int8_dynamic.pt`: Quantized INT8 weights (22.4 MB).
4. `experiments/quantization_012/quantization_benchmark_results.json`: Full benchmark metrics.
5. `data/sealed_eval_012/sealed_benchmark_150.json`: 150-item sealed benchmark dataset.
6. `experiments/sealed_eval_012/run_sealed_eval_150.py`: Independent evaluation engine.
7. `experiments/sealed_eval_012/benchmark_comparison_summary.json`: Summary comparison across 3 configurations.
8. `experiments/sealed_eval_012/benchmark_comparison_details.json`: Comprehensive turn-by-turn evaluation log.

---

## 3. Cryptographic Hashes of Key Checkpoints
- `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt`: `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269`
- `experiments/genesis_pretraining_010/sft_checkpoints/genesis_17m_sft_best.pt`: `ac62c43fda2e33acb3a24cbdad1038d104d8f2a8653ac39bd6c081606aac56dd`
- `data/experiments/vocab_4096/vocab.json`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
