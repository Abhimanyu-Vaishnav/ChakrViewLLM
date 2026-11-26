# NR-019: Artifact Inventory & Integrity Manifest

## 1. Canonical Foundation & Model Checkpoints
- **Canonical Foundation Checkpoint (NR-011)**:
  - Path: `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt`
  - SHA-256: `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269`
  - Role: Preserved foundation model for baseline comparisons (unmodified).
- **Neural Intent Router Checkpoint**:
  - Path: `experiments/genesis_stage_h_018/neural_router_17m.pt`
  - SHA-256: `f0be894bc1cc6af9322cc7b4aee3ae7a2cd62741d33874561b26fe8fc64ae488`
  - Role: Neural operation routing engine for Systems C, D, and E.

---

## 2. Generated Datasets & Benchmark Sets (NR-019)
1. **Training Set (3,000 samples)**:
   - Path: `data/stage_i_019/routing_train_3000.json`
   - SHA-256: `0bdecd4aa4dc0171f8953e691297d34441826576f399848b96c60d0b4646a4a3`
2. **Validation Set (500 samples)**:
   - Path: `data/stage_i_019/routing_val_500.json`
   - SHA-256: `84b2db544d61204c0ed031ddefb164dc6597666da09bbfbd34b30229d7f87f72`
3. **Sealed 800-Task Benchmark (0 duplicates)**:
   - Path: `data/stage_i_019/sealed_benchmark_800.json`
   - SHA-256: `4a62ba707af491cbfa6dda5cc756e6bfabba875ac25b30b3a5ad46654f222e6e`

---

## 3. Evaluation and Audit Result Datasets
1. **800-Task 5-System Benchmark Results**:
   - Path: `data/stage_i_019/benchmark_800_results.json`
   - SHA-256: `cfa6d6919fbb6716881388dce639b871dfabc5c2dc82dfd40c9d62012b1e46c7`
2. **Expanded Security Audit Results (20 vectors)**:
   - Path: `data/stage_i_019/security_audit_expanded.json`
   - SHA-256: `b13913e3cb08b3aa218560a23864f0c22300cb10abd498f6786365b6af7f31df`
3. **Latency & Conversational Regression Results**:
   - Path: `data/stage_i_019/performance_and_regression.json`
   - SHA-256: `a7c8956843b21642e997a2321bc8c1515ef56271d971c46b5ce73f4e7136260f`

---

## 4. Production Code Additions
1. **Multilingual Normalizer**:
   - Path: `chakrview/runtime/multilingual_normalizer.py`
2. **Exact Source-Span Grounding Engine**:
   - Path: `chakrview/runtime/source_grounding.py`
3. **5-System Benchmark Runner**:
   - Path: `experiments/genesis_stage_i_019/run_5system_benchmark.py`
4. **Expanded Security Test Suite**:
   - Path: `experiments/genesis_stage_i_019/test_expanded_security.py`
5. **Latency & Regression Script**:
   - Path: `experiments/genesis_stage_i_019/benchmark_latency_and_regression.py`
