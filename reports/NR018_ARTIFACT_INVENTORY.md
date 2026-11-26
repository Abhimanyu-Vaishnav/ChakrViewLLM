# NR-018: Artifact Inventory & Integrity Manifest

## Primary Model Checkpoints
- **Canonical Foundation (NR-011)**:
  - Path: `experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt`
  - SHA-256: `2c21632ebae4f0b05b9ff7091797ecc956ac0ad00ef6ec934381223bc273b269`
  - Role: Canonical base model for inference and initialization (unmodified).
- **NR-018 Neural Intent Router**:
  - Path: `experiments/genesis_stage_h_018/neural_router_17m.pt`
  - SHA-256: `f0be894bc1cc6af9322cc7b4aee3ae7a2cd62741d33874561b26fe8fc64ae488`
  - Role: Fine-tuned intent routing checkpoint for structured tool dispatch.

---

## Datasets & Benchmark Sets
1. **Routing Training Set (1,500 samples)**:
   - Path: `data/stage_h_018/routing_train_1500.json`
   - SHA-256: `349d86279f427f1cf93122ffd7096186d45a856646942d04c281d560c97019bf`
2. **Routing Validation Set (300 samples)**:
   - Path: `data/stage_h_018/routing_val_300.json`
   - SHA-256: `56bc25ad964c0e8e06618c26eef034b0b8d98155176a1ba2a4e03ba7a2748ca8`
3. **Sealed 500-Task Benchmark**:
   - Path: `data/stage_h_018/sealed_routing_benchmark_500.json`
   - SHA-256: `ee28cc42ca161d215ece303a3276a99190ac5e735e8ee30376f6a57e3196f1be`

---

## Benchmark & Audit Results Data
1. **System Comparison 500-Task Results**:
   - Path: `data/stage_h_018/system_comparison_results.json`
   - SHA-256: `1fadcc1c015b0c56707f224a39b37505526849483d3b1eea563c82cdf33aaafe`
2. **Security Audit Results (16 Attacks)**:
   - Path: `data/stage_h_018/security_audit_results.json`
   - SHA-256: `e5d51a00676915c84c53bc183ebbab27081f3916e49398fe261566f4626dc671`
3. **Generalization Analysis Summary**:
   - Path: `data/stage_h_018/generalization_analysis_summary.json`
   - SHA-256: `1757ce78aae16bdb3a7c5796f1c1c4e764e6b1d203282f32f9dc63d98c58459a`
4. **Performance & Hardware Profile**:
   - Path: `data/stage_h_018/performance_profile.json`
   - SHA-256: `a4a822e431281aed66cbc683c8b9d27c044318aac96daefeca03338465273906`

---

## Code Implementations
1. **Restricted Execution Engine**:
   - Path: `chakrview/runtime/restricted_executor.py`
2. **Dataset Generator**:
   - Path: `experiments/genesis_stage_h_018/generate_routing_data.py`
3. **System Comparison Runner**:
   - Path: `experiments/genesis_stage_h_018/run_system_comparison.py`
4. **Security Audit Suite**:
   - Path: `experiments/genesis_stage_h_018/test_security_audit.py`
5. **Generalization Analysis Script**:
   - Path: `experiments/genesis_stage_h_018/analyze_generalization.py`
6. **Performance Profiling Script**:
   - Path: `experiments/genesis_stage_h_018/profile_performance.py`
