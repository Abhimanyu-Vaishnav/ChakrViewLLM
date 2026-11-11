# CR-EXP-16 Stage A Leakage Audit
## Verification of Benchmark Isolation and Zero-Leakage Guarantee

**Mission Identifier:** CR-EXP-16  
**Audited Datasets:** `stage_a_train.json` (1,000 samples) and `stage_a_val.json` (150 samples)  
**Evaluated Against:** `cr16_synthetic_benchmark_1000.json` (1,000 probes, SHA-256: `0b61e73d57d903f34dc789736ecd9957c5633c1018fc8c8264bd1e1cb73cac97`)  

---

## 1. Audit Methodology

The automated audit script (`audit_stage_a_leakage.py`) scanned all training and validation sequences for:
1. Exact sequence matches against all 1,000 benchmark targets.
2. Token-level overlap with reserved benchmark entities (`Dev`, `Roy`, `Prem`, `Mukesh`, `Siddharth`, etc.).
3. Value overlap with reserved benchmark cities (`Delhi`, `Jaipur`, `Pune`, etc.).
4. Substring contamination in prompt headers.

---

## 2. Audit Findings

| Audit Check | Permitted Threshold | Detected Overlap | Status |
|---|---|---|---|
| **Exact Target Sequence Overlap (Train vs Bench)** | 0 | **0** | **PASS** |
| **Exact Target Sequence Overlap (Val vs Bench)** | 0 | **0** | **PASS** |
| **Reserved Benchmark Entity Leakage** | 0 | **0** | **PASS** |
| **Reserved Benchmark Location Leakage** | 0 | **0** | **PASS** |

**Audit Determination:** **ZERO LEAKAGE CONFIRMED.** The training and validation sets contain no test sequences, entities, or cities.

