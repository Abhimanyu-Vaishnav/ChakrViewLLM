# CHAKRVIEW v0.8 SINGLE-TASK JSON SFT DATASET CARD

## Summary
- **Dataset Identifier**: `chakrview_v08_single_task_json_sft`
- **Location**: `experiments/research_v08_json_sft/data/`
- **Total Records**: 200
- **Random Seed**: 42 (reproducible deterministic generation via `generate_dataset.py`)
- **Status**: FROZEN (Post-validation cryptographically locked)

---

## Split Allocation & Cryptographic Hashes

| Split | Filename | Record Count | JSON Tasks | Replay Tasks | SHA-256 Checksum |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Train** | `train.jsonl` | 140 | 100 | 40 | `666bab3401532f757bf73aad2edae9ebc7da85cf258edde135e9afa88a790b0a` |
| **Validation** | `val.jsonl` | 30 | 20 | 10 | `b7814fac8a65f1b36e8b2a17a322e0d0f071e42b15bfc72813500ad67e25f7f5` |
| **Test** | `test.jsonl` | 30 | 20 | 10 | `b28bd0332712462fad162b416597d5ec180896e5a408ce4a6d0cce0d60a9c890` |
| **Manifest** | `manifest.json` | - | - | - | `91ea2803b306b3bc4f1a26b2df4a5f4bb572f4f1ce2b9f93ee74f2617f1a3b3f` |

---

## Schema and Partitioning Guarantees

### 1. JSON Extraction Task
- **Fixed Schema**: `{"status": <string>, "code": <integer>}`
- **Disjoint Component Entities**:
  - Train: 100 subsystems (`subsystem_alpha_01-25`, `hydraulic_actuator_01-25`, `power_converter_01-25`, `thermal_sensor_01-25`)
  - Val: 20 subsystems (`propulsion_thruster_01-10`, `solar_inverter_01-10`)
  - Test: 20 subsystems (`cryo_pump_01-10`, `optical_relay_01-10`)
  - **Overlap**: 0% (asserted by automated validator).
- **Disjoint Status Strings**:
  - Train (20 values): `nominal`, `active`, `degraded`, `overheated`, `idle`, `restarting`, `calibrated`, `throttled`, `bypassed`, `discharged`, `synchronized`, `pressurized`, `saturated`, `drained`, `dampened`, `isolated`, `cooling`, `recharged`, `primed`, `unlatched`.
  - Val (4 values): `standby_normal`, `frequency_drift`, `voltage_spike`, `pressure_drop`.
  - Test (4 values): `standby_degraded`, `beam_misaligned`, `cavity_depressurized`, `flux_fluctuation`.
  - **Overlap**: 0% (asserted by automated validator).
- **Disjoint Numeric Code Ranges**:
  - Train: `[106, 696]` (sampled from `[100, 699]`)
  - Val: `[706, 796]` (sampled from `[700, 799]`)
  - Test: `[800, 898]` (sampled from `[800, 899]`)
  - **Overlap**: 0% (asserted by automated validator).

### 2. Continuation Replay Task
- **Domain**: General scientific laws and mathematical theorems.
- **Append EOS**: `append_eos=False` (continuation mode).
- **Replay Disjointness**: Train (40 distinct facts), Val (10 distinct facts), Test (10 distinct facts) share 0 text overlap.
- **Contamination Guard**: Zero mentions of telemetry, status keys, or JSON delimiters in replay facts.

---

## Tokenizer & Context Limit Audit
- Tokenizer: Canonical 4096-vocab BPE (`data/experiments/vocab_4096`).
- Maximum Observed Sequence Length: **134 tokens** (Record `TEST_REPLAY_009`).
- Model Context Window: 256 tokens (Capacity utilization: 52.3%, zero truncation risk).
