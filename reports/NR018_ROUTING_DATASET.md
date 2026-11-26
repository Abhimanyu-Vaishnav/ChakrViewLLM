# NR-018 Phase 3 & 4 — Neural Intent Router Design & Programmatic Dataset
**Mission ID:** CV-GENESIS-NR-018  
**Date:** 2026-11-26  
**Implementation:** `experiments/genesis_stage_h_018/generate_routing_data.py` (Seed: 2026)  
**Training Corpus (1,500 samples):** `data/stage_h_018/routing_train_1500.json` (SHA-256: `349d86279f427f1cf93122ffd7096186d45a856646942d04c281d560c97019bf`)  
**Validation Corpus (300 samples):** `data/stage_h_018/routing_val_300.json` (SHA-256: `56bc25ad964c0e8e06618c26eef034b0b8d98155176a1ba2a4e03ba7a2748ca8`)  
**Sealed Benchmark (500 tasks):** `data/stage_h_018/sealed_routing_benchmark_500.json` (SHA-256: `ee28cc42ca161d215ece303a3276a99190ac5e735e8ee30376f6a57e3196f1be`)

---

## 1. Intent Router Interface Specification
Instead of generating free-form Python code or calculation tokens, the Neural Intent Router emits a machine-readable JSON object:
```json
{
  "operation": "math_add",
  "arguments": {"a": 12, "b": 15},
  "status": "ready"
}
```
### Intent Status Handling:
- `"status": "ready"`: Operation and arguments extracted with complete confidence.
- `"status": "clarify"`: Ambiguous query where parameters are missing.
- `"status": "unsupported"`: Out-of-scope query or security injection attempt.

---

## 2. Dataset Composition (English, Hindi Devanagari, Hinglish)
- **Supported Operational Tasks:** Arithmetic (`add`, `sub`, `mul`, `div`), state accumulator updates, string mutations (`upper`, `reverse`), list operations (`reverse`, `max`), and conditional branching.
- **Linguistic Diversity:** Interleaved prompts across English (`Calculate sum of 12 and 15`), Hindi (`12 और 15 का योग निकालें`), and Hinglish (`12 aur 15 ko add karke batao`).
- **Disjoint OOD Splits:** In-distribution numerical range $[5, 50]$ vs OOD range $[60, 150]$; in-distribution variables (`x, y, a, b`) vs OOD Greek variables (`delta, gamma, omega`).
- **Robustness Classes:** Ambiguous prompts, unsupported real-world requests (flights, pizza), and adversarial code-injection prompts (`rm -rf /`, `subprocess.run`).
