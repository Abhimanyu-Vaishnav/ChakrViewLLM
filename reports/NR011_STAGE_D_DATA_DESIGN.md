# CV-GENESIS-NR-011 Stage-D Data Design Specification

## 1. Executive Summary

Mission **CV-GENESIS-NR-011 Phase 2 & 3** engineered the **Stage-D Multilingual SFT Corpus**, expanding the training data foundation for the `ChakrGenesis-17M` model to **5,513 unique, quality-controlled examples** (an expansion over NR-008's 4,401 examples and NR-005's 2,901 examples).

---

## 2. Dataset Schema & Invariant Enforcement

Every Stage-D record conforms to a strict JSONL schema:
```json
{
  "id": "String (Unique identifier)",
  "category": "String (Task domain: PYTHON, HIN, EXPL, QA, INST, ARITH, etc.)",
  "language": "String (en, hi, hi-en, sa)",
  "prompt": "String (Canonical 'User: <input>\\nAssistant:' format)",
  "response": "String (Supervised target completion ending in EOS)",
  "source": "String (Curated source identifier)",
  "token_count": "Integer (Total BPE tokens <= 240)"
}
```

---

## 3. Split Accounting & Linguistic Distribution

- **Total Unique Samples:** **5,513**
- **Train Split (85%):** **4,686 examples**
  - English (`en`): 2,799 (59.7%)
  - Devanagari Hindi (`hi`): 1,423 (30.4%)
  - Hinglish (`hinglish` & `hi-en`): 373 (7.9%)
  - Sanskrit (`sa`): 91 (1.9%)
- **Validation Split (10%):** **551 examples**
- **Test Split (5%):** **276 examples**

### Domain Representation:
- Python Code & Software Engineering: 12.5%
- Technical & Scientific Explanations: 18.3%
- Structured Reasoning & Arithmetic: 18.3%
- Conversational Dialogue & Social Interaction: 24.5%
- Multilingual Factual QA: 26.4%
