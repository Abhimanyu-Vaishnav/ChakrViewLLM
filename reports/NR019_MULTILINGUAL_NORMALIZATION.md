# NR-019: Multilingual Normalization Layer

## 1. Scope & Linguistic Coverage
The multilingual normalizer (`chakrview/runtime/multilingual_normalizer.py`) unifies numeral representations across three core linguistic modes:
1. **English**: ASCII digits (`15`), English number words (`fifteen`, `twenty-five`).
2. **Hindi (Devanagari)**: Devanagari numerals (`०, १, २, ३, ४, ५, ६, ७, ८, ९`), Devanagari number words (`शून्य, एक, दो, ..., पंद्रह, सौ`).
3. **Hinglish (Romanized Hindi)**: Transliterated words (`ek, do, teen, char, paanch, das, gyarah, barah, pandrah, bees, pachaas`).

---

## 2. Tokenizer Fragmentation Forensics
Investigation into the canonical 4,096-vocabulary BPE tokenizer (`data/experiments/vocab_4096`):
- English word `"Multiply"` $\to$ 3 tokens.
- ASCII `"15"` $\to$ 1 token.
- Devanagari numerals `"१५"` $\to$ **4 raw byte tokens** (`\xe0\xa5`, `\xa7`, `\xe0\xa5`, `\xab`).
- Devanagari numeral sequence `"१२३४५"` $\to$ **10 raw byte tokens**.

Because each Devanagari digit spans 3 UTF-8 bytes and was partially unmerged during 4,096 BPE vocabulary construction, generating Devanagari numerals auto-regressively requires 2 to 4 tokens per digit. This quadrupled the probability of token dropout and generation corruption.

---

## 3. Normalization Strategy
By implementing source-span grounding prior to execution:
1. Text is preserved in full fidelity without lossy ASCII replacement.
2. Numeral substrings are normalized into standard integers and floats with zero tokenization overhead.
3. Decoupling extraction from neural token generation completely bypasses BPE fragmentation.
