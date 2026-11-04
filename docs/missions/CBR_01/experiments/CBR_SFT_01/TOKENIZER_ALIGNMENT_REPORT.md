# CBR-SFT-01 — TOKENIZER & PROMPT ALIGNMENT REPORT

---

## 1. Tokenizer Artifacts
- **Vocab File:** `data/experiments/vocab_4096/vocab.json` (SHA-256: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`)
- **Merges File:** `data/experiments/vocab_4096/merges.json` (SHA-256: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`)
- **Total Vocab Size:** 4,096 tokens.

---

## 2. Prompt Delimiter Policy Alignment
- In legacy runs, `strip_prompt_delimiters=True` called `.strip()` on prompt text. This stripped trailing spaces or newlines following delimiters (such as `'Assistant:'` vs `'Assistant:\n'`), causing prompt token representations to diverge from autoregressive inference prompts.
- In CBR-SFT-01, `strip_prompt_delimiters=False` is enforced.
- **Verification:** Tokenizing `'User: Hello\nAssistant:'` with `strip_prompt_delimiters=False` yields the identical subword sequence in both training collation and inference session prefill.

---

## 3. Special Token Alignment
- `bos_id`: 0 (`<|bos|>`)
- `eos_id`: 1 (`<|eos|>`)
- `pad_id`: 2 (`<|pad|>`)
- `ignore_index`: -100
- Prompt tokens are masked with `-100`.
- Response tokens and `eos_id` are unmasked, providing active gradient supervision strictly on assistant generation turns.

