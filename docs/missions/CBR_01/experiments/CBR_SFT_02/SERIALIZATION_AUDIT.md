# CBR-SFT-02 — SERIALIZATION & TOKENIZER VERIFICATION REPORT

---

## 1. Frozen Tokenizer Specification
- Custom Byte-Pair Encoding (BPE) with UTF-8 byte fallback.
- **Vocabulary Size:** 4,096 tokens (Locked).
- **Special Tokens:**
  - `bos_id`: 0 (`<|bos|>`)
  - `eos_id`: 1 (`<|eos|>`)
  - `pad_id`: 2 (`<|pad|>`)
  - `ignore_index`: -100 (loss ignore label)

---

## 2. Delimiter Contract & Scaffolding Invariant
In CBR-SFT-02, 100% of training, validation, and evaluation records adhere to bit-exact chat turn scaffolding:
- **Single Turn:**
  ```
  User: {user_prompt}\nAssistant:
  ```
  Response string begins with a space delimiter (` {assistant_response}`) and terminates with `[EOS]`.
- **Multi-Turn Context:**
  ```
  User: {turn_1_user}\nAssistant: {turn_1_asst}\nUser: {turn_2_user}\nAssistant:
  ```
- **Serialization Equivalence:**
  Because `strip_prompt_delimiters=False` is enforced, the token IDs generated during training sequence encoding (`[BOS] + p_toks + r_toks`) are bit-for-bit identical to the tokens generated during autoregressive inference prefill (`session.generate(prompt)`).
- **Masking Verifiability:**
  All tokens up to and including `'Assistant:'` receive target `-100`. Gradients are applied strictly to the assistant answer tokens and the trailing `[EOS]` token.

