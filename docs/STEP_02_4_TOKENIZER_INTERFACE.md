# ChakrView Step 2.4 — Tokenizer-to-Neural-Core Interface Contract

**Document Version**: 0.1.0  
**Phase**: Step 2.4 — Tokenizer Interface Specification & Neural Core Contract  
**Status**: RATIFIED & FROZEN FOR INTERFACE HANDOFF  
**Provisional Target**: $\text{VOCAB\_SIZE} = 4096, \quad d_{\text{model}} = 192, \quad T_{\text{max}} = 512$  

---

> ### SCIENTIFIC DISCLAIMER & STATUS NOTICE
> **$V = 4096$ is the current provisional tokenizer configuration selected from the Step 2.3 benchmark corpus.**  
> It remains replaceable if later neural-core experiments demonstrate that another vocabulary size (such as $V = 8192$) provides a better system-level tradeoff. The neural core interface is decoupled to ensure that altering $V$ requires zero structural rewrites of attention, MLP, or normalization layers.
>
> **Core Invariant**:  
> $$\text{Decode}(\text{Encode}(\text{text})) \equiv \text{text} \quad \forall \text{ supported inputs}$$
> *"Tokenizer correctness takes precedence over compression."*

---

## 1. Tokenizer API Contract

The tokenizer provides a deterministic, state-isolated interface for the neural runtime:

```python
# 1. Text-level encoding
encode(text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]

# 2. Text-level decoding
decode(tokens: List[int], skip_special_tokens: bool = True) -> str

# 3. Raw byte-level encoding
encode_bytes(data: bytes) -> List[int]

# 4. Raw byte-level decoding
decode_bytes(tokens: List[int], skip_special_tokens: bool = True) -> bytes
```

### Invariants & Expected Behaviors
| Scenario | Expected API Behavior |
| :--- | :--- |
| **Empty Input** | `encode("")` returns `[]`. If `add_bos=True, add_eos=True`, returns `[0, 1]`. `decode([])` returns `""`. |
| **Single Whitespace / Newline** | Faithfully mapped to foundational byte tokens (`" "` $\to$ `[35]`, `"\n"` $\to$ `[13]`). Never collapsed or stripped. |
| **Unicode Code Points** | Any valid Unicode string encodes via its exact UTF-8 byte stream into base or merged tokens. Reversible with zero normalization loss. |
| **Malformed Binary Bytes** | `encode_bytes(data)` processes arbitrary non-UTF-8 bytes (e.g. `0x80`, truncated prefixes) into byte primitives; `decode_bytes()` recovers exact raw binary. |
| **Literal Special Strings** | User text containing `"<BOS>"`, `"<EOS>"`, or `"<PAD>"` encodes to literal characters/bytes. NEVER produces special token IDs. |
| **Determinism** | For any input $x$, $\text{encode}(x) \equiv \text{encode}(x)$ identically across all runs. |
| **Type & Bounds Errors** | Non-string input to `encode` raises `TypeError`. Token IDs outside $[0, V - 1]$ passed to neural handoff raise `ValueError`. |

---

## 2. Token ID Partitioning Contract ($V = 4096$)

The vocabulary budget $[0, V - 1]$ is strictly partitioned to eliminate any collision between control tokens, byte primitives, and subwords:

$$\begin{aligned}
[0, 2] &\implies \mathbf{3\text{ Reserved Special Tokens}} \quad (\langle\text{BOS}\rangle=0, \langle\text{EOS}\rangle=1, \langle\text{PAD}\rangle=2) \\
[3, 258] &\implies \mathbf{256\text{ Foundational Byte Primitives}} \quad (\text{Token ID } = b + 3 \quad \forall b \in \{0x00 \dots 0xFF\}) \\
[259, 4095] &\implies \mathbf{3,837\text{ Learned BPE Subword Merges}} \quad (\text{Learned from ChakrView corpus})
\end{aligned}$$

- **Valid Token ID Range**: $\text{ID} \in [0, 4095]$.
- **Mathematical Total**: $3 + 256 + 3837 = 4,096$ unique token IDs.
- **Zero Collision Guarantee**: Special token IDs $0, 1, 2$ can never be produced by raw bytes or unconstrained text.

---

## 3. Special Tokens Contract

| Token | ID | Role in Causal Decoder | In v0.1? | Architectural Justification |
| :---: | :---: | :--- | :---: | :--- |
| `<BOS>` | **0** | Sequence start demarcation; conditions generation | **YES** | Mandatory for prompt boundary marking |
| `<EOS>` | **1** | Sequence end termination; stops autoregressive loop | **YES** | Mandatory to prevent runaway generation |
| `<PAD>` | **2** | Padding token for batch alignment | **YES** | Mandatory for batched CPU/GPU tensor execution |
| `<UNK>` | — | Unknown character fallback | **NO** | **Forbidden**: 256 byte primitives guarantee zero OOV |
| `<MASK>` | — | Masked language modeling | **NO** | Causal decoder-only model uses causal attention masking |
| `<SEP>` | — | Segment separator | **NO** | Redundant; delimited via prompt formatting |
| `<NL>` | — | Explicit newline token | **NO** | Raw byte primitive `0x0A` (ID 13) natively represents newlines |

---

## 4. Context Contract & Length Policy

- **Target Context Window for Chakr-Micro**: $\text{MAX\_CONTEXT} = 512$ tokens.
- **Critical Architectural Distinction**:
  - $T_{\text{max}} = 512$ is a **neural model constraint** (dictated by RoPE frequencies, attention matrix memory $O(T^2)$, and KV-cache budget).
  - The **tokenizer has no arbitrary sequence length limit** and will encode text of any length into an arbitrary-length integer sequence.
- **Runtime Policy for Long Sequences**:
  1. **Strict Truncation**: Cuts sequences at $T_{\text{max}}$ (optional append of `<EOS>`).
  2. **Sliding-Window Chunking**: Partitions long documents into overlapping segments (e.g., $512$ tokens with $64$-token stride) for document processing.
  3. **No Silent Data Loss**: The tokenizer must never silently drop or truncate text without explicit runtime invocation.

---

## 5. Neural Core Input Batch Contract

For batched forward passes on CPU and GPU, the handoff format is codified as:

```python
input_ids:      shape = [batch_size, sequence_length], dtype = int32 / int64
attention_mask: shape = [batch_size, sequence_length], dtype = int32 / int64
```

- In `attention_mask`:
  - `1`: Indicates an active, semantically valid token participating in causal attention.
  - `0`: Indicates `<PAD>` padding position, masked out in attention score computation.
- Padding is applied uniformly on the right (or left during autoregressive generation).

---

## 6. Embedding & Weight Tying Contract

In the Chakr-Micro architecture:
- **Embedding Table**: $E \in \mathbb{R}^{V \times d_{\text{model}}} = [4096, 192]$
- **Output Unembedding Projection**: $W_{\text{out}} = E^T \in \mathbb{R}^{d_{\text{model}} \times V} = [192, 4096]$

```
Token ID stream [batch, seq_len]
              │
              ▼
   Embedding Lookup E [4096, 192]
              │
              ▼
   Activation Tensor [batch, seq_len, 192]
              │
              ▼
   [Transformer Blocks + RoPE]
              │
              ▼
   Pre-Norm Output [batch, seq_len, 192]
              │
              ▼
   Output Projection W_out = E^T [192, 4096]
              │
              ▼
   Logits [batch, seq_len, 4096]
```

### Weight Tying Benefit
By tying $W_{\text{out}} = E^T$, the output head adds **0 additional unique parameters**.

---

## 7. Memory Accounting: Static Weight Storage vs. Runtime Memory

### Static Parameter Storage ($V = 4096, \quad d = 192$)
$$\text{Parameters} = 4096 \times 192 = 786,432 \text{ parameters} \quad (\approx 0.79\text{M})$$

| Precision | Bytes per Param | Embedding Table Storage | Tied Output Head Storage | Total Embedding Weight RAM |
| :---: | :---: | :---: | :---: | :---: |
| **FP32** | 4 bytes | 3.00 MB | 0 MB (shared) | **3.00 MB** |
| **FP16 / BF16** | 2 bytes | 1.50 MB | 0 MB (shared) | **1.50 MB** |
| **INT8** | 1 byte | 0.75 MB | 0 MB (shared) | **0.75 MB** |
| **INT4** | 0.5 bytes | 0.375 MB | 0 MB (shared) | **0.375 MB** |

> ### CRITICAL MEMORY BOUNDARY NOTICE
> The figures above represent **STATIC PARAMETER STORAGE ONLY**.  
> They explicitly **EXCLUDE**:
> - Dynamic forward activations ($B \times T \times d \times L$)
> - Runtime memory allocator overhead (malloc, PyTorch/NumPy slab fragmentation)
> - Temporary intermediate tensors (attention logits, QKV projections)
> - Key-Value (KV) cache for inference
> - Tokenizer trie lookup table memory (~898 KB)
> - Runtime framework engine footprint
>
> **CPU Cache Claim Rebuttal**: We do **NOT** claim the model is guaranteed to fit entirely inside CPU L3 cache. Cache-local execution is an empirical engineering hypothesis that must be measured on hardware.

---

## 8. Positional Information Contract (RoPE)

- The tokenizer emits pure categorical integer token IDs without positional tags.
- The embedding table maps token IDs to semantic vector representations in $\mathbb{R}^{d}$.
- Rotary Position Embedding (RoPE) is applied inside the transformer attention blocks to the Query and Key representations:
  $$\mathbf{q}_m = \mathbf{R}_{\Theta, m}^{d_k} \mathbf{W}_q \mathbf{x}_m, \quad \mathbf{k}_n = \mathbf{R}_{\Theta, n}^{d_k} \mathbf{W}_k \mathbf{x}_n$$
- The tokenizer has zero dependency on RoPE; positional dynamics are strictly handled within the neural core.

---

## 9. Serialization Contract (`Chakr-BPETokenizer-v0.1`)

The tokenizer artifact is serialized into a standalone directory (`data/tokenizer_experiments/v4096/`):

1. `metadata.json`: Configuration, version string (`Chakr-BPETokenizer-v0.1`), creation timestamp, tie-breaking rule, and corpus fingerprint.
2. `merges.json`: Ordered dictionary of merge rules `{"p0,p1": new_id}`.
3. `vocab.json`: Explicit dictionary mapping `token_id` to its hex-encoded byte representation.
4. `train_stats.json`: Training diagnostic metrics (time, initial bytes, final tokens, compression ratio).
5. `eval_stats.json`: Validation benchmark metrics (tokens/word, latencies, memory footprint).

*Rule: The training corpus is never embedded inside the tokenizer artifact.*

---

## 10. Neural Core Handoff Simulation

Validated via automated unit test:
```python
# Simulated Pipeline
tokens = tokenizer.encode("ChakrView indigenous neural architecture v0.1", add_bos=True, add_eos=True)
embeddings = fake_embedding_lookup(tokens, d_model=192)

# Output Tensor Verification
assert len(embeddings) == len(tokens)
assert len(embeddings[0]) == 192  # Exact d_model match
```

---

## 11. Test Suite Pass Verification

All 107 tests across 10 modules pass cleanly:
- 85 Step 2.2 prototype & adversarial tests
- 12 Step 2.3 corpus pipeline & trainer tests
- 10 Step 2.4 neural-core interface & handoff tests

```
============================= 107 passed in 0.19s =============================
```
