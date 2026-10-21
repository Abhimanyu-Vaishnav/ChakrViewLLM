# ChakrView Experiment 11 — Hybrid Generator-Pointer Design Specification

## 1. Mission and Research Objective

ChakrView Experiment 11 builds an isolated research prototype combining ChakrView's native causal language model output head with the span-pointer mechanism developed in Experiment 10.

The mission is to generate valid structured JSON syntax autoregressively via the causal language model, while precisely extracting and copying requested values (including unseen multi-token entity names) from the input prompt via the span-pointer.

---

## 2. Architectural Blueprint: The Hybrid Generator-Pointer

### 2.1 The Core Duality
1. **Generator Role**: Causal language modeling head ($W_{\text{vocab}} \in \mathbb{R}^{d \times V}$). Emits structural JSON syntax: `{"`, `key`, `": "`, `", "`, `"}`.
2. **Pointer Role**: Span pointer head ($W_{\text{start}}, W_{\text{end}} \in \mathbb{R}^{d \times d}$). Emits contiguous context spans for target slot values: $[s_k, e_k]$.

### 2.2 Explicit Routing Mechanism
To ensure full inspectability, determinism, and avoid unverified gating heuristics, the hybrid defines an explicit routing contract:
- The generator emits tokens step-by-step.
- When generating JSON values inside string quotes following a target key (e.g. `{"component": "` or `{"status": "`), control routes to the pointer head.
- The pointer head selects the span $[s, e]$ from the candidate positions in the prompt context using candidate masking.
- The multi-token span tokens are emitted, followed by closing quotes `"` and continuation of structural JSON synthesis.
- Alternatively, in end-to-end token routing mode: the generator emits a special trigger or value delimiter that invokes the pointer to fill the slot.

---

## 3. Synthetic Benchmark Specification

### 3.1 Schema
A fixed, standardized telemetry / system observation JSON schema:
```json
{
  "entity": "<extracted_value>",
  "status": "<extracted_value>",
  "code": "<extracted_code_or_integer>"
}
```

### 3.2 Key Challenges Tested
1. **Single-field vs Multi-field extraction**: Instructions request 1, 2, or 3 fields.
2. **Single-token vs Multi-token entities**: Values range from 1 token (e.g. `832`) to 4 tokens (e.g. `cryo_pump_01`, `beam_misaligned`).
3. **Punctuation-adjacent values**: Values surrounded by quotes, brackets, or colons.
4. **Distractor values**: Irrelevant key-value pairs in context.
5. **Repeated / ambiguous values**: Distractor values that share subwords with the target.
6. **Missing requested fields**: If a requested field is not present in context, emit `"null"` or `"none"` per documented policy.
7. **Familiar vs Held-Out Entities**:
   - Split into Train, Validation, Test A (Familiar values, novel contexts), and Test B (Completely held-out values never seen in training).

---

## 4. Evaluation and Baselines

Three baselines evaluated on identical test examples:
1. **Baseline A**: Native Causal LM (`ChakrMicroForCausalLM` without pointer).
2. **Baseline B**: Exp 10 Span Pointer (evaluated on copy fields only).
3. **Baseline C**: Hybrid Generator-Pointer Prototype.
4. **Control**: Deterministic lookup parser (for sanity check).

---

## 5. Provenance Hashes & Safety Commitments
- Parent checkpoint (`Step-10000`): `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- Tokenizer vocab: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
- Tokenizer merges: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`
- Session runtime: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`
- Work strictly confined to `experiments/research_v08_json_sft/exp11_hybrid_json/`.
