# ChakrView v0.7.0 Known Limitations & Capability Boundaries

**Model Tested**: `ChakrMicro` Step 10,000 Canonical Core (3,443,136 unique parameters, SHA `96357705...e9f471ee`)  
**Evaluation Scope**: 6-Category Sandboxed Python Coding Benchmark  

---

## 1. Verified Native vs Tool-Assisted Capabilities

| Capability Area | Classification | Operational Mechanism | Observed Behavior |
| :--- | :---: | :--- | :--- |
| **Simple Assert Completion** | **VERIFIED NATIVE** | Causal autoregressive token continuation | Correctly emits assert statement for arithmetic target (`assert add(2, 3) == 5`) |
| **Python Token Syntax** | **PARTIAL NATIVE** | Pretrained BPE subword modeling | 50% of prompt completions produce valid Python AST nodes |
| **Complex Logic / Algorithms** | **UNLEARNED** | Requires multi-token state tracking | Emits repetitive token sequences (`# It is the result of the number...`) |
| **Bug Identification** | **UNLEARNED** | Base model lacks instruction/diff alignment | Fails to emit replacement blocks; requires SFT tuning |
| **Sandboxed AST Execution** | **TOOL-ASSISTED** | Python `ast` + isolated `subprocess.run` | 100% reliable execution and safety filtering |
| **Memory Isolation** | **TOOL-ASSISTED** | In-memory `ConversationStore` | Clean session resetting without weight drift |

---

## 2. Real Limitations of the 3.44M Parameter Base Core

1. **Greedy Attractor Collapse**:
   - The microscopic 3.44M core suffers from diffuse logits that collapse into self-reinforcing attractor loops during long completions.
   - For code generation tasks requiring $> 16$ tokens, completions degrade into repetitive phrases unless bounded.

2. **Absence of Specialized Code SFT**:
   - The base model was trained on general bilingual text, not source code repositories.
   - It recognizes basic Python syntax tokens (`def`, `return`, `assert`, `class`), but lacks relational grounding to construct multi-line control structures.

3. **No Dynamic Copy Mechanism**:
   - As documented in the copy-capability forensics, causal attention alone without pointer/copy mechanisms cannot reliably copy novel variable names or identifiers from prompt context.

---

## 3. Road to v0.8 & v1.0 Coding Specialization

- **v0.8.0**: Implementation of copy-augmented attention objectives and structured code SFT datasets built via `CodingDataPipeline`.
- **v0.9.0**: Integration hardening and LoRA/adapter pretraining on synthetic code corpora.
- **v1.0.0**: Public coding-specialized model release with full GGUF export compatibility.
