# ChakrView v0.8.0 Known Limitations & Epistemic Boundaries

**Model Tested**: `ChakrMicro` Step 10,000 Canonical Core (3,443,136 unique parameters, SHA-256 `96357705...e9f471ee`)  
**Architecture**: 6-layer Decoder-Only Transformer, Hidden Dim 192, MLP 512, Context 512, RoPE  
**Evaluation Scope**: Native Language Modeling, Exp18 Out-of-Distribution Scientific Prompts, Coding Benchmark  

---

## 1. Verified Native vs Tool-Assisted Capabilities

| Capability Area | Classification | Operational Mechanism | Observed Behavior |
| :--- | :---: | :--- | :--- |
| **In-Distribution Token Continuation** | **VERIFIED NATIVE** | Causal autoregressive token continuation | Predicts expected high-frequency tokens and short phrases within training corpus distribution. |
| **Simple Syntax / Keyword Modeling** | **PARTIAL NATIVE** | Pretrained BPE subword modeling | Generates basic syntactic markers (`def`, `return`, `assert`, punctuation) with ~50% syntactic validity. |
| **Complex Logic & Multi-Hop Reasoning** | **UNLEARNED** | Base pretraining core | Fails to maintain logical state across multiple reasoning steps; emits repetitive or degenerative phrases. |
| **Out-of-Distribution Scientific Reasoning** | **FAILED (20/20 Rejections)** | Exp18 Human Review | 100% rejection rate on biological/physical explanation queries. Does not possess scientific world knowledge. |
| **Novel Identifier Copying** | **UNLEARNED (in base)** | Causal attention alone | Cannot reliably copy out-of-vocabulary entities from prompt context without external retrieval or pointer mechanisms. |
| **Deterministic Fact Retrieval** | **TOOL-ASSISTED** | Runtime memory bridge & cognitive router | Structured lookup for defined terms (e.g. CPU) and relational memory slots. |
| **Sandboxed AST Code Execution** | **TOOL-ASSISTED** | Isolated subprocess runner with AST filter | Validates and runs user-provided Python assertions in safe isolation. |

---

## 2. Inherent Boundaries of the 3.44M Parameter Model

1. **Greedy Attractor Loops**:
   - The microscopic 3.44M core possesses low logit entropy outside its tightest training distributions. Under greedy decoding ($T=0$), completions exceeding 16–20 tokens frequently collapse into self-reinforcing n-gram attractor loops.
2. **Absence of Large-Scale Pretraining**:
   - Unlike multi-billion parameter foundation models trained on trillions of tokens, ChakrMicro was trained on a curated low-resource corpus (~5,534 Stage C documents). It cannot act as a general-purpose knowledge base or factual encyclopedia.
3. **Prompt Brittleness**:
   - Small variations in phrasing or punctuation can alter generation trajectory significantly. Generation requires deterministic bounds and maximum token constraints (`max_new_tokens <= 64`).

---

## 3. Quarantined Components & Preview Features

1. **Experimental Pointer Copying (`chakrview/runtime/pointer_mechanism.py`)**:
   - Quarantined in research workbenches. While pointer attention heads showed promise on synthetic single-token slot extraction tasks, forensic audits (Exp27–Exp29) revealed generalization degradation on non-template inputs. It is strictly disabled in canonical v0.8.0 runtime execution.
2. **GGUF Runtime Support**:
   - The repository provides GGUF inspection and export prototypes (`chakrview/tools/gguf_export.py`), but does not include native C++ engine execution (`llama.cpp`). Production inference is conducted via the pure PyTorch/Python CPU runtime adapter.
