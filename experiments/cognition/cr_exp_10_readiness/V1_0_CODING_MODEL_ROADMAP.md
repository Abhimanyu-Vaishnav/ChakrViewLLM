# v1.0 SPECIALIZED CODING MODEL ROADMAP
**Architecture Foundation:** ChakrView Indigenous Neural Model  
**Target Milestone:** v1.0 Coding Intelligence  
**Scope:** Strict Local PyTorch CPU / GGUF deployment without external LLM weights or APIs.

---

## 1. CURRENT GAP ANALYSIS (TRANSITIONING FROM CHAKRMICRO TO CHAKRCODE)

1. **Context Window Constraint**:  
   ChakrMicro v0.1 has a maximum context length of $128$ tokens. Coding tasks (function definitions, AST context, multi-line test fixtures) require at least $512$ to $1024$ tokens.
2. **Tokenizer Vocabulary Density**:  
   The current 4,096 ByteBPE vocabulary is tuned on conversational Hindi/English text. Code indentation (whitespace runs, tab indentation, brackets, camelCase/snake_case) suffers from severe token fragmentation ($>3.5$ tokens per code line).
3. **Parameter Representation Capacity**:  
   At 3.44M parameters ($d=192$), attention heads must balance conversational roles, state arithmetic, and logic. Coding syntax and scope tracking will require scaling to **ChakrSmall** ($\sim 15\text{M}-25\text{M}$ parameters, $d=384, l=12, h=8$).

---

## 2. 4-STAGE TECHNICAL ROADMAP

### Stage 1: Code-Specialized ByteBPE Tokenizer (Vocab 8,192)
- Re-train ByteBPE with preserved whitespace sequences (`    `, `        `) and programming syntax symbols (`->`, `==`, `!=`, `def `, `return `, `import `).
- Compress code tokenization density to $<1.8$ tokens per AST node.

### Stage 2: Architecture Evolution (ChakrMicro $\to$ ChakrCode v0.1)
- Context length expansion: $128 \to 512$ tokens using Rotary Position Embeddings (RoPE) instead of absolute positional embeddings.
- Capacity scaling: $d_{\text{model}}=384, n_{\text{layers}}=8, n_{\text{heads}}=8$ ($\sim 14.8\text{M}$ parameters).

### Stage 3: Syntactic Pretraining & Execution-Guided Objective
- Pretrain on high-quality indigenous Python and C algorithmic snippets.
- Execution-guided loss: Token weighting on function signature parameters, return statements, and test assertion values.

### Stage 4: Dedicated 5-Tier Coding Benchmark
- Tier 1: Syntax & Token Completion (HumanEval-Micro Python).
- Tier 2: Bug Localization (Identify line-level syntax/logic errors).
- Tier 3: Docstring-to-Function Generation.
- Tier 4: Test Assertion Prediction.
- Tier 5: GGUF Quantization & Packaging for CPU terminal execution.
