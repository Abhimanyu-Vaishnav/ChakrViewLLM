# Batch 94: Synthesis & Scientific Audit

## 1. Architectural & Reproducibility Invariants
- **Canonical Model Architecture**: 3,443,136 unique parameters ($d_{model}=192$, $n_{heads}=6$, $d_{ff}=512$, $L=6$, $V=4096$, $max\_seq\_len=512$). Untouched and verified.
- **Canonical Fresh Init SHA-256**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`.
- **Canonical Step-10,000 SHA-256**: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`.
- **Zero Modification to Untracked Work**: `chakrview/runtime/session.py` left strictly untouched.

## 2. Scientific Synthesis: Tool-Calling vs Native Arithmetic
Throughout Batches 63 to 90, native arithmetic training on `ChakrMicro` faced severe representational bottlenecks:
1. **Direct Arithmetic**: Collapsed to global constants (e.g. `|277`, `6`, or unbalanced distributions).
2. **Scratchpad Generation**: Reduced collapse but required multi-token chain reasoning beyond 3.4M parameter capacity, resulting in digit hallucination.
3. **Structured Tool-Calling (Batches 91–94)**:
   - **Syntax & Schema Generalization**: 100% successful. The model learned to formulate valid, executable JSON payloads with exact tool names and operations without runtime patching.
   - **Dynamic Argument Binding**: Remained subject to mode collapse (`{"a": 1, "b": 1}`). The model treated the tool call as an invariant static subroutine invocation rather than binding dynamic operand slots from the context window.
   - **Conclusion**: A 3.4M parameter model can effortlessly master formal tool protocol syntax and deterministic software delegation workflows, but reliably routing dynamic numerical arguments from natural language prompts requires richer contextual attention capacity or targeted prefix-slot alignment.

## 3. Product Synthesis: Local Workbench Evolution
The Local Workbench now provides:
- Live Training Job Monitoring with real-time canvas loss curves (Batches 87–90).
- Side-by-side Checkpoint Output Comparison with unified diff calculation and deterministic oracle reference outputs (Batches 91–94).
- Offline JSON report persistence and instant re-inspection without GPU/CPU recomputation.
