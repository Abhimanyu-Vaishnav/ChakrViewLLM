# CPU Vectorization and SIMD Dispatch

## 1. Instruction Sets
- Primary x86 vector extension: AVX2 with FMA3 instruction interleaving.
- Fallback x86 instruction set: SSE4.2 vector instructions.
- ARM vectorization: ARM Neon 128-bit SIMD registers.
- Thread pool worker ceiling for CPU execution: 4 physical worker threads.
