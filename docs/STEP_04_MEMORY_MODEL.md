# ChakrView — Step 4.1: Theoretical & Empirical Memory Model

**Document Version**: 1.0.0  
**Phase**: Step 4.1 — Architecture Verification & Baseline  
**Model Target**: ChakrMicro v0.1 ($3,443,136$ parameters, $V=4096, d_{\text{model}}=192, N=6, H=6, d_{\text{ff}}=512, T_{\text{max}}=512$)  
**Status**: ARCHITECTURE FREEZE MEMORY SPECIFICATION  

---

## 1. Static Parameter Memory Footprint

ChakrMicro v0.1 has exactly **$3,443,136$ unique trainable parameters**. Weight tying between the token embedding table and the output language modeling head ensures zero duplicated parameters.

| Precision | Bits / Param | Bytes / Param | Exact Weight Memory (Bytes) | Exact Weight Memory (MiB) | Exact Weight Memory (MB) |
|---|:---:|:---:|:---:|:---:|:---:|
| **FP32 (Single)** | 32 | 4.0 | $13,772,544\text{ B}$ | **$13.1345\text{ MiB}$** | $13.7725\text{ MB}$ |
| **FP16 / BF16 (Half)** | 16 | 2.0 | $6,886,272\text{ B}$ | **$6.5673\text{ MiB}$** | $6.8863\text{ MB}$ |
| **INT8 (Quantized)** | 8 | 1.0 | $3,443,136\text{ B}$ | **$3.2836\text{ MiB}$** | $3.4431\text{ MB}$ |
| **INT4 (Ultra-compact)** | 4 | 0.5 | $1,721,568\text{ B}$ | **$1.6418\text{ MiB}$** | $1.7216\text{ MB}$ |

### Breakdown by Architectural Component (FP32):
- **Token Embedding Table** ($4096 \times 192$): $3,145,728\text{ B}$ ($3.00\text{ MiB}$)
- **Attention Projections** ($6\text{ layers} \times 4 \times 192 \times 192$): $3,538,944\text{ B}$ ($3.375\text{ MiB}$)
- **SwiGLU FFN Projections** ($6\text{ layers} \times 3 \times 192 \times 512$): $7,077,888\text{ B}$ ($6.75\text{ MiB}$)
- **RMSNorm Scales** ($13 \times 192$): $9,984\text{ B}$ ($0.0095\text{ MiB}$)
- **Output LM Head**: **$0\text{ B}$** (Tied storage pointer with embedding)

---

## 2. Activation Memory Analysis

Activation memory is dynamic and scales with batch size $B$ and context length $T$.

### 2.1 Per-Layer Activation Tensors (Inference Workspace)
In inference mode without backpropagation, activations do not need to be retained across layers. Memory can be reused across layers using an execution ping-pong workspace.

For batch $B=1$ and context $T=512$:
1. **Residual Stream State**: $[B, T, d_{\text{model}}] = [1, 512, 192] = 98,304\text{ floats} = 384\text{ KiB}$
2. **Q, K, V Projections**: $3 \times [B, H, T, d_{\text{head}}] = 3 \times [1, 6, 512, 32] = 294,912\text{ floats} = 1.125\text{ MiB}$
3. **Attention Score Matrix**: $[B, H, T, T] = [1, 6, 512, 512] = 1,572,864\text{ floats} = 6.00\text{ MiB}$
4. **SwiGLU Gate & Up Projections**: $2 \times [B, T, d_{\text{ff}}] = 2 \times [1, 512, 512] = 524,288\text{ floats} = 2.00\text{ MiB}$
5. **Output Vocabulary Logits**: $[B, T, V] = [1, 512, 4096] = 2,097,152\text{ floats} = 8.00\text{ MiB}$

> [!NOTE]
> In autoregressive generation ($T=1$), the attention score matrix shrinks to $[1, 6, 1, T_{\text{past}}]$, requiring less than $12\text{ KiB}$ of temporary activation workspace!

### 2.2 Training Activation Footprint (Full Backpropagation)
During training, intermediate activations across all $N=6$ layers must be stored in memory for the backward pass (unless activation checkpointing is used):
- **Stored Activations per Token ($B=1, T=512$)**: $\approx 35\text{ - }45\text{ MiB}$ at FP32.

---

## 3. KV Cache Footprint (Autoregressive Generation)

During autoregressive generation, keys and values for preceding tokens are cached across all $N=6$ layers to avoid $O(T^2)$ recomputation.

### Formula:
$$\text{KV Elements} = 2 \times N_{\text{layers}} \times H_{kv} \times d_{\text{head}} \times B \times T$$
$$\text{KV Elements} = 2 \times 6 \times 6 \times 32 \times B \times T = 2,304 \times B \times T \text{ elements}$$

| Context Length ($T$) | Batch Size ($B$) | Elements | FP32 Footprint (4B) | FP16 Footprint (2B) | INT8 Footprint (1B) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$T = 16$** | 1 | 36,864 | $0.1406\text{ MiB}$ | $0.0703\text{ MiB}$ | $0.0352\text{ MiB}$ |
| **$T = 64$** | 1 | 147,456 | $0.5625\text{ MiB}$ | $0.2813\text{ MiB}$ | $0.1406\text{ MiB}$ |
| **$T = 128$** | 1 | 294,912 | $1.1250\text{ MiB}$ | $0.5625\text{ MiB}$ | $0.2813\text{ MiB}$ |
| **$T = 256$** | 1 | 589,824 | $2.2500\text{ MiB}$ | $1.1250\text{ MiB}$ | $0.5625\text{ MiB}$ |
| **$T = 512$** | 1 | 1,179,648 | **$4.5000\text{ MiB}$** | **$2.2500\text{ MiB}$** | **$1.1250\text{ MiB}$** |
| **$T = 512$** | 2 | 2,359,296 | **$9.0000\text{ MiB}$** | **$4.5000\text{ MiB}$** | **$2.2500\text{ MiB}$** |

---

## 4. Runtime Overhead & Total Working Set

### 4.1 PyTorch Python Runtime (Development Baseline)
When executing inside PyTorch on Windows:
- Python interpreter & standard runtime: $\approx 25\text{ - }35\text{ MB}$
- PyTorch dynamic library bindings (torch, MKL, oneDNN): $\approx 150\text{ - }180\text{ MB}$
- Model weights (FP32): $13.13\text{ MiB}$
- Peak working set during forward evaluation: **$240\text{ - }280\text{ MB}$**

### 4.2 Embedded Standalone C/C++ Runtime (Step 6 Target)
In a pure C/C++ standalone deployment without Python or heavy framework overhead:
- Total static memory = Weights + KV Cache + Ping-Pong Activation Workspace
- **FP32 Full-Context ($T=512, B=1$)**:
  - Weights: $13.13\text{ MiB}$
  - KV Cache: $4.50\text{ MiB}$
  - Workspace: $\approx 8.50\text{ MiB}$
  - **Total Embedded Memory**: **$\approx 26.13\text{ MiB}$**
- **INT8 Full-Context ($T=512, B=1$)**:
  - Weights: $3.28\text{ MiB}$
  - KV Cache: $1.13\text{ MiB}$
  - Workspace: $\approx 2.50\text{ MiB}$
  - **Total Embedded Memory**: **$\approx 6.91\text{ MiB}$**

---

## 5. Architectural Myth Busting: CPU Cache Residency

> [!WARNING]
> **Engineering Rule**: Do NOT claim that the entire model "fits inside CPU cache" based solely on weight size.

### Why Weight Size Alone Does Not Guarantee Cache Residency:
1. **Hierarchical Slicing**:
   - Modern CPUs have small L1 data caches ($32\text{ - }48\text{ KB}$ per core) and moderate L2 caches ($1\text{ - }2\text{ MB}$ per core).
   - Only the shared L3 cache ($16\text{ - }36\text{ MB}$ on modern desktop CPUs, but often $\le 2\text{ - }4\text{ MB}$ on low-spec edge CPUs) is large enough to hold weights.
2. **Active Working Set Invalidation**:
   - The total operational working set at step $t$ includes:
     $$\text{Working Set} = \text{Current Layer Weights} + \text{Input Activation} + \text{Output Activation} + \text{KV Cache} + \text{Instruction Cache}$$
   - Matrix multiplications stream through weights row-by-row or tile-by-tile.
3. **Low-Spec CPU Targets**:
   - Older 28 nm / 32 nm architectures (e.g. AMD Jaguar, older Intel Celerons, or ARM Cortex-A53) frequently feature $\le 2\text{ MB}$ total L2/L3 cache. On those systems, memory access is strictly DDR bandwidth-bound.

---

## 6. Summary Decision

The memory model confirms that **ChakrMicro v0.1** satisfies all architectural resource constraints:
- Static FP32 weights require only **$13.13\text{ MiB}$**.
- KV Cache at full $512$ context requires only **$4.50\text{ MiB}$**.
- The model is primed for low-spec CPU deployment and ready for future INT8 quantization ($< 7\text{ MB}$ total working set).
