# ChakrView Architecture & Neural Core Specification

## 1. Overview
ChakrView is an indigenous modular AI platform. The neural core, ChakrMicro, is an autoregressive causal decoder transformer designed for local inference and training without external API dependencies.

## 2. Transformer Topology
ChakrMicro v0.6 features:
- Hidden dimension: 384
- Attention heads: 6
- Feed-forward dimension: 1536
- Transformer layers: 6
- Vocabulary size: 4096 tokens
- Maximum context window: 512 tokens
- Parameter count: 14.8 million parameters

## 3. Tokenizer
ChakrView utilizes a custom BPE (Byte-Pair Encoding) tokenizer trained directly from raw domain corpora. The tokenizer model file is located at `models/tokenizer.json`. Special reserved tokens include `<unk>`, `<s>`, `</s>`, and `<pad>`.

## 4. Hardware and Runtime Adapters
ChakrMicro is engineered to execute across heterogeneous compute tiers:
- CPU Execution: Utilizes quantized int8 operations and PyTorch AVX2/AVX-512 vectorization.
- GPU Acceleration: Supports CUDA execution with automatic precision switching (FP16/BF16).
- Memory Ceiling: Minimum RAM required for CPU inference is 1.5 GB.
