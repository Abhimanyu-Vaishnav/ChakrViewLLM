# Neural Representation Audit — Batches 1141–1180
## Provenance, Architecture & Empirical Capability of ChakrBiEncoder

### 1. Model Identity and Lineage
- **Model Checkpoint**: `experiments/batch171_180_pointer_learning/checkpoints/improved_bi_encoder.pt`
- **Checkpoint SHA-256**: `d55d8272f2d93e5066a34c8a2bda0728c89c8913ffad48ec436a5faeb30d43e2`
- **Architecture**: `ChakrBiEncoder`
  - Vocabulary Size: 4,096 tokens (indigenous Byte-Pair Encoding from `data/experiments/vocab_4096`).
  - Model Dimension (`d_model`): 128
  - Projection Embedding Dimension (`d_embed`): 64
  - Number of Transformer Encoder Layers: 2
  - Attention Heads: 4
  - Maximum Sequence Length: 128
  - Pooling: Mean pooling over non-padded tokens, followed by linear projection and L2 unit normalization.

### 2. Training Data and Objective Provenance
- **Training Script**: `experiments/batch171_180_pointer_learning/run_retrieval_improvement.py`
- **Training Dataset**: `experiments/batch161_170_neural_learning/data/split_manifest.json`
- **Dataset Size**: **Only 16 total examples** across all splits:
  - Training split: 8 examples (atomic numbers of Hydrogen, Helium, Lithium; moons of Mars, Jupiter; boiling points of Water, Ethanol; distance of Moon).
  - Validation split: 4 examples.
  - Test split: 4 examples.
- **Training Loss**: In-batch InfoNCE cross-entropy loss over cosine similarity ($T=0.1$) trained for 120 epochs on CPU.

### 3. Critical Findings on Benchmark Overlap & Generalization
1. **Zero Domain Overlap with Knowledge Assistant Corpus**:
   - The bi-encoder was trained solely on 8 basic physical science facts (chemical elements and planetary moons).
   - It was **never trained** on enterprise cloud specifications, network SLAs, billing terms, disaster recovery SOPs, or encryption protocols.
2. **What the Bi-Encoder Learned**:
   - The model learns to map co-occurring character 3-grams and frequent BPE tokens to distinct vectors in 64-dimensional Euclidean space.
   - It exhibits non-trivial cosine similarity on related terms (e.g. 0.4428 on `"network traffic"` vs `"in-transit communication"`).
3. **Why It Does Not Outperform BM25 on Domain Corpus**:
   - On the 52-document / 116-chunk corpus:
     - BM25 Pure Lexical: **Recall@1 = 93.33%**, **Recall@3 = 98.33%**, **MRR = 0.9583**.
     - Learned Hybrid: **Recall@1 = 91.67%**, **Recall@3 = 100.00%**, **MRR = 0.9556**.
   - While the neural hybrid improves Recall@3 to 100%, its top-1 retrieval rank is slightly lower than BM25 on technical questions where exact keywords (`"1536"`, `"AVX2"`, `"P1"`) provide unambiguous discrimination.
4. **Scientific Conclusion**:
   - The `ChakrBiEncoder` is a functioning indigenous neural bi-encoder, but cannot be claimed as an advanced semantic reasoner.
   - Its capability on technical documentation is that of a continuous subword representation space, not human-level semantic comprehension.
