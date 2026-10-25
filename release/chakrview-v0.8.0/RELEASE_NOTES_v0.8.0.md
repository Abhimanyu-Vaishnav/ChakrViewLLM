# ChakrView v0.8.0 Release Notes — Independent Neural Language Model Research Preview

**Version**: `v0.8.0`  
**Classification**: Empirical AI Research Preview & Reproducible Baseline  
**Architecture**: `ChakrMicro` (3,443,136 unique parameters, tied embeddings, 6 layers, 6 heads, width 192, MLP 512, RoPE)  
**Target Hardware**: 100% CPU-first, low-resource edge architectures (ARM/x86_64)  
**Governance & Release Status**: Verified Canonical Research Baseline  

---

## 1. Executive Summary

ChakrView is an indigenous, from-scratch neural language model built entirely without external pretrained weights, closed API wrappers, or hosted inference services.

**ChakrView v0.8.0** delivers a fully audited, reproducible research baseline centered on the canonical **Step 10,000 checkpoint** (`training_checkpoint_step_10000.pt`, 42,212,787 bytes). This release establishes complete epistemic transparency: every capability claim is strictly bounded by reproducible evaluation, and all negative experimental findings are documented without concealment.

Key milestones delivered in v0.8.0:
1. **Canonical Checkpoint & Artifact Verification**: Bit-exact SHA-256 verification of the 42.2 MB canonical model checkpoint, vocabulary (4,096 tokens), merge table, and protected runtime components.
2. **Corpus Provenance & Licensing Closure**: Document-by-document audit reconciling all 5,534 Stage C training documents with open licenses (MIT, CC-BY-SA 4.0, PSF-2.0, CC-BY 4.0, GODL-India) and verified shard checksums.
3. **Rigorous Checkpoint Privacy Audit**: Zero private keys, access tokens, personal identifiers, or developer host paths embedded in model weights or training history.
4. **Transparent Epistemic Boundary (Exp18 Rejections)**: Human evaluation of 20 novel scientific / biological explanation prompts resulted in 20/20 rejections (100% failure rate) for high-level reasoning, proving that the microscopic 3.44M core cannot synthesize ungrounded scientific explanations.
5. **Experimental Pointer Quarantine**: Experimental pointer copying heads and speculative attention enhancements remain strictly quarantined in research benches and are excluded from production inference.
6. **GGUF Status Classification**: Native GGUF inference remains an active prototype/preview; native execution is powered directly via the pure Python/PyTorch runtime engine.

---

## 2. Canonical Artifact Specifications

| Asset | Path | Size | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| **Canonical Checkpoint** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | 42,212,787 bytes | `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` |
| **Tokenizer Vocabulary** | `data/experiments/vocab_4096/vocab.json` | 112,627 bytes | `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` |
| **Tokenizer Merges** | `data/experiments/vocab_4096/merges.json` | 76,299 bytes | `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` |
| **Protected Model Core** | `chakrview/brain/model.py` | 7,162 bytes | `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361` |
| **Protected Session Core** | `chakrview/runtime/session.py` | 12,511 bytes | `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` |

### Parameter Accounting
- **Total Unique Parameters**: `3,443,136` float32 parameters.
- **Tied Embeddings**: Input token embeddings and output projection head share weights (`4096 * 192 = 786,432` parameters counted once).
- **Transformer Configuration**: 6 decoder layers, hidden dimension 192, feed-forward dimension 512, 6 attention heads (head dimension 32), sequence length 512, Rotary Position Embeddings (RoPE).

---

## 3. Human Evaluation & Boundary Audit (Exp18 Rejections)

In comprehensive qualitative testing of the 3.44M parameter model against 20 out-of-distribution scientific prompts (Exp18), **all 20 outputs were rejected by human reviewers**:
- **Findings**: The model produced repetitive loops, hallucinatory definitions, or syntactically plausible but factually incorrect explanations.
- **Root Cause**: A 3.44M parameter model trained on low-resource corpora develops localized n-gram and syntactic correlations, but does not acquire deep conceptual or multi-hop relational world knowledge.
- **Epistemic Principle**: ChakrView does not claim "broad general reasoning" or "emergent intelligence." Native generation is strictly bounded to in-distribution token continuation and deterministic runtime queries.

---

## 4. Experimental Architecture Governance

1. **Neural Pointer Copying Quarantined**:
   - Neural pointer and copy-augmented attention heads developed across research batches (Exp22–Exp30) demonstrated localized synthetic utility on fixed-template extraction, but degraded out-of-vocabulary robustness.
   - Pointer networks are **excluded from the canonical v0.8.0 inference pipeline** and remain quarantined in `chakrview/runtime/pointer_mechanism.py` and research benches pending fair compatibility baselines.
2. **GGUF Native Inference Status**:
   - Structural GGUF inspection tools and format converters exist in the codebase, but native C/C++ GGUF inference runtimes (e.g. `llama.cpp` integration) are classified as **Experimental Preview / Prototype**.
   - Production inference relies on the validated CPU Python runtime adapter.

---

## 5. Licensing & Attribution Notice

ChakrView source code is licensed under the Apache License 2.0.  
Training corpus provenance is documented and attributed in `release/chakrview-v0.8.0/DATASET_PROVENANCE_AND_LICENSES.md`:
- **GSM8K**: MIT License
- **Simple English Wikipedia**: CC-BY-SA 4.0
- **Hindi Wikipedia**: CC-BY-SA 4.0
- **Sanskrit Wikipedia**: CC-BY-SA 4.0
- **TheAlgorithms Python**: MIT License
- **CPython Documentation**: Python Software Foundation License (PSF-2.0)
- **OpenStax High School Biology**: CC-BY 4.0
- **L3Cube Marathi/Hindi Corpus**: CC-BY 4.0
- **Open Government Data (OGD) India**: Government Open Data License (GODL-India)
