# ChakrView Normal Chat Gate v0.1 -- Reproducibility Specification

## 1. Environment and Hardware
- **OS**: Windows (AMD64)
- **Python Version**: Python 3.14.7 (tags/v3.14.7:823f032)
- **PyTorch Version**: 2.14.0+cpu (CPU execution strictly enforced)
- **Repository Root**: D:\Project\ChakrView
- **Active Branch**: dev/v0.6-v0.7-validation
- **Commit HEAD**: 30136a5dc02934180a503fdf0a4c287469bf106

---

## 2. Model and Checkpoint Artifacts
- **Model Architecture**: ChakrMicro (ModelConfig(), 3,443,136 parameters)
- **Canonical Architecture Fresh-Init Parameter Hash**: c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da  
  *(Note: This is the SHA-256 parameter digest of an untrained model instantiated with torch.manual_seed(42), NOT a source-code file hash).*
- **Canonical Model Source File (chakrview/brain/model.py) SHA-256**: 1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361
- **Canonical Checkpoint Path**: experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt
- **Canonical Checkpoint SHA-256**: 96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee
- **Tokenizer Artifacts Directory**: data/experiments/vocab_4096/
  - ocab.json SHA-256: db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a
  - merges.json SHA-256: da4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7

---

## 3. Decoding and Inference Settings
- **Inference Harness**: chakrview.runtime.neural_runtime_adapter.NeuralRuntimeAdapter
- **Decoding Mode**: greedy (	emperature: 0.0)
- **Max New Tokens**: 48 tokens
- **Repetition Penalty**: 1.0 (unpenalized raw model baseline)
- **Top-K**: 0, **Top-P**: 1.0
- **Stop Tokens**: Default contract stop tokens (EOS_ID: 1)
- **Stop on EOS**: Enabled in adapter verification
