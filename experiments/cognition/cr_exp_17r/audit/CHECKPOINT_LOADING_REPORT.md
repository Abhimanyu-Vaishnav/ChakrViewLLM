# CR-EXP-17R: Phase C — Independent Checkpoint Loading Report

## 1. Checkpoint Identification
- **Path:** `D:\Project\ChakrView\releases\v0.9.1_conversational_preview\chakrview_v091_model.pt`
- **File Size:** 13,796,677 bytes
- **SHA-256:** `f161ccdc501bca34f6ed2019252f2e51d18fa113ce0cc90187c4ff0cdd180d10`
- **Verification Status:** **PASS (Exact Bit Match)**

## 2. Checkpoint Metadata in Payload
- **Step:** 500
- **Best Step:** 500
- **Validation Loss:** 2.8516250282526014
- **Validation Token Accuracy:** 50.28%
- **Parent Checkpoint:** `D:\Project\ChakrView\experiments\cbr01_sft\runs\cbr_sft_04\checkpoints\sft_checkpoint_step_200.pt`
- **Parent SHA-256:** `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153`

## 3. Structural & Architectural Verification
- **Architecture:** `ChakrMicro v0.1`
- **Total Parameters:** 3,443,136 (100% match with canonical budget: 3,443,136)
- **Layer Count ($L$):** 6
- **Hidden Dimension ($d$):** 192
- **Attention Heads ($H$):** 6
- **MLP Dimension ($d_{\text{mlp}}$):** 512
- **Context Ceiling ($T$):** 512
- **State Dict Load Mode:** `strict=True`
- **Missing Keys:** 0
- **Unexpected Keys:** 0

## 4. Tokenizer Integration
- **Tokenizer Vocab Size:** 4096
- **Special Token Mapping:**
  - `BOS Token ID`: 0
  - `EOS Token ID`: 1

## 5. Execution Platform & Determinism
- **Device Tested:** Native Host CPU
- **Forward Determinism:** Verified (Max absolute logits discrepancy across consecutive forward passes = 0.000000).
- **External Dependencies:** Zero external API or pretrained backends. Self-contained indigenous architecture.
