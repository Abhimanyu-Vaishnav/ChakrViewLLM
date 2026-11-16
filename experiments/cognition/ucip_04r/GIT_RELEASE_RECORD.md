# UCIP-04R: Git Release Record

**Mission ID:** UCIP-04R  
**Branch:** `exp/cbr-01-conversational-breakthrough`  
**Date:** 2026-11-16  

---

## 1. Summary of Changes

- Repaired `DirectNeuralRunner` in `scripts/run_m3_evaluations.py` to dynamically load model configuration from checkpoint metadata or tensor shapes before initializing `ChakrMicro(cfg)`, enabling seamless strict loading of both ChakrMicro (3.44M) and ChakrMedium (12.93M) architectures.
- Updated `ConversationalChatCLI.print_info()` in `chakrview/chat.py` to dynamically report the active model architecture, parameter count, layer count, and dimensions.
- Added automated unit test suite `tests/test_architecture_loader.py` covering strict loading, forward pass, dynamic `/info` verification, generation step, and unsupported architecture handling.
- Produced `releases/ucip_04_candidate/OWNER_MEDIUM_TEST_GUIDE.md` for owner testing.
- Preserved bit-for-bit checksum of `chakrview/runtime/session.py` (`9080ff51...`).

---

## 2. Checkpoint SHA-256 Hashes (Unmodified)

- `releases/ucip_04_candidate/chakrview_ucip04_model.pt`: `234d04d598b6ff2f6c4514d9b973c51233872c118ad97e8fb6e60dec79d68bd6`
- `releases/ucip_04_candidate/chakrview_ucip04_medium_model.pt`: `694c5888479bbd0a90d1ca116787c14e2e5362a6da0dfe782ea7304a55bd0693`
- `chakrview/runtime/session.py`: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`
