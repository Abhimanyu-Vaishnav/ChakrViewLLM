# ChakrView — Pre-Manual Baseline Verification Gate Report
**Audit & Gate Date**: October 21, 2026  
**Auditor**: Antigravity AI Engine (Independent Verification Gate)  
**Deliverables Directory**: `docs/preflight_verification_gate_2026_10/`  
**Execution Environment**: Microsoft Windows 11 Enterprise (64-bit), Python 3.14.7, PyTorch 2.14.0+cpu, 14 Threads (Pure CPU Execution).

---

## 1. Executive Summary & Gate Status

The objective of this gate is to independently verify the claims made in `docs/architecture_audit_2026_10/` against runnable code, automated tests, and a bounded CPU-only native smoke test of the canonical ChakrMicro checkpoint (Step-10000) and native Byte-Level BPE tokenizer.

### Gate Verdict:
**PREFLIGHT VERIFIED — READY FOR BOUNDED MANUAL TESTING OF CONTROLLER/POINTER; NATIVE MODEL UNABLE TO COMPLETE TASKS AUTONOMOUSLY**

The verification gate confirms:
1. **Infrastructure & Tokenizer Invariants (100% VERIFIED)**:
   - Canonical checkpoint, tokenizer vocabulary, merges, and runtime session hashes match known canonical values bit-for-bit.
   - BPE Tokenizer achieves 100% lossless UTF-8 roundtrip reconstruction across complex multilingual and code stress cases.
   - All 57 core unit tests (`test_brain_model.py`, `test_tokenizer_utf8_lossless.py`, `test_hardware_adaptive_engine.py`, and `test_exp12.py`) pass cleanly in 20.86s.
2. **Resource & Memory Bounds (100% VERIFIED)**:
   - Model loads into memory in **0.16 seconds**.
   - Peak process memory during inference was strictly bounded at **314.95 MB** (well below the 512 MB Tier 1 ceiling).
   - Generation terminates cleanly without unhandled exceptions or memory leaks.
3. **Native Model Autoregressive Capability (CRITICAL EMPIRICAL FINDING)**:
   - When evaluated purely natively (no pointer, no oracle, no schema controller, no regex rules, no retrieval), the canonical 3.4M ChakrMicro causal LM **completely fails** all 7 prompt probes (paraphrase, negation, instruction following, distractor resistance, multi-sentence context, uncertainty, and coding).
   - In 6 out of 7 probes, the model immediately collapses into repetitive synthetic attractor loops (`"Step-by-Step Solution:\nThe total number of the city of the city of the city..."` or `"# This is the number of the number of the number..."`).

---

## 2. Audit Claims Verification Matrix

| Audit Claim | Claimed in Document | Verification Status | Repository Evidence & Verification Result |
| :--- | :--- | :---: | :--- |
| **Tokenizer Lossless Invariant** | `CURRENT_STATE_INVENTORY.md` (Sec 2.1) | **VERIFIED** | `tests/test_tokenizer_utf8_lossless.py` passed 100% across all 33 test strings (Hindi, English, Code, Math, Emojis). Full 4096 vocab roundtrip verified. |
| **Exact Parameter Invariant (3,443,136)** | `CURRENT_STATE_INVENTORY.md` (Sec 2.1) | **VERIFIED** | Verified dynamically via PyTorch reflection: `sum(p.numel() for p in model.parameters()) == 3,443,136`. Weights tied identically (`embedding.weight is lm_head.weight`). |
| **Canonical Checkpoint Immutability** | `AUDIT_MANIFEST.json` | **VERIFIED** | SHA-256 matches `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` bit-identically. |
| **Protected Session Runtime Immutability** | `AUDIT_MANIFEST.json` | **VERIFIED** | `chakrview/runtime/session.py` SHA-256 matches `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`. |
| **Schema Controller 100% Syntax Guarantee** | `EXP12_REPORT.md` / `INVENTORY` | **VERIFIED** | `experiments/.../test_exp12.py` passed 100%. Schema FSM guarantees valid JSON and eliminates duplicate keys. |
| **Hardware Adaptive Engine Tiering** | `HARDWARE_ADAPTATION_PLAN.md` | **VERIFIED** | `tests/test_hardware_adaptive_engine.py` passed 100% (11/11 tests pass, verifying CPU-only fallback, memory limits, and cooperative yielding). |
| **Native LM General Language Understanding** | Historical doc claims (`PROJECT_STATUS.md`) | **NOT VERIFIED (DISPROVEN)** | Smoke test proves native model cannot follow simple instructions, answer questions, or generate Python code without external scaffolding. |
| **Multi-Node Distributed Federation** | `DISTRIBUTED_COMPUTING_PLAN.md` | **PARTIALLY VERIFIED** | Code runs on loopback sockets (`127.0.0.1`), but cross-device physical WAN/LAN execution remains unverified. |

---

## 3. Native Smoke Test: Prompts, Latency, and Raw Outputs

All prompts evaluated with greedy decoding ($T=0.0$), pure native PyTorch CPU forward pass, bounded to max 40 new tokens.

### Raw Per-Prompt Execution Log:

1. **Paraphrase Probe**:
   - *Prompt*: `"Rewrite the following sentence in simple words:\nThe atmospheric conditions precipitated an unexpected deluge."`
   - *Raw Output*: `"\n\nReferences"` (Terminated at token 1 / EOS immediately)
   - *Latency*: $0.0762\text{ s}$
   - *Result*: **FAIL** (Emitted empty/reference artifact instead of paraphrase)

2. **Negation Probe**:
   - *Prompt*: `"Context: The power converter is not active.\nQuestion: Is the power converter active?\nAnswer:"`
   - *Raw Output*: `"\nStep-by-Step Solution:\nThe total number of the city of the city of the city of the city of the city "`
   - *Latency*: $0.4939\text{ s}$
   - *Result*: **FAIL** (Collapsed into mathematical pretraining attractor loop; completely ignored context)

3. **Instruction Following Probe**:
   - *Prompt*: `"Instruction: Output only the single word SUCCESS and nothing else.\nOutput:"`
   - *Raw Output*: `"\n\nThe first of the first of the world of the world of the world of the world of the world of the worl"`
   - *Latency*: $0.4783\text{ s}$
   - *Result*: **FAIL** (Repeated ungrounded n-grams; zero instruction comprehension)

4. **Distractor Resistance Probe**:
   - *Prompt*: `"Context: Device A is offline. Device B is running. Device C is standby.\nQuestion: What is the status of Device B?\nAnswer:"`
   - *Raw Output*: `"\n\nStep-by-Step Solution:\nThe total of the city of the city of the city of the city of the show "`
   - *Latency*: $0.5534\text{ s}$
   - *Result*: **FAIL** (Repeated generic math step attractor loop)

5. **Multi-Sentence Context Probe**:
   - *Prompt*: `"Alice gave the blue key to Bob. Bob placed the blue key inside the metal cabinet.\nQuestion: Where is the blue key?\nAnswer:"`
   - *Raw Output*: `"\n\nStep-by-Step Solution:\nThe total number of the city of the city of the city of the city of the c"`
   - *Latency*: $0.5599\text{ s}$
   - *Result*: **FAIL** (Unable to perform 2-hop entity tracking)

6. **Uncertainty Probe**:
   - *Prompt*: `"Context: The turbine speed is 1500 RPM.\nQuestion: What is the internal temperature of the turbine?\nAnswer:"`
   - *Raw Output*: `"\n\nStep-by-Step Solution:\nThe total number of the city of the city of the centre of the city of "`
   - *Latency*: $0.5084\text{ s}$
   - *Result*: **FAIL** (Failed to express uncertainty or abstain; generated hallucinated loop)

7. **Coding Probe**:
   - *Prompt*: `"Write a Python function named \`is_even(n)\` that returns True if n is even, else False:\n\`\`\`python\n"`
   - *Raw Output*: `"\n    # This is the number of the number of the number of the number of the number of the number of the number of the number of the numb"`
   - *Latency*: $0.4880\text{ s}$
   - *Result*: **FAIL** (Looping comment line; zero syntactically valid Python code)

---

## 4. Hardware, Performance, and Memory Profiling

- **Host Operating System**: Microsoft Windows 11 Enterprise (Build 26200, 64-bit AMD64)
- **Host CPU**: 13th Gen Intel(R) Core(TM) i9-13900H (14 physical cores, 20 logical threads)
- **CPU Execution**: Pure CPU (Torch CUDA intentionally unavailable/bypassed)
- **Thread Allocation**: 14 active PyTorch CPU compute threads
- **Memory Footprint**:
  - Initial Python Process RSS: $228.35\text{ MB}$
  - Model & Tokenizer Loaded RSS: $304.24\text{ MB}$ ($\Delta = +75.89\text{ MB}$)
  - Peak Inference RSS: **$314.95\text{ MB}$** ($\Delta = +10.71\text{ MB}$ during active decoding)
  - Process Memory Ceiling: Maintained strictly within Tier 1 budget ($< 512\text{ MB}$)
- **Inference Latency**:
  - Time to first token + 40 tokens greedy: $\sim 0.48 - 0.55\text{ s}$ ($\sim 75 - 80\text{ tokens/sec}$ on CPU)

---

## 5. Artifact Integrity Reconciliation

Before and after the preflight gate execution, cryptographic SHA-256 checksums were verified:
- `experiments/batch16_21_combined/.../training_checkpoint_step_10000.pt`: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` [MATCH]
- `data/experiments/vocab_4096/vocab.json`: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a` [MATCH]
- `data/experiments/vocab_4096/merges.json`: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7` [MATCH]
- `chakrview/runtime/session.py`: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67` [MATCH]
- `experiments/research_v08_json_sft/exp11_hybrid_json/checkpoints/exp11_hybrid_best.pt`: `454fbd1559a686cc30fe5f661f8267a6507d861dccbcda3946987a0044c081e2` [MATCH]

Zero Git mutations, commits, merges, tags, or restores occurred.

---

## 6. Conclusions & Next Recommendation

1. **Which audit claims were verified**:
   - Tokenizer lossless bijection, exact model parameter accounting (3.4M), determinism, checkpoint integrity, hardware adaptive engine fallbacks, and schema controller guarantees were **100% verified**.
2. **Which claims still need stronger evidence**:
   - Multi-node distributed federation under real physical network packet loss.
   - Low-RAM edge execution under active OS swapping ($<2\text{GB}$ total system RAM).
3. **Is the project ready for bounded manual testing**:
   - **YES, but strictly for the Constrained Hybrid / Schema-Controlled Pointer pipeline (Arms C, D, E)**.
   - **NO for unassisted native causal LM generation**. Testing human users against the unassisted native causal LM will yield only repetitive n-gram attractor loops.
4. **The most important native capability failure**:
   - **Autoregressive Attractor Collapse**: The native causal LM lacks instruction-following representations. Once it encounters a novel prompt format, it collapses into repetitive n-grams learned from its pretraining corpus (e.g. `Step-by-Step Solution: The total number of the city of...`).
5. **One recommended next experiment**:
   - **Milestone 1 / Experiment 13: Field-Conditioned Pointer Attention**.
   - Rather than relying on the causal LM to generate free-form answers or unconditioned queries, train explicit slot-conditioned query tokens (`[QUERY_ENTITY]`, `[QUERY_STATUS]`, `[QUERY_CODE]`) into the pointer head with the backbone frozen. This directly targets the slot entanglement failure identified in Experiment 12.
