# ChakrView Capability Registry — Independent Diagnosis (CR-EXP-05)

**Audit Date**: 2026-11-06  
**Candidate Checkpoint**: `conversational_pilot_best_val.pt` (SHA-256: `ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd`)  
**Parent Checkpoint**: `sft_checkpoint_step_200.pt` (SHA-256: `327118ce7057af868d38c64ce3df35dba3b85b820b8022e18cb5be1f86cfa153`)  
**Canonical Baseline**: `ChakrMicro` (SHA-256: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`)  

---

## 1. Verified Capability Status Matrix

| Capability ID | Domain / Function | Status | Evidence & Diagnostic Findings |
|---|---|---|---|
| **CAP-01** | **Deterministic CPU-First Runtime** | **PROVEN** | 3.44M parameter architecture executes cleanly with bit-exact reproducibility and 0 CUDA dependencies. |
| **CAP-02** | **Persona Identity & Self-Introduction** | **PROVEN** | Generates `"I am ChakrView, an indigenous neural language model developed from first principles."` reliably (100% on direct persona probes). |
| **CAP-03** | **Polite Conversational Openers & Closers** | **PROVEN** | Greets users (`"Good morning!"`, `"Good afternoon!"`) and adheres to turn-ending boundaries without infinite repetition loops. |
| **CAP-04** | **Held-Out Cross-Entropy Reduction** | **PROVEN** | Achieves 41.78% cross-entropy reduction on in-distribution held-out validation (Parent: 5.2353 $\to$ Candidate: 3.0480, $p < 10^{-18}$). Confirmed robust across all 26 template clusters ($t = 10.99$). |
| **CAP-05** | **Anti-Attractor Resistance** | **PROVEN** | Completely eliminates earlier `'Cube'` and `'Mumbai'` mode collapses (0 occurrences across 100 tests). |
| **CAP-06** | **Out-of-Distribution Entity Extraction** | **DISPROVEN** | 0% success (0/8 on unseen entity retrieval). Emits unrelated attractors (`'Celsia'`) or generic assistant phrases. |
| **CAP-07** | **Variable Binding & Relational Assignment** | **DISPROVEN** | Fails all forward, reverse, and dynamic update variable-binding synthetic probes (0/8). Target entity is consistently ranked $>30$ in next-token distribution. |
| **CAP-08** | **Context-Position Robustness** | **DISPROVEN** | Fails retrieval regardless of fact placement (Beginning, Middle, End) or context length (35 to 153 tokens). |
| **CAP-09** | **Multi-Hop & State Update Tracking** | **DISPROVEN** | 0% success on 2-hop compositional queries, dynamic state updates, and long-range follow-ups (0/6). |
| **CAP-10** | **Arithmetic & Quantitative Reasoning** | **DISPROVEN** | 0% accuracy (0/10) on elementary addition, subtraction, and comparison tasks. |
| **CAP-11** | **Multilingual Generalization** | **DISPROVEN** | 0% accuracy (0/10) on Devanagari Hindi and Hinglish due to severe subword fragmentation and lack of merged Hindi tokens. |

---

## 2. Invariants & Capabilities That Must Be Preserved in Future Work
1. **Persona Integrity**: The exact wording of the self-introduction must not be degraded by future regularizers.
2. **Turn Boundary Discipline**: Stop token handling and clean generation termination without repetition loops.
3. **CPU Compatibility**: Maximum sequence length 512, d_model=192, 6 layers, CPU-only latency $<1.5$s per turn.
4. **Frozen Baseline Immutability**: $\Delta W = 0$ on canonical baseline checkpoints.

