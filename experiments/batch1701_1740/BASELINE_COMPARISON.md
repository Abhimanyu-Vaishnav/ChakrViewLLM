# BASELINE COMPARISON & MULTI-SYSTEM ROUTING MATRIX
## Batches 1701–1740 Comprehensive Evaluation on 180 Frozen Held-Out Tasks

### 1. Benchmark Setup & Evaluation Scope

Batches 1701–1740 evaluated whether replacing autoregressive JSON generation with a **Direct Neural Classification Head** allows ChakrMicro to learn useful semantic intent classification and structured argument routing.

Evaluated on the exact same frozen, balanced held-out benchmark (180 tasks, 30 per intent class, SHA-256: `63bc6ca2...`):
1. **Majority Class Baseline:** Predicts the most conservative default class (`OUT_OF_SCOPE`).
2. **Governed Deterministic Router:** Production regex-based routing with fail-closed verification.
3. **Keyword Heuristic Baseline:** Lexical substring matching with rule-based argument capture.
4. **Variant A (Frozen Backbone + Linear Head):** Canonical checkpoint weights frozen, training only 1,350 classification head parameters.
5. **Variant B (Fine-Tuned Backbone + Linear Head):** Full end-to-end gradient updates across all 3.44M parameters on 100% data budget.
6. **Governed Neuro-Symbolic Hybrid:** Routes confident neural classifications to tools, falling back to deterministic router for ungrounded or low-confidence queries.

---

### 2. Multi-System Performance Matrix

| Evaluation Dimension | Majority Baseline | Governed Deterministic | Keyword Heuristic | Variant A (Frozen Backbone) | Variant B (Fine-Tuned 100%) | Governed Hybrid |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Intent Accuracy** | 16.67% (30/180) | 25.56% (46/180) | 53.89% (97/180) | 65.56% (118/180) | **77.78%** (140/180) | 48.33% (87/180) |
| **Argument Accuracy** | 16.67% (30/180) | 18.33% (33/180) | 47.78% (86/180) | 58.89% (106/180) | **69.44%** (125/180) | 47.22% (85/180) |
| **Macro-F1 (6 Classes)**| 0.0476 | 0.1903 | 0.5073 | 0.6398 | **0.7772** | 0.4422 |
| **JSON / Format Validity** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **Unsafe Execution Rate**| **0.0%** (0/180) | 0.56% (1/180) | 20.56% (37/180) | 6.11% (11/180) | 3.89% (7/180) | **0.56%** (1/180) |
| **Average Latency (CPU)**| **0.002 ms** | **0.051 ms** | **0.009 ms** | 3.896 ms | 4.557 ms | 4.142 ms |
| **Throughput (req/sec)**| >500,000 | ~20,000 | >100,000 | ~256 | ~220 | ~240 |

---

### 3. Key Scientific Conclusions

1. **The Direct Classification Head Breakthrough:**
   - In Batch 1661–1700, the autoregressive neural candidate scored **19.17% intent accuracy** with an **85.8% JSON syntax error rate** and a latency of **259 ms**.
   - By eliminating token-by-token generation and reading directly from the Transformer sequence representation, **Variant B achieved 77.78% intent accuracy (Macro-F1: 0.7772)** with **100% output validity** in **4.55 ms** (~57x faster).
2. **Frozen vs. Fine-Tuned Backbone:**
   - **Variant A (Frozen)** achieved **65.56% intent accuracy** without touching a single backbone weight. This proves that the pre-existing 3.4M ChakrMicro representations already encode substantial semantic intent separability.
   - **Variant B (Fine-Tuned)** achieved an additional **+12.22% accuracy boost** (reaching 77.78%), demonstrating that task-specific gradient tuning aligns attention heads toward task distinctions.
3. **Outperforming Deterministic and Keyword Baselines:**
   - Variant B clearly outperformed the Keyword Baseline (77.78% vs 53.89%, $+23.89\%$) and the Governed Deterministic Router (77.78% vs 25.56%, $+52.22\%$) on conversational paraphrases.
