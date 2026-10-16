# Neural Capability Evaluation Specification (Batches 1621–1660)

## 1. Mission Intent & Strategic Framing
The objective of Batches 1621–1660 is to directly investigate, evaluate, and challenge the **indigenous ChakrMicro neural core** itself—without relying on deterministic tools, AST parsers, BM25 knowledge retrieval, regex answer extractors, or external pretrained model weights.

Previous milestones successfully created production-grade governance layers around ChakrView:
- Deterministic SafeCalculator (arithmetic AST parsing)
- StructuredPropositionVerifier (dependency-anchored evidence extraction)
- GovernedToolRouter (intent classification & tool dispatch)
- OciContainerBackend (fail-closed container execution)

**Core Mandate:** A passing deterministic-tool benchmark is NOT evidence of neural intelligence or mathematical reasoning. We now assess what the native 1.5M parameter ChakrMicro transformer model can actually learn, retain, and generalize when trained on a controlled corpus.

---

## 2. Evaluation Categories & Metrics

The benchmark tests 5 core capabilities with strict partition separation (Train, Val, Held-Out Test):

| Capability Track | Evaluation Tasks | Target Behavior | Primary Metric | Baseline Comparison |
|:---|:---:|:---|:---:|:---|
| **Track 1: Language Continuation & Syntax** | 30 | Syntactic closure of technical sentences & code blocks | Cross-Entropy, PPL, Top-1 Acc | Unigram & Bigram language model |
| **Track 2: Arithmetic Pattern Continuation** | 30 | Exact continuation of arithmetic identity sequences | Exact numeric match | Deterministic calculator (reference) |
| **Track 3: Code Pattern Completion** | 30 | Python function header & variable assignment completion | Exact syntactic match & syntax valid | Random token baseline |
| **Track 4: Compositional Generalization** | 30 | Unseen combinations of rules taught in training | Exact-match on held-out compositions | Memorization threshold |
| **Track 5: Controlled Factual Learning** | 30 | Knowledge learned from synthetic micro-corpus vs unlearned | Extractive exact-match & PPL | Canonical untrained baseline |

Total tasks evaluated: **150 tasks** (held-out test partition).

---

## 3. Strict Non-Contamination & Partitioning Rules
1. **Rule-Based Split:** Partitions are divided by underlying operational rule and template family, not random shuffle of near-identical sentences.
2. **Held-Out Compositional Rules:** The training set teaches $A \rightarrow B$ and $B \rightarrow C$; the held-out test set tests $A \rightarrow C$ (compositional reasoning), ensuring zero lexical overlap.
3. **No Dynamic Tuning:** No prompt engineering, sampling temperature tweaking, or hyperparameter searching is permitted against the held-out test set.
4. **Standalone Execution:** Evaluation runs strictly through `model.forward()` and greedy autoregressive token generation (`torch.argmax(logits)`). No deterministic tools are called.
