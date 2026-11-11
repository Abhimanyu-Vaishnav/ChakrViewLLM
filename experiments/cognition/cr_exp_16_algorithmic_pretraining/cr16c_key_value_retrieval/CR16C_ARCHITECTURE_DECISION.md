# CR-EXP-16C Architecture Decision

## 1. Compliance and Architecture Invariance
During CR-EXP-16C, the canonical ChakrMicro v0.1 architecture was preserved without modification:
- $V = 4,096$
- $d_{\text{model}} = 192$
- $n_{\text{layers}} = 6$
- $n_{\text{heads}} = 6$ ($d_{\text{head}} = 32$)
- $d_{\text{mlp}} = 512$
- $T_{\text{max}} = 512$
- Total Parameters: $3,440,064$
- Zero external memory, zero RAG databases, zero hidden pointer networks added.

## 2. Capacity Assessment
ChakrMicro clearly possesses sufficient architectural capacity to:
- Bind keys to values in working memory.
- Perform selective query matching.
- Overwrite stale values with updated values (99.5% accuracy).
- Filter out conflicting distractor statements (99.7% accuracy).

The failure to generalize to open-vocabulary keys/values is an **exposure and training objective limitation**, not a parameter bottleneck.

## 3. Formal Architectural Decision
- **PRESERVE CANONICAL CHAKRMICRO ARCHITECTURE:** No parameter expansion or layer modifications are warranted at this stage.
- **RECOMMEND MULTI-TASK BALANCED TRAINING:** Rather than expanding width or depth, subsequent experiments must co-train copying and retrieval in a unified mixture to prevent catastrophic forgetting.

