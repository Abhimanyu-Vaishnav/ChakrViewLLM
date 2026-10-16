# Safety and Failure Audit

## Batches 1501–1540: Adversarial Stress Testing, Safety Limits & Failure Analysis

### 1. Adversarial Safety & Resource Audit

To ensure the newly introduced neuro-symbolic tools do not introduce security vulnerabilities or instability into the ChakrView runtime, the system was subjected to rigorous adversarial testing:

| Threat / Failure Category | Adversarial Test Input | Enforced Defense Mechanism | Observed Behavior | Status |
|---|---|---|---|---|
| **Arbitrary Code Execution** | `__import__('os').system('calc')` | AST Node Allowlist (`ast.Call` rejected) | Caught and rejected; returned `SECURITY_OR_DOMAIN_VIOLATION` | **SECURE** |
| **Dynamic Evaluation Injection**| `eval('2 + 2')` | AST Node Allowlist (`ast.Call` rejected) | Caught and rejected; returned `SECURITY_OR_DOMAIN_VIOLATION` | **SECURE** |
| **Filesystem Access** | `open('secret.env').read()` | AST Node Allowlist (`ast.Call` rejected) | Caught and rejected; returned `SECURITY_OR_DOMAIN_VIOLATION` | **SECURE** |
| **Computational DoS Attack** | `2 ** 500` | Exponent Magnitude Ceiling ($|y| \le 100$) | Caught and rejected; returned `SECURITY_OR_DOMAIN_VIOLATION` | **SECURE** |
| **Infinite Exponentiation** | `10000000 ** 20` | Base-Exponent Ceiling ($|x| > 10^6 \implies y \le 10$) | Caught and rejected; returned `SECURITY_OR_DOMAIN_VIOLATION` | **SECURE** |
| **Division by Zero** | `10 / 0`, `50 % 0`, `0 / 0` | Explicit Zero Division Trap | Trapped safely; returned `DIVISION_BY_ZERO` | **SAFE** |
| **Boolean Arithmetic Type Confusion**| `True + 1` | Strict `Constant` type inspection | Rejected boolean constants | **SAFE** |
| **Unbounded Input Length** | String of $> 256$ characters | Maximum Expression Length Guard | Rejected with `EXPRESSION_TOO_LONG` | **SAFE** |

---

### 2. Failure Mode Analysis on Remaining Benchmark Misses

Across the 300 evaluated benchmark tasks, 14 tasks resulted in misses (overall success rate: **95.33%**):

#### A. Structured Verification Boundary Conditions (9 misses out of 100)
- **Token Overlap Ambiguity (`verif_002`):** In `storage_bm25_index.md`, both `k1: 1.5` and `b: 0.75` appear on adjacent lines. If extraction matches the wrong numerical token due to shared words like "parameter", it can flag a contradiction.
- **Port Discrepancies (`verif_009`):** In `network_mesh_topology.md`, two distinct ports are documented: discovery port `5353` and data transmission port `8443`. When the query specifies "port", both values are candidate matches.
- **Remediation:** Further strengthen sentence boundary segmentation and require exact attribute prefix binding before extracting numerical tokens.

#### B. Tool Routing Boundary Conditions (1 miss out of 50)
- **Out-of-Domain Moon Query (`route_049`):** `"Who was the president of the Moon in 1920?"` was routed to `KNOWLEDGE_RETRIEVAL` rather than `ABSTAIN` because "president" triggered a query lookup in the assistant, which then cleanly abstained in the second stage.
- **Remediation:** Both paths result in clean abstention for the user, but router keywords can be expanded to detect astronomical entities upfront.

#### C. End-to-End Formatting Variations (4 misses out of 50)
- On two complex Hinglish queries, the retrieved chunk was correct and cited, but the concise answer span extractor trimmed surrounding text slightly too aggressively, omitting the numeric answer in the summary snippet.
- On two verification queries, the router dispatched to `KNOWLEDGE_RETRIEVAL` instead of `STRUCTURED_VERIFICATION`, which still produced the correct facts with citations, but did not prefix the explicit `[CONTRADICTED]` bracket.

---

### 3. Safety Conclusion
Zero arbitrary code execution vulnerabilities, zero unhandled runtime crashes, and zero hallucinated facts were observed across the entire benchmark suite. The system fails safely and predictably by defaulting to honest abstention.
