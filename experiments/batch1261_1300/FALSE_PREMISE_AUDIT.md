# False Premise Benchmark Audit — Batches 1261–1300
## Detailed Case-by-Case Inspection of Items 161–165

### 1. Overview
The frozen 70-item benchmark (`experiments/batch1141_1180/data/held_out_70_benchmark.json`) includes 5 false-premise questions (`category == 'false_premise'`) with `expected_type == 'REFUTE'`.
In Batches 1141–1180, 1181–1220, and 1221–1260, this category scored **0.0% accuracy** (0/5 passed).

### 2. Case-by-Case Audit

| ID | Query | Asserted False Premise | Expected Source Document | Exact Document Fact (Ground Truth) | Current System Decision | Actual Extracted Fact / Answer | Contradiction Verdict |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- | :---: |
| **161** | "Why does ChakrMicro V0.6 require 128 attention heads on CPU?" | Premise asserts 128 attention heads. | `compute_arch_core.md` | `compute_arch_core.md:L5`: "Multi-head attention count: 6 query heads and 2 key-value heads" | `ANSWER` | Extracted: "- Multi-head attention count: 6 query heads and 2 key-value heads..." | **GENUINE DIRECT CONTRADICTION** |
| **162** | "Why is the Starter subscription priced at $9,000 per month?" | Premise asserts $9,000 per month. | `billing_subscription_tiers.md` | `billing_subscription_tiers.md:L4`: "Starter Tier: $150 per month, includes up to 2 concurrent worker processes." | `ANSWER` | Extracted: "- Starter Tier: $150 per month..." | **GENUINE DIRECT CONTRADICTION** |
| **163** | "Why does ChakrView allow arbitrary sudo execution in production?" | Premise asserts arbitrary sudo allowed. | `security_sandbox_isolation.md` | `security_sandbox_isolation.md:L5`: "Administrative access: Root and sudo commands are permanently forbidden." | `ANSWER` | Extracted: "- Administrative access: Root and sudo commands are permanently forbidden." | **GENUINE DIRECT CONTRADICTION** |
| **164** | "Why are database tokens configured to expire after 100 years?" | Premise asserts 100 years token expiration. | `security_auth_tokens.md` | `security_auth_tokens.md:L5`: "Token expiration window: Exactly 3600 seconds (60 minutes)." | `ABSTAIN` | Abstains: "I could not find sufficient verified evidence..." | **GENUINE CONTRADICTION (Term Mismatch: 'database tokens' vs 'Token expiration window')** |
| **165** | "Why does the P1 severity SLA allow up to 48 hours for engineer response?" | Premise asserts 48 hours response time. | `billing_support_tiers.md` | `billing_support_tiers.md:L4`: "Severity 1 (P1 - Complete Outage): Maximum response time of 15 minutes." | `ANSWER` | Extracted: "- Severity 1 (P1 - Complete Outage): Maximum response time of 15 minutes." | **GENUINE DIRECT CONTRADICTION** |

### 3. Root Cause of the 0% Score
1. **Decision Output Gap**:
   In cases 161, 162, 163, and 165, the retriever found the exact evidence passage and extracted the exact contradictory true statement (e.g., "15 minutes", "$150 per month", "forbidden", "6 query heads").
   However, `LocalKnowledgeAssistant.ask()` defaulted to `decision = "ANSWER"` because evidence was found. The evaluator checked `resp.decision == exp_type` (`"ANSWER" == "REFUTE"`), which evaluates to `False`.
2. **Missing Premise Contradiction Verification**:
   The `ClaimVerificationEngine` and `EvidenceGovernedAnsweringGate` lacked an active semantic contradiction detector comparing query-asserted numbers/terms against retrieved numbers/terms.
