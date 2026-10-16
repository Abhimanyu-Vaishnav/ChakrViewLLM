# Decision Contract — Batches 1261–1300
## Evidence-Governed Decision Policy: ANSWER, REFUTE, and ABSTAIN

### 1. The Tripartite Semantic Contract
In ChakrView's Local Knowledge Assistant, every query processed against indexed local documentation must result in one of three deterministic decision outcomes:

| Decision | Semantic Meaning | Preconditions Required | Output Requirement |
| :--- | :--- | :--- | :--- |
| **`ANSWER`** | Evidence supports responding to the user's information request. | - Query contains non-stopword entity tokens.<br>- Retrieved candidate chunk has overlap with query.<br>- No explicit contradiction between query assertions and evidence. | Factually supported answer span with source document ID, char offsets, and confidence. |
| **`REFUTE`** | The question asserts a materially false or contradictory factual premise, and direct evidence contradicts that premise. | - Query contains an asserted premise (e.g., number, count, duration, feature).<br>- Retrieved candidate from local corpus contains verified opposing fact.<br>- Contradiction is directly verified against citation slice. | Explicit refutation stating the contradictory claim and presenting the true fact with source citation. |
| **`ABSTAIN`** | Available evidence does not support a sufficiently reliable answer or refutation. | - No retrieved candidate chunk satisfies token overlap or confidence threshold.<br>- Evidence is missing or ambiguous. | Standard honest abstention: *"I could not find sufficient verified evidence in the indexed local documents to answer this question."* |

---

### 2. Decision Matrix: 10 Representative Cases

| Case # | Scenario Description | Query Example | Corpus Ground Truth | Expected Decision | Minimum Evidence Requirement |
| :---: | :--- | :--- | :--- | :---: | :--- |
| **1** | Supported factual question | "What is the parameter count of ChakrMicro V0.6?" | `compute_arch_core.md`: "10.8 million trainable float parameters" | `ANSWER` | Top chunk contains `10.8 million` with citation. |
| **2** | Explicitly false numerical premise | "Why does ChakrMicro V0.6 require 128 attention heads on CPU?" | `compute_arch_core.md`: "6 query heads and 2 key-value heads" | `REFUTE` | Citation to `compute_arch_core.md` stating `6 query heads and 2 key-value heads`. |
| **3** | False pricing / duration premise | "Why is the Starter subscription priced at $9,000 per month?" | `billing_subscription_tiers.md`: "Starter tier costs $49 per month" | `REFUTE` | Citation to `billing_subscription_tiers.md` proving `$49 per month`. |
| **4** | False premise with insufficient evidence | "Why are database tokens configured to expire after 100 years?" | Corpus mentions API bearer tokens (15m), but no "database token" document exists. | `ABSTAIN` | Abstain honestly; missing evidence is NOT proof of falsity. |
| **5** | Ambiguous or partially supported premise | "Is ChakrView faster on ARM than x86?" | No benchmark comparisons exist between ARM and x86 in indexed documents. | `ABSTAIN` | Abstain honestly due to insufficient comparative data. |
| **6** | Missing fact / Out-of-domain | "What is the speed of sound in liquid helium at 1 Kelvin?" | No physics docs indexed in technical corpus. | `ABSTAIN` | Unindexed domain must trigger `ABSTAIN` with zero citations. |
| **7** | Negation and paraphrase | "Can the community edition be used commercially without fees?" | `billing_licensing_compliance.md`: "Apache License 2.0" | `ANSWER` | Correctly retrieve Apache 2.0 citation and answer. |
| **8** | Conflicting source versions | "What is the parameter count in V0.7?" | `compute_arch_v07_draft.md`: "24.5 million" (vs V0.6 10.8M) | `ANSWER` | Top-1 chunk must be `compute_arch_v07_draft.md`, not `compute_arch_core.md`. |
| **9** | Follow-up query with pronoun | "What symmetric cipher algorithm is used with it?" (context: TLS 1.3) | `network_encryption_transit.md`: "AES-256-GCM" | `ANSWER` | Anaphora resolution grounds "it" to TLS 1.3; retrieves AES-256-GCM. |
| **10** | Contradicted security claim | "Why does ChakrView allow arbitrary sudo execution in production?" | `security_sandbox_isolation.md`: "unprivileged worker processes; sudo prohibited" | `REFUTE` | Citation to `security_sandbox_isolation.md` refuting sudo execution. |
