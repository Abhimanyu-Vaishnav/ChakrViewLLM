# CBR-SFT-04 Scorer Comparison & Diagnostic Analysis

## 1. Motivation

In prior experiments (CBR-SFT-02, CBR-SFT-03A, and CBR-SFT-03B), evaluation relied heavily on brittle word-boundary regular expressions (`\bconcept\b`). While objective and fast, this approach introduced systematic evaluation artifacts:
1. **False Negatives via Punctuation:** Responses ending with periods, slashes, or hyphens (e.g., `10.0.4.12.` or `/var/log/audit.log`) frequently failed word-boundary assertions depending on regex engine parsing of dot characters in IP addresses and paths.
2. **False Negatives via Semantic Paraphrasing:** In abstention questions, when a model truthfully replied *"That information is not provided in the context."*, a regex looking strictly for `\bnot provided\b` could fail if the model generated *"The context does not state the administrator password."*
3. **False Positives via Inappropriate Blends:** A regex looking for `\bmorning\b` would mark a hallucinated blend like *"Good morning! Have a pleasant afternoon"* as passing both morning and afternoon tests simultaneously.

To resolve these biases without breaking historical metric continuity, CBR-SFT-04 deploys **Dual Scoring**.

---

## 2. Scorer Comparison Matrix

| Evaluation Dimension | Legacy Regex Scorer | Semantic / Task-Specific Scorer | Impact / Remediation |
| :--- | :--- | :--- | :--- |
| **IP Addresses & Paths** | Fails or behaves inconsistently on dotted notation (`192.168.1.1`) due to regex `\b` definition bordering dots. | Normalizes delimiters and tests exact token sequence containment. | Eliminates spurious failures on network and filesystem grounding tasks. |
| **Abstention Responses** | Rigidly requires exact keyword substring (`"not provided"`). | Evaluates semantic abstention clusters (`"does not specify"`, `"cannot determine"`, `"not provided"`, `"unknown"`). | Eliminates false rejections of valid epistemic refusals. |
| **Temporal Exclusivity** | Permits contradictory greetings if keyword appears anywhere in the output buffer. | Enforces negative checks against opposing temporal terms (e.g., rejects evening greetings containing "morning"). | Prevents gaming of greeting scores by incoherent multi-term generation. |
| **Punctuation & Case** | Case-insensitive regex, but vulnerable to attached punctuation. | Strips leading/trailing punctuation and normalizes internal whitespace. | Robust against tokenization boundary shifts. |

---

## 3. Reporting Requirement

In all CBR-SFT-04 evaluation reports:
- **Both metrics must be reported side by side** (`Legacy Regex Pass Rate` vs `Semantic Pass Rate`).
- Any divergence between the two scores must be explicitly itemized in the experiment audit report with the exact prompt, candidate completion, and rationale for divergence.

