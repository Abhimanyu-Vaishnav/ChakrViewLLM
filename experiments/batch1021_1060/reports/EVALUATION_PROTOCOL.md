# Frozen Evaluation Protocol: 50-Item Held-Out Evaluation Set

## 1. Overview and Objective
This protocol defines the formal, reproducible evaluation procedure for the ChakrView Local Knowledge Assistant MVP across Batches 1021–1060.
The objective is to measure practical, verifiable utility:
1. Exact citation validity to real indexed document spans.
2. Factually supported answer correctness.
3. Multi-turn anaphora and follow-up contextual resolution.
4. Honest abstention on unanswerable/out-of-domain queries.
5. Domain refutation on false premises and contradictory questions.

## 2. Dataset Structure and Integrity
The benchmark consists of 50 structured questions divided into 7 balanced categories:
- **Category 1: Direct Factual Lookup (Items 1–10):** Specific facts across architecture, security, billing, and release policies.
- **Category 2: Numeric Extraction & Quantities (Items 11–18):** Parameter counts, context limits, fees, memory constraints, and SLA percentages.
- **Category 3: Negations, Exceptions & Policy Restrictions (Items 19–25):** Forbidden actions, unsupported runtime versions, and penalty thresholds.
- **Category 4: Multi-Turn Follow-Ups (Items 26–32):** Queries requiring prior conversational context (e.g., "How many layers does that model have?").
- **Category 5: Multi-Document & Synthesis (Items 33–38):** Cross-document queries requiring knowledge of distinct system tiers.
- **Category 6: Out-of-Domain & Unindexed Questions (Items 39–46):** Queries requiring honest abstention when evidence is absent from the indexed corpus.
- **Category 7: False Premise & Contradiction Refutation (Items 47–50):** Questions containing invalid assumptions (e.g., capital city of Mars, division by zero) requiring explicit refutation.

All document and benchmark SHA-256 hashes are recorded in `experiments/batch1021_1060/reports/DATASET_MANIFEST.json`.

## 3. Strict Scoring Rules
- **Citation Validity:** A citation is valid IF AND ONLY IF:
  - The cited document exists in the indexed corpus.
  - The character offsets `[start_char:end_char]` accurately bound the referenced text.
- **Citation Support:** The cited passage must directly support the claim made in the answer, rather than merely containing a topical keyword.
- **Answer Correctness:**
  - For answerable queries (`expected_type: "ANSWER"`): The response must return `decision == "ANSWER"` and contain the expected factual substring or equivalent numerical value.
  - For unanswerable queries (`expected_type: "ABSTAIN"`): The response must return `decision == "ABSTAIN"` and explicitly state insufficient evidence.
  - For false premise queries (`expected_type: "REFUTE"`): The response must return `decision == "REFUTE"` and disclose the false premise or contradiction.
- **Failure Conditions:**
  - Any fabricated document name, page number, or character offset is an immediate citation failure.
  - Emitting an assertive answer when evidence is lacking counts as an unsupported claim failure.
