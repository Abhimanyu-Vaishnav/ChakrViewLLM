# ChakrView Exp16 v3.1 — Human Semantic Review Guide

## 1. Operating Boundaries & Purpose
This guide accompanies `SEMANTIC_REVIEW_QUEUE.csv` and `HUMAN_REVIEW_PACKET_REPAIRED.md`.
Automated audits have verified byte-level span alignments, token bounds, candidate masking, and cryptographic hashes. However, structural correctness does not guarantee semantic validity.

> [!IMPORTANW]
> Antigravity is an automated coding assistant and CANNOT approve or sign off on human review items. All reviewer checkboxes and decisions in the review packet must remain blank until evaluated by a qualified human annotator.

---

## 2. Review Instructions for Human Annotators

For each record in `SEMANTIC_REVIEW_QUEUE.csv`:
1. **Verify Context Grounding**:
   - Check whether the extracted `target_text` is explicitly stated and uniquely true in the conversation context.
2. **Verify Abstention Truth**:
   - Check whether the context genuinely lacks sufficient evidence to answer, or if there is genuine ambiguity.
3. **Verify Question-Target Construct Validity**:
   - Confirm whether the question asks for the type of entity extracted.

---

## 3. Mandatory Review Targets (High Priority)

1. **EXP16_V31_TRAIN_UNCERTAIN_000** (Category 8 - Factual Uncertainty Handling):
   - **Question**: “Did power grid feeder bus tr_0 record a transient state during the surge?”
   - **Target**: `SURGE_FAULT`
   - **Reviewer Task**: The question is phrased as a polar (yes/no) question, but the target span is a categorical fault name. The reviewer must decide whether this discrepancy is acceptable for pointer extraction or if a question rewrite proposal (see `PROPOSED_CORRECTIONS.jsonl`) should be adopted for v3.2.
2. **EXP16_V31_TRAIN_DYN_BIND_024** (Category 0 - Dynamic Entity Binding Abstention):
   - **Context**: Supervisor says “We checked sensor_tr_0_0 and sensor_tr_0_1, but the logging channel failed.” Operator replies “Telemetry connection dropped before any sensor was engaged.”
   - **Reviewer Task**: Evaluate whether there is an unintended contradiction between “We checked...” and “before any sensor was engaged”, or if this naturally reflects conversational telemetry failure.
3. **Category 9 (`constrained_code_synthesis`) Construct Validity**:
   - **Reviewer Task**: All examples extract a single function identifier name (e.g., `sync_packet_0`) mandated in context. Annotator/Architect must decide whether this represents "constrained code synthesis" or should be formally documented as "identifier retrieval".
