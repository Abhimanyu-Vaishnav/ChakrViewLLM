# ChakrView Exp16 v3.1 — Master Pre-Training Audit Instructions

## Objective

Perform a rigorous, non-destructive pre-training audit of Exp16 v3.1. The purpose is to establish dataset integrity, semantic label quality, leakage risks, evaluation validity, and a clear go/no-go recommendation.

**This is an audit-only task. Do not train, fine-tune, run an optimizer, update checkpoints, or change model weights.**

## Hard safety and preservation rules

1. Do not modify or overwrite any canonical model/checkpoint, tokenizer, existing v3/v3.1 dataset, manifest, or protected runtime file.
2. Treat `chakrview/runtime/session.py` as protected. Do not edit it.
3. Do not overwrite or delete any files. Create audit outputs in a new timestamped or uniquely named directory under:
   `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3_1/audit_runs/`
   If creating that directory would conflict with existing data, choose a new unique child directory.
4. Do not run training, evaluation that mutates checkpoints, optimizer steps, or any command that writes model weights.
5. Do not commit, reset, checkout, stash, or otherwise mutate Git state.
6. Before and after the audit, calculate SHA-256 for all protected artifacts and the six v3/v3.1 JSONL datasets. Compare against the expected hashes listed below. Report exact full hashes and any mismatch. A mismatch is a stop condition; investigate read-only and do not attempt to “fix” the file.
7. Distinguish machine-verified facts, heuristic findings, and human-review decisions. Never claim that AI review is official human sign-off.
8. Never silently relabel, rebalance, repair, or replace dataset records. Any proposed corrections must be written to a separate proposal file with original ID, original value, proposed value, rationale, and impact. The original dataset must remain byte-for-byte unchanged.

## Repository and artifacts

Work from the existing ChakrView repository root. Confirm the repository root and current Git status first; do not change Git status.

### Expected SHA-256 values

- v3.1 train: `7b0c636df298a36aa35fb50bc3532a72d9af9bda0d969d50b2a4511a57e1fe14`
- v3.1 validation: `e7cff8b889782e0977ce150c3e2e32c552b2f9a2af7b18e0d24349d19a06fe36`
- v3.1 test: `bc53766b33090743affe553f25092e83432252af6e3ff3053e33dcd183d4d8af`
- v3 train: `d5d8b76871faad8e449d0a59df7834e7db1f37b3c21673743f20ee87249873dc`
- v3 validation: `c9954fb71fd5251065076ad96cfdc78812e2f970eb3cf55dc09c94c38844e95e`
- v3 test: `096d018f357878e89560f982018446c8f09e5143767c1a5425d74bea9560ad10`
- canonical checkpoint: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
- tokenizer vocab: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
- tokenizer merges: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`
- protected `chakrview/runtime/session.py`: `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`

Canonical architecture hash previously recorded:
`c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`

If the architecture file path is discoverable from the repository's existing manifest/audit documentation, hash it read-only and report it. Do not guess the path.

## Required audit phases

### Phase 0 — Safety preflight
- Print repository root and `git status --short` (read-only).
- Verify all expected paths exist before reading them.
- Record UTC/local timestamp and Python version.
- Record whether any relevant process is training; do not stop or alter processes unless explicitly authorized.
- Do not execute a training entrypoint.

### Phase 1 — Integrity and duplicate-ID audit
For each of the six JSONL files:
- Recompute full SHA-256 and compare with expected.
- Parse every non-empty line as JSON and report line numbers for malformed rows.
- Count records, unique IDs, duplicate IDs, missing IDs, and repeated raw lines.
- **Do not load records only into a dictionary before duplicate counting.** Count each ID occurrence from the raw row stream first; dictionaries can conceal duplicates.
- Validate required keys, field types, split labels, categories, and category/field_id consistency.
- Check for train/validation/test ID overlap and exact duplicate prompts/records across splits.
- Report all findings without editing any file.

Expected v3.1 counts from the current project notes: train 300, validation 60, test 120 (total 480). Treat these as expectations to verify, not facts to hard-code into the result.

### Phase 2 — Span, token, and candidate-mask contracts
For every record in all three v3.1 splits:
- Answerable records: verify character span bounds, `prompt[cs:ce] == target_text`, token span bounds, and all target token positions are allowed by `candidate_mask`.
- Abstention records: verify empty target, `[0,0]` character span, `[0,0]` token span, and candidate mask allows the abstention sentinel slot.
- Check `len(candidate_mask) == len(input_ids)`.
- Re-tokenize prompts only if the repository's existing tokenizer interface and preprocessing contract make that a deterministic read-only operation. Do not overwrite stored token IDs.
- Check whether token spans actually align with the target substring under the project's documented token-offset convention. Do not assume token indices are correct merely because they are in bounds.
- Verify that no target span crosses into the assistant query/header or any prohibited region.
- Check all input lengths against the 512-token context limit and report any truncation or boundary edge cases.
- Use existing tests where safe; do not modify test files to make them pass.

### Phase 3 — Label and semantic audit
- Create a complete machine-generated review inventory covering **all 480 v3.1 records**, grouped by split, category, and label.
- For each record, include example ID, category, target/abstention label, prompt, target, rationale/metadata, and flags from automated checks.
- Automated semantic heuristics are triage only. Do not mark them as human-reviewed.
- Generate a prioritized human review queue for every record with contradictions, unsupported targets, ambiguous references, label/prompt mismatch, question-target type mismatch, suspicious shortcuts, or weak category validity.
- Keep the review queue separate from the source datasets.
- Do not mark every record PASS simply because structural tests pass.

#### Mandatory spot-checks from the already supplied 20-example packet

1. `EXP16_V31_TRAIN_DYN_BIND_024` (category 0 abstention): the prompt says sensors `sensor_tr_0_0` and `sensor_tr_0_1` were checked, but then says telemetry dropped before any sensor was engaged. Flag this possible internal contradiction for human review; do not assume PASS.
2. `EXP16_V31_TRAIN_UNCERTAIN_000` (category 8 answerable): the question is “Did power grid feeder bus tr_0 record a transient state during the surge?” but the target is `SURGE_FAULT`. That target is a state value, not a yes/no answer. Flag as a likely question-target type mismatch. Propose either a question rewrite such as “What transient state was recorded ...?” or a compatible yes/no target, but do not alter the original.
3. Category 9 is named `constrained_code_synthesis`, but the supplied examples ask for a function identifier (`sync_packet_0`) rather than synthesizing code. Flag category construct validity. Do not relabel or replace without an explicit, versioned proposal.
4. Category 6 abstention: verify the “high load” clue cannot be resolved from any other context.
5. Category 5 answerable/abstention pair: check whether the target node's state is uniquely supported and whether any distractor language adds ambiguity.
6. Inspect all other examples for similar contradictions or mismatched question/target types, not just these five.

### Phase 4 — Leakage and shortcut audit
For all v3.1 splits:
- Extract queries using the actual documented boundary field/logic, not a guessed substring rule.
- Compare exact and normalized query overlap across splits.
- Compute label distributions globally, per category, per query, per template, and per entity-pattern family.
- Compute train-prior, global-majority, per-category-majority, exact-query-majority, and simple template/entity-pattern baselines. Clearly distinguish in-sample baselines from train-to-validation/test baselines.
- Identify query-only, answer-position, target-string, entity suffix, template, category, field ID, and metadata/rationale shortcuts.
- Check whether answerable and abstention examples for a shared query differ in predictable formatting, context length, number of turns, vocabulary, punctuation, or entity pattern.
- Report whether baselines were evaluated on the correct held-out split. Do not claim leakage is eliminated merely because a measured baseline is weak.
- Do not tune against the test split or use it to make design decisions. Keep test results descriptive and isolated.

### Phase 5 — Benchmark and evaluation contract
- Confirm what the pointer model is intended to predict: span selection versus abstention sentinel.
- Check that target type matches the question type and evaluation metric.
- Document exact-span, answerable accuracy, abstention accuracy, and any other existing project metrics with definitions.
- Verify evaluator/decoder/candidate-mask agreement by reading code and safe unit tests; do not run weight-mutating code.
- Identify whether category-level results would be meaningful given current category examples and class counts.
- Treat category 9 as a construct-validity decision, not just a naming cleanup.
- Report known baseline results only if reproducible from current artifacts; label prior reported figures as historical if not rerun.

### Phase 6 — Tests and safe test execution
- Discover and run only relevant non-training unit/contract tests.
- Before running a test, inspect its code or command to ensure it cannot train, overwrite artifacts, mutate protected files, or make network calls.
- Do not install packages, access hosted model APIs, download pretrained models, or use GPU.
- Record exact commands, exit codes, and test counts.
- If a test might have side effects, skip it and explain why.

### Phase 7 — Outputs
Create a unique audit output directory and write:
1. `AUDIT_SUMMARY.md` — executive summary, go/no-go, verified facts, open issues, exact commands/tests.
2. `INTEGRITY_REPORT.json` — hashes, row counts, duplicate IDs, split overlap, schema failures.
3. `SEMANTIC_REVIEW_QUEUE.csv` — all records or flagged records with reason and priority; no fabricated reviewer decisions.
4. `LEAKAGE_REPORT.md` — methods, baselines, split distinctions, shortcuts, limitations.
5. `PROPOSED_CORRECTIONS.jsonl` — only proposed changes, each with original ID/value, suggested value, rationale, risk, and required human decision. It may be empty if no correction is sufficiently justified.
6. `HUMAN_REVIEW_GUIDE.md` — instructions and focused examples, clearly separating automated flags from reviewer decisions.
7. `POST_AUDIT_HASHES.txt` — post-audit hashes for protected files and datasets.
8. `COMMAND_LOG.txt` — exact commands, timestamps, return codes, and skipped commands with reasons.

Do not write into source dataset paths.

## Go/no-go criteria

Do **not** recommend training if any of these remain:
- protected artifact or source dataset hash mismatch;
- malformed records, duplicate IDs, split leakage, invalid spans, candidate-mask violations, or tokenizer-offset disagreement;
- unresolved high-priority semantic label errors;
- unexplained question/target type mismatch;
- no clear decision on category 9's intended construct;
- evaluator/decoder contract not established;
- human review still pending for flagged semantic examples.

A passing structural audit is not equivalent to semantic approval. A completed AI-assisted audit is not a substitute for human sign-off.

## Final report format

At the end, provide:
1. `GO`, `NO-GO`, or `CONDITIONAL NO-GO` for training.
2. What was actually verified and what was not.
3. Exact number of rows checked in each split and duplicate-ID count.
4. Hash verification status for every protected artifact.
5. Highest-priority semantic issues with example IDs.
6. Leakage baseline results, with in-sample and held-out results clearly separated.
7. Tests run and tests skipped.
8. Output directory and filenames.
9. A short list of human decisions still required.
10. An explicit statement: “No training was run; no source dataset or protected artifact was modified” — only if the logs and hashes support that statement.
