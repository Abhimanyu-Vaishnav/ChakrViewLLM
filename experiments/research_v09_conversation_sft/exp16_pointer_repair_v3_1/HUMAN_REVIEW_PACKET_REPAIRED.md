# ChakrView Exp16 v3.1 — Human Review Packet (Repaired from Source JSONL)

**Review Status**: PENDING HUMAN REVIEW (NO SIGN-OFF PERFORMED)  
**Target Dataset**: `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3_1/`  
**Source File**: `experiments/research_v09_conversation_sft/exp16_pointer_repair_v3_1/data/train_exp16_v3_1.jsonl`  
**Dataset Version**: `v3.1.0`  
**Dataset Hashes**:
- `train_exp16_v3_1.jsonl`: `7b0c636df298a36aa35fb50bc3532a72d9af9bda0d969d50b2a4511a57e1fe14`
- `val_exp16_v3_1.jsonl`: `e7cff8b889782e0977ce150c3e2e32c552b2f9a2af7b18e0d24349d19a06fe36`
- `test_exp16_v3_1.jsonl`: `bc53766b33090743affe553f25092e83432252af6e3ff3053e33dcd183d4d8af`

---

## Instructions for Human Reviewer
For each of the 20 stratified paired examples below:
1. Read the full conversation prompt and verify whether the context evidence genuinely supports the label.
2. Verify that the answerable target is explicitly, uniquely supported by the context and fully captured in the stated character span.
3. Verify that the abstention case genuinely lacks sufficient evidence or is genuinely ambiguous / unanswerable.
4. Record your decision under **Reviewer Decision**: `[ ] PASS`, `[ ] FAIL`, or `[ ] UNCERTAIN`.
5. If rejected or uncertain, record specific notes and proposed corrections.

> [!IMPORTANT]
> Decision fields and reviewer notes are intentionally blank. Antigravity does NOT approve or reject records on behalf of human reviewers.

---

### Category 0: Dynamic Entity Binding (Field ID: 0)

#### Pair 0-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_DYN_BIND_000` (Split: `train`)
- **Category**: `dynamic_entity_binding` | **Field ID**: `0`
- **Target Text**: `sensor_tr_0_0`
- **Char Span**: `[276, 289]` | **Token Span**: `[126, 134]` | **Abstention**: `False`
- **Matched Pair Reference**: Matched with Pair 0-B (`EXP16_V31_TRAIN_DYN_BIND_024`)
- **Semantic Review Requirement**: Reviewer must verify that `sensor_tr_0_0` is explicitly and uniquely grounded in context.
- **Full Prompt**:
```text
Supervisor: Boiler feed pump telemetry confirmed positive head pressure. Factory automation PLC bus 3 validated synchronous pulse timing. We evaluated sensor_tr_0_1 and sensor_tr_0_2 during the test cycle.
Operator: Acknowledged. We routed the primary monitoring telemetry to sensor_tr_0_0 during execution.
Supervisor: Which specific sensor node was selected for monitoring?
Assistant:
```
- **Generator Metadata (Rationale)**: Telemetry routed to sensor_tr_0_0 (slot 0) among distractors.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 0-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_DYN_BIND_024` (Split: `train`)
- **Category**: `dynamic_entity_binding` | **Field ID**: `0`
- **Target Text**: `` (Abstention empty target)
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Matched Pair Reference**: Matched with Pair 0-A (`EXP16_V31_TRAIN_DYN_BIND_000`)
- **Semantic Review Requirement**: Reviewer must verify that context genuinely lacks sufficient evidence to answer, necessitating abstention.
- **Full Prompt**:
```text
Supervisor: Pneumatic conveyor assembly operating within nominal pressure limits. We checked sensor_tr_0_0 and sensor_tr_0_1, but the logging channel failed.
Operator: Telemetry connection dropped before any sensor was engaged.
Supervisor: Which specific sensor node was selected for monitoring?
Assistant:
```
- **Generator Metadata (Rationale)**: Connection dropped; no sensor was engaged.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 1: Turn-Dependent Retrieval (Field ID: 1)

#### Pair 1-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_TURN_RET_000` (Split: `train`)
- **Category**: `turn_dependent_retrieval` | **Field ID**: `1`
- **Target Text**: `1050 MHz`
- **Char Span**: `[229, 237]` | **Token Span**: `[100, 104]` | **Abstention**: `False`
- **Matched Pair Reference**: Matched with Pair 1-B (`EXP16_V31_TRAIN_TURN_RET_024`)
- **Semantic Review Requirement**: Reviewer must verify that `1050 MHz` is explicitly and uniquely grounded in context.
- **Full Prompt**:
```text
Engineer: Pneumatic conveyor assembly operating within nominal pressure limits. Calibrator stage 1 was initialized at 1005 MHz.
Lead: Stage 2 increased output frequency to 1025 MHz.
Senior Tech: Final stage adjusted frequency to 1050 MHz before lock.
Auditor: What is the latest locked operating frequency setting?
Assistant:
```
- **Generator Metadata (Rationale)**: Latest frequency adjustment locked at 1050 MHz in turn 3.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 1-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_TURN_RET_024` (Split: `train`)
- **Category**: `turn_dependent_retrieval` | **Field ID**: `1`
- **Target Text**: `` (Abstention empty target)
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Matched Pair Reference**: Matched with Pair 1-A (`EXP16_V31_TRAIN_TURN_RET_000`)
- **Semantic Review Requirement**: Reviewer must verify that context genuinely lacks sufficient evidence to answer, necessitating abstention.
- **Full Prompt**:
```text
Engineer: Pneumatic conveyor assembly operating within nominal pressure limits. Calibrator stage 1 was initialized at 1005 MHz, but calibration was aborted.
Auditor: What is the latest locked operating frequency setting?
Assistant:
```
- **Generator Metadata (Rationale)**: Calibration aborted; no frequency was locked.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 2: Anaphora Reference Resolution (Field ID: 2)

#### Pair 2-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_ANAPHORA_015` (Split: `train`)
- **Category**: `anaphora_reference_resolution` | **Field ID**: `2`
- **Target Text**: `Cluster_tr_Beta_0`
- **Char Span**: `[212, 229]` | **Token Span**: `[87, 97]` | **Abstention**: `False`
- **Matched Pair Reference**: Matched with Pair 2-B (`EXP16_V31_TRAIN_ANAPHORA_000`)
- **Semantic Review Requirement**: Reviewer must verify that `Cluster_tr_Beta_0` is explicitly and uniquely grounded in context.
- **Full Prompt**:
```text
Admin: Industrial turbine hall reports stable rotation velocity. Assembly line robotic arm controllers synchronized via fieldbus. We deployed the compute cluster on Cluster_tr_Alpha_0 and the database cluster on Cluster_tr_Beta_0 in rack 2.
DevOps: The database cluster completed the assigned task successfully.
Manager: Which cluster completed the assigned task?
Assistant:
```
- **Generator Metadata (Rationale)**: Descriptive anaphora 'the database cluster' uniquely resolves to Cluster_tr_Beta_0.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 2-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_ANAPHORA_000` (Split: `train`)
- **Category**: `anaphora_reference_resolution` | **Field ID**: `2`
- **Target Text**: `` (Abstention empty target)
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Matched Pair Reference**: Matched with Pair 2-A (`EXP16_V31_TRAIN_ANAPHORA_015`)
- **Semantic Review Requirement**: Reviewer must verify that context genuinely lacks sufficient evidence to answer, necessitating abstention.
- **Full Prompt**:
```text
Admin: Factory automation PLC bus 3 validated synchronous pulse timing. Industrial turbine hall reports stable rotation velocity. Both Cluster_tr_Alpha_0 and Cluster_tr_Beta_0 processed the workload batches.
DevOps: It completed the assigned task after encountering a socket retry.
Manager: Which cluster completed the assigned task?
Assistant:
```
- **Generator Metadata (Rationale)**: Pronoun 'It' is ambiguous between Cluster_tr_Alpha_0 and Cluster_tr_Beta_0.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 3: Superseded Instruction Adherence (Field ID: 3)

#### Pair 3-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_SUPERSEDED_000` (Split: `train`)
- **Category**: `superseded_instruction_adherence` | **Field ID**: `3`
- **Target Text**: `Parquet`
- **Char Span**: `[244, 251]` | **Token Span**: `[109, 111]` | **Abstention**: `False`
- **Matched Pair Reference**: Matched with Pair 3-B (`EXP16_V31_TRAIN_SUPERSEDED_024`)
- **Semantic Review Requirement**: Reviewer must verify that `Parquet` is explicitly and uniquely grounded in context.
- **Full Prompt**:
```text
Client: Factory automation PLC bus 3 validated synchronous pulse timing. Industrial turbine hall reports stable rotation velocity. Generate the output in YAML format.
Architect: Correction: do not use YAML. All downstream pipelines now require Parquet instead.
Operator: What is the current mandatory serialization format?
Assistant:
```
- **Generator Metadata (Rationale)**: Architect instruction explicitly supersedes YAML with Parquet.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 3-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_SUPERSEDED_024` (Split: `train`)
- **Category**: `superseded_instruction_adherence` | **Field ID**: `3`
- **Target Text**: `` (Abstention empty target)
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Matched Pair Reference**: Matched with Pair 3-A (`EXP16_V31_TRAIN_SUPERSEDED_000`)
- **Semantic Review Requirement**: Reviewer must verify that context genuinely lacks sufficient evidence to answer, necessitating abstention.
- **Full Prompt**:
```text
Client: Factory automation PLC bus 3 validated synchronous pulse timing. Transformer relay bank 4 passed voltage insulation check. Generate the output in TOML format.
Architect: Cancel that export request entirely; do not serialize anything yet.
Operator: What is the current mandatory serialization format?
Assistant:
```
- **Generator Metadata (Rationale)**: Export request was cancelled without replacement.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 4: User Correction Handling (Field ID: 4)

#### Pair 4-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_CORRECTION_000` (Split: `train`)
- **Category**: `user_correction_handling` | **Field ID**: `4`
- **Target Text**: `node_tr_sec_0`
- **Char Span**: `[255, 268]` | **Token Span**: `[121, 128]` | **Abstention**: `False`
- **Matched Pair Reference**: Matched with Pair 4-B (`EXP16_V31_TRAIN_CORRECTION_024`)
- **Semantic Review Requirement**: Reviewer must verify that `node_tr_sec_0` is explicitly and uniquely grounded in context.
- **Full Prompt**:
```text
Dispatcher: Boiler feed pump telemetry confirmed positive head pressure. Factory automation PLC bus 3 validated synchronous pulse timing. Assign incoming telemetry payload to node_tr_pri_0.
Dispatcher: Wait, that was wrong. Route the telemetry payload to node_tr_sec_0 immediately.
Routing Agent: Which destination node is the payload assigned to?
Assistant:
```
- **Generator Metadata (Rationale)**: User explicitly corrects assignment from node_tr_pri_0 to node_tr_sec_0.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 4-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_CORRECTION_024` (Split: `train`)
- **Category**: `user_correction_handling` | **Field ID**: `4`
- **Target Text**: `` (Abstention empty target)
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Matched Pair Reference**: Matched with Pair 4-A (`EXP16_V31_TRAIN_CORRECTION_000`)
- **Semantic Review Requirement**: Reviewer must verify that context genuinely lacks sufficient evidence to answer, necessitating abstention.
- **Full Prompt**:
```text
Dispatcher: Pneumatic conveyor assembly operating within nominal pressure limits. Transformer relay bank 4 passed voltage insulation check. Assign incoming telemetry payload to node_tr_pri_0.
Dispatcher: Stop dispatch immediately. Cancel all active destination node assignments until further research.
Routing Agent: Which destination node is the payload assigned to?
Assistant:
```
- **Generator Metadata (Rationale)**: Assignment cancelled with no replacement target.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 5: Distractor Resistance (Field ID: 5)

#### Pair 5-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_DISTRACTOR_000` (Split: `train`)
- **Category**: `distractor_resistance` | **Field ID**: `5`
- **Target Text**: `READY`
- **Char Span**: `[263, 268]` | **Token Span**: `[126, 130]` | **Abstention**: `False`
- **Matched Pair Reference**: Matched with Pair 5-B (`EXP16_V31_TRAIN_DISTRACTOR_024`)
- **Semantic Review Requirement**: Reviewer must verify that `READY` is explicitly and uniquely grounded in context.
- **Full Prompt**:
```text
Health Monitor: Transformer relay bank 4 passed voltage insulation check. We checked multiple systems. node_tr_aux_a_0 is MAINTENANCE and node_tr_aux_b_0 is OFFLINE.
Technician: Confirmed. Meanwhile, primary system node_tr_target_0 passed diagnostics and remains READY right now.
Auditor: What is the operating state of node_tr_target_0?
Assistant:
```
- **Generator Metadata (Rationale)**: node_tr_target_0 is READY, ignoring node_tr_aux_a_0 and node_tr_aux_b_0.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 5-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_DISTRACTOR_024` (Split: `train`)
- **Category**: `distractor_resistance` | **Field ID**: `5`
- **Target Text**: `` (Abstention empty target)
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Matched Pair Reference**: Matched with Pair 5-A (`EXP16_V31_TRAIN_DISTRACTOR_000`)
- **Semantic Review Requirement**: Reviewer must verify that context genuinely lacks sufficient evidence to answer, necessitating abstention.
- **Full Prompt**:
```text
Health Monitor: Generator exciter current calibrated to standard excitation curves. Pneumatic conveyor assembly operating within nominal pressure limits. We checked multiple auxiliary lines. node_tr_aux_a_0 is OVERHEATED and node_tr_aux_b_0 is MAINTENANCE.
Auditor: What is the operating state of node_tr_target_0?
Assistant:
```
- **Generator Metadata (Rationale)**: node_tr_target_0 was not checked or mentioned in telemetry.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 6: Ambiguity Clarification (Field ID: 6)

#### Pair 6-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_AMBIGUITY_015` (Split: `train`)
- **Category**: `ambiguity_clarification` | **Field ID**: `6`
- **Target Text**: `pod_tr_emerald_0`
- **Char Span**: `[292, 308]` | **Token Span**: `[127, 136]` | **Abstention**: `False`
- **Matched Pair Reference**: Matched with Pair 6-B (`EXP16_V31_TRAIN_AMBIGUITY_000`)
- **Semantic Review Requirement**: Reviewer must verify that `pod_tr_emerald_0` is explicitly and uniquely grounded in context.
- **Full Prompt**:
```text
Operator: Industrial turbine hall reports stable rotation velocity. Factory automation PLC bus 3 validated synchronous pulse timing. We need clarification on whether to reboot pod_tr_emerald_0 or pod_tr_sapphire_0 today.
Client: After checking the incident log, please proceed with rebooting pod_tr_emerald_0 immediately.
Engineer: Which specific pod was selected for reboot?
Assistant:
```
- **Generator Metadata (Rationale)**: Client explicitly clarifies pod_tr_emerald_0 as the reboot target.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 6-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_AMBIGUITY_000` (Split: `train`)
- **Category**: `ambiguity_clarification` | **Field ID**: `6`
- **Target Text**: `` (Abstention empty target)
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Matched Pair Reference**: Matched with Pair 6-A (`EXP16_V31_TRAIN_AMBIGUITY_015`)
- **Semantic Review Requirement**: Reviewer must verify that context genuinely lacks sufficient evidence to answer, necessitating abstention.
- **Full Prompt**:
```text
Operator: Transformer relay bank 4 passed voltage insulation check. Pneumatic conveyor assembly operating within nominal pressure limits. Reboot the active container pod now.
Support: Both pod_tr_emerald_0 and pod_tr_sapphire_0 are running on this host.
Support: Please specify which pod requires the reboot.
Operator: The container with high load.
Engineer: Which specific pod was selected for reboot?
Assistant:
```
- **Generator Metadata (Rationale)**: Load metrics missing; pod identity cannot be disambiguated.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 7: Missing Context Abstention (Field ID: 7)

#### Pair 7-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_MISSING_000` (Split: `train`)
- **Category**: `missing_context_abstention` | **Field ID**: `7`
- **Target Text**: `10.5 ohms`
- **Char Span**: `[155, 164]` | **Token Span**: `[66, 71]` | **Abstention**: `False`
- **Matched Pair Reference**: Matched with Pair 7-B (`EXP16_V31_TRAIN_MISSING_024`)
- **Semantic Review Requirement**: Reviewer must verify that `10.5 ohms` is explicitly and uniquely grounded in context.
- **Full Prompt**:
```text
User: Boiler feed pump telemetry confirmed positive head pressure. Diagnostic telemetry for generator unit unit_tr_0 recorded stator winding resistance as 10.5 ohms during testing.
Technician: Confirmed recorded reading for unit_tr_0.
User: What is the stator winding resistance on generator unit unit_tr_0?
Assistant:
```
- **Generator Metadata (Rationale)**: Context explicitly states stator winding resistance of unit_tr_0 is 10.5 ohms.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 7-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_MISSING_024` (Split: `train`)
- **Category**: `missing_context_abstention` | **Field ID**: `7`
- **Target Text**: `` (Abstention empty target)
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Matched Pair Reference**: Matched with Pair 7-A (`EXP16_V31_TRAIN_MISSING_000`)
- **Semantic Review Requirement**: Reviewer must verify that context genuinely lacks sufficient evidence to answer, necessitating abstention.
- **Full Prompt**:
```text
User: Assembly line robotic arm controllers synchronized via fieldbus. Generator exciter current calibrated to standard excitation curves. All auxiliary telemetry lines report standard baseline dissipation.
User: What is the stator winding resistance on generator unit unit_tr_0?
Assistant:
```
- **Generator Metadata (Rationale)**: Query component unit_tr_0 is completely absent from context.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 8: Factual Uncertainty Handling (Field ID: 8)

#### Pair 8-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_UNCERTAIN_000` (Split: `train`)
- **Category**: `factual_uncertainty_handling` | **Field ID**: `8`
- **Target Text**: `SURGE_FAULT`
- **Char Span**: `[269, 280]` | **Token Span**: `[113, 123]` | **Abstention**: `False`
- **Matched Pair Reference**: Matched with Pair 8-B (`EXP16_V31_TRAIN_UNCERTAIN_024`)
- **Semantic Review Requirement**: Reviewer must verify that `SURGE_FAULT` is explicitly and uniquely grounded in context.
- **Full Prompt**:
```text
Risk Analyst: Industrial turbine hall reports stable rotation velocity. Generator exciter current calibrated to standard excitation curves. Historical surge incident log audited for power grid feeder bus tr_0.
Auditor: The recorded transient state during the surge was SURGE_FAULT during testing.
Analyst: Did power grid feeder bus tr_0 record a transient state during the surge?
Assistant:
```
- **Generator Metadata (Rationale)**: Verifiable historical log confirms state was SURGE_FAULT.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 8-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_UNCERTAIN_024` (Split: `train`)
- **Category**: `factual_uncertainty_handling` | **Field ID**: `8`
- **Target Text**: `` (Abstention empty target)
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Matched Pair Reference**: Matched with Pair 8-A (`EXP16_V31_TRAIN_UNCERTAIN_000`)
- **Semantic Review Requirement**: Reviewer must verify that context genuinely lacks sufficient evidence to answer, necessitating abstention.
- **Full Prompt**:
```text
Risk Analyst: Transformer relay bank 4 passed voltage insulation check. Generator exciter current calibrated to standard excitation curves. Reviewing general operational telemetry logs without surge instrumentation.
Analyst: Did power grid feeder bus tr_0 record a transient state during the surge?
Assistant:
```
- **Generator Metadata (Rationale)**: Surge instrumentation absent; transient state during surge cannot be verified.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---

### Category 9: Constrained Code Synthesis (Field ID: 9)

#### Pair 9-A: Answerable Example
- **Example ID**: `EXP16_V31_TRAIN_CODE_000` (Split: `train`)
- **Category**: `constrained_code_synthesis` | **Field ID**: `9`
- **Target Text**: `sync_packet_0`
- **Char Span**: `[256, 269]` | **Token Span**: `[95, 103]` | **Abstention**: `False`
- **Matched Pair Reference**: Matched with Pair 9-B (`EXP16_V31_TRAIN_CODE_024`)
- **Semantic Review Requirement**: Reviewer must verify that `sync_packet_0` is explicitly and uniquely grounded in context.
- **Full Prompt**:
```text
Developer: Factory automation PLC bus 3 validated synchronous pulse timing. Assembly line robotic arm controllers synchronized via fieldbus. We are designing the cryptographic integrity verification module.
Tech Lead: The validation function must be named sync_packet_0 in accordance with our API specification.
Developer: What is the required function identifier for the routine?
Assistant:
```
- **Generator Metadata (Rationale)**: API specification explicitly mandates function name sync_packet_0.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

#### Pair 9-B: Abstention Example (Matched Query)
- **Example ID**: `EXP16_V31_TRAIN_CODE_024` (Split: `train`)
- **Category**: `constrained_code_synthesis` | **Field ID**: `9`
- **Target Text**: `` (Abstention empty target)
- **Char Span**: `[0, 0]` | **Token Span**: `[0, 0]` | **Abstention**: `True`
- **Matched Pair Reference**: Matched with Pair 9-A (`EXP16_V31_TRAIN_CODE_000`)
- **Semantic Review Requirement**: Reviewer must verify that context genuinely lacks sufficient evidence to answer, necessitating abstention.
- **Full Prompt**:
```text
Developer: Industrial turbine hall reports stable rotation velocity. We need to implement a data validation helper function.
Tech Lead: I will decide on the function naming convention next week.
Developer: What is the required function identifier for the routine?
Assistant:
```
- **Generator Metadata (Rationale)**: No function name approved yet.
- **Reviewer Decision**: `[ ] PASS  |  [ ] FAIL  |  [ ] UNCERTAIN`
- **Reviewer Notes / Proposed Correction**: __________________________________________________

---
